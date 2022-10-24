import importlib
import numpy as np
import os
import random
import mxnet as mx
from  mxnet import nd
import metrics.writer as metrics_writer
from k_means_constrained import KMeansConstrained
from client import Client
from server import TopServer, MiddleServer
from baseline_constants import MODEL_PARAMS
from utils.args import parse_args
from utils.model_utils import read_data
import copy


def main():
    args = parse_args()
    num_rounds = args.num_rounds
    eval_every = args.eval_every
    clients_per_group = args.clients_per_group
    ctx = mx.gpu(args.ctx) if args.ctx >= 0 else mx.cpu()

    param_list=[]
    ratio_list=[]
    gradient_list=[]
    weight_list=[]
    acc_histo=[]
    loss_histo=[]

    log_dir = os.path.join(
        args.log_dir, args.dataset, str(args.log_rank))
    os.makedirs(log_dir, exist_ok=True)
    log_fn = "output.%i.function=%s,a=%f.beta=%i,frequency=%i,round=%i" % (args.log_rank,args.function,args.a,args.beta,args.frequency,num_rounds)
    log_file = os.path.join(log_dir, log_fn)
    log_fp = open(log_file, "w+")

    # Set the random seed, affects client sampling and batching
    random.seed(1 + args.seed)
    np.random.seed(12 + args.seed)
    mx.random.seed(123 + args.seed)

    # Import the client model and server model
    client_path = "%s/client_model.py" % args.dataset
    server_path = "%s/server_model.py" % args.dataset
    if not os.path.exists(client_path) \
            or not os.path.exists(server_path):
        print("Please specify a valid dataset.",
              file=log_fp, flush=True)
        return

    client_path = "%s.client_model" % args.dataset
    server_path = "%s.server_model" % args.dataset
    mod = importlib.import_module(client_path)
    ClientModel = getattr(mod, "ClientModel")
    mod = importlib.import_module(server_path)
    ServerModel = getattr(mod, "ServerModel")
    # sum = sum(p.numel() for p in model.parameters())

    # learning rate, num_classes, and so on
    param_key = "%s.%s" % (args.dataset, args.model)
    model_params = MODEL_PARAMS[param_key]
    if args.lr != -1:
        model_params_list = list(model_params)
        model_params_list[0] = args.lr
        model_params = tuple(model_params_list)
    num_classes = model_params[1]

    # Create the shared client model
    client_model = ClientModel(
        args.seed, args.dataset, args.model, ctx, *model_params)
    w_epoch = list(client_model.get_params())
    sum_p=0

    for p in range(len(w_epoch)):
        sum_p+= len(w_epoch[p].data().reshape(1,-1)[0])
        # print("w_epoch:",w_epoch[p].data().reshape(1,-1))
        print("w_epoch",w_epoch[p])
        print("p:",p,len(w_epoch[p].data().reshape(1,-1)[0]))
        print("sum=",sum_p)
    # Create the shared middle server model
    x=nd.ones((1,1, 28, 28),ctx)
    print("x",x)
    print("client model", client_model.net.summary(x))
    middle_server_model = ServerModel(
        client_model, args.dataset, args.model, num_classes, ctx)
    middle_merged_update = ServerModel(
        None, args.dataset, args.model, num_classes, ctx)
    # Create the top server model
    top_server_model = ServerModel(
        client_model, args.dataset, args.model, num_classes, ctx)
    top_merged_update = ServerModel(
        None, args.dataset, args.model, num_classes, ctx)
    # Create clients
    clients, groups = setup_clients(client_model, args)
    _ = get_clients_info(clients)
    client_ids, client_groups, client_num_samples = _
    print("Total number of clients: %d" % len(clients),
          file=log_fp, flush=True)

    # Measure the global data distribution
    global_dist, _, _ = get_clients_dist(
        clients, display=False, max_num_clients=20, metrics_dir=args.metrics_dir)
    clients_new, scores, clients_new_ids = rank_clients(clients, global_dist)
    print("scores:", scores)
    print("scores:", scores, file=log_fp, flush=True)
    print("ids:", clients_new_ids)
    clients_new, group_list, groups = reassign_clients(clients_new, args)
    # _,_,groups_2=kmeans(clients,args.num_groups)
    _,groups_3=Equal_kmeans(clients,args.num_groups) #开始只有一组，num_groups为1
    print("group_list", group_list)
    print("group_list", group_list, file=log_fp, flush=True)
    # Create middle servers
    middle_servers = setup_middle_servers(
        middle_server_model, middle_merged_update, groups_3,args)
    # [middle_servers[i].brief(log_fp) for i in range(args.num_groups)]
    print("Total number of middle servers: %d" % len(middle_servers),
          file=log_fp, flush=True)

    # Create the top server
    top_server = TopServer(
        top_server_model, top_merged_update, middle_servers,args)
    #Caculate mean distance for Fedrank/random group
    distance=0
    j=0
    i=0
    num_samples=0
    distance_list=[]
    for m1 in middle_servers:
        i=i+1
        j=0
        for m2 in middle_servers:
            j=j+1
            if(i<j):
                distance_t=m1.get_server_scores(m1.clients,m2.get_server_dist(m2.clients),"wasserstein")
                distance_list.append(distance_t)

    print("----------distance_list----------")
    print(distance_list)
    print("length:",len(distance_list))

    distance=0
    j=0
    num_samples=0
    distance_list_2=[]
    selected_clients =random.sample(clients, 12)
    print(selected_clients)
    i=0
    for c1 in selected_clients:
        i = i + 1
        j = 0
        for c2 in selected_clients:
            j = j + 1
            if (i < j):
                distance_t=c1.client_score(base_dist=c2.train_sample_dist/c2.train_sample_dist.sum(), score_cal="wasserstein")
                print(distance_t)
        # num_sample_t=c._num_train_samples
        # distance += distance_t*num_sample_t
        # num_samples += num_sample_t
                distance_list_2.append(distance_t)
    # print("sum_distance:",distance)
    # print("num_samples:",num_samples)
    # distance = distance / num_samples
    print("------Fedavg distance list------")
    print(distance_list_2)
    print("length:", len(distance_list_2))

    # Display initial status
    print("--- Random Initialization ---",
          file=log_fp, flush=True)
    stat_writer_fn = get_stat_writer_function(
        client_ids, client_groups, client_num_samples, args)
    sys_writer_fn = get_sys_writer_function(args)
    acc,loss=print_stats(
        0, top_server, client_num_samples, stat_writer_fn,
        args.use_val_set, log_fp)
    temp1=ClientModel(args.seed, args.dataset, args.model, mx.cpu(), *model_params)
    # Training simulation
    for r in range(1, num_rounds + 1):
        # Select clients
        big_round = int(r / args.frequency - 0.1 + 1)
        top_server.select_mediators(r,num_mediators=1,base_dist=global_dist,sampler="random")
        top_server.select_clients(
            r, clients_per_group, global_dist, display=False,
            metrics_dir=args.metrics_dir)
        _ = get_clients_info(top_server.selected_clients)
        c_ids, c_groups, c_num_samples = _

        print("---Big round %d Round %d of %d: Training %d clients ---"
              % (big_round, r, num_rounds, len(c_ids)),
              file=log_fp, flush=True)



        # Simulate server model training on selected clients' data
        #model_list [p1,p2,]
        sys_metrics,param = top_server.train_model(
            r, args.num_epochs, args.batch_size,param_list,weight_list,log_fp)
        if(param!=0):

            param_list.append(copy.deepcopy(temp1))
            param_list[-1].set_params(param)
            # temp_model.net.collect_params().reset_ctx(mx.cpu())

        #caculate similarity
            if(r>1):
                ratio,gradient=Similarity(param_list[-1],param_list[-2])
                ratio_list.append(ratio)
                gradient_list.append(gradient)
                a = 1.0
                weight_list.append(a)
                c=np.array(weight_list)
                c=c*ratio
                weight_list=c.tolist()

        print("weightlist:",weight_list)
        print("param:",param_list)
        print("gradient",gradient_list)
        print("ratio",ratio_list)
        sys_writer_fn(r, c_ids, sys_metrics, c_groups, c_num_samples)

        # Test model
        if r % eval_every == 0 or r == num_rounds:
            acc,loss=print_stats(
                r, top_server, client_num_samples, stat_writer_fn,
                args.use_val_set, log_fp)
        acc_histo.append(acc)
        loss_histo.append(loss)
        eval_every2=10
        if r%eval_every2==0:
            print("---accuracy history",file=log_fp, flush=True)
            print(acc_histo,file=log_fp, flush=True)
            print("---loss history", file=log_fp, flush=True)
            print(loss_histo,file=log_fp, flush=True)

    # Save the top server model
    top_server.save_model(log_dir)
    log_fp.close()
def  Similarity(model_a,model_b):
    print("model a:",model_a)
    print("model b",model_b)
    list_a=[]
    list_b=[]
    w_a = list(model_a.get_params())
    w_b=list(model_b.get_params())
    sum_p = 0
    for p in range(len(w_a)-4):
        list_a.extend(w_a[p].data().reshape(1, -1)[0].asnumpy().tolist())
        list_b.extend(w_b[p].data().reshape(1, -1)[0].asnumpy().tolist())
    array_a=np.array(list_a)
    array_b = np.array(list_b)
    print("list_a",len(list_a))
    print("list_b",len(list_b))
    a=np.linalg.norm(array_a, ord=1)
    b=np.linalg.norm(array_b, ord=1)
    gradient = array_a - array_b
    c=np.linalg.norm(gradient, ord=1)
    ratio=(1-c/a)
    print("a:",a)
    print("b:",b)
    print("gradient", c)
    return ratio,c
    # print("gradient:",gradient)

def get_array(params):
    w_a2 = list(params)
    list_a2 = []
    for p in range(len(w_a2)):
        list_a2.extend(w_a2[p].data().reshape(1, -1)[0].asnumpy().tolist())
    array_a2 = np.array(list_a2)
    return array_a2


        # print("w_epoch:",w_epoch[p].data().reshape(1,-1))

def reassign_clients(clients,args):
    group_list=[]
    segment=int(len(clients)/args.num_groups)+1
    index=0
    for i in range(1,segment):
        np.random.seed(i)
        L=random.sample(range(1,args.num_groups+1),args.num_groups)
        group_list.extend(L)
    L2=random.sample(range(1,args.num_groups),len(clients)-(segment-1)*args.num_groups)
    group_list.extend(L2)
    for c in clients:
        c.group=group_list[index]
        index+=1
    groups = group_clients(clients, args.num_groups)
    return clients,group_list,groups

def rank_clients(clients,base_dist):
    scores=[]
    ids=[]
    base_dist_ = base_dist / base_dist.sum()
    index=0
    for index in range(0,len(clients)-1):
        for index_2 in range(0,len(clients)-2):
            if clients[index_2].client_score(base_dist_)<clients[index_2+1].client_score(base_dist_):
                b=clients[index_2]
                clients[index_2]=clients[index_2+1]
                clients[index_2+1]=b
    for c in clients:
        scores.append(c.client_score(base_dist_))
        ids.append(c.id)
    return clients,scores,ids
def create_clients(users, groups, train_data, test_data, model, args):
    # Randomly assign a group to each client, if groups are not given
    random.seed(args.seed)
    if len(groups) == 0:
        groups = [random.randint(0, args.num_groups - 1)
                  for _ in users]

    # Instantiate clients
    clients = [Client(args.seed, u, g, train_data[u],
                      test_data[u], model)
               for u, g in zip(users, groups)]

    return clients


def group_clients(clients, num_groups):
    """Collect clients of each group into a list.
    Args:
        clients: List of all client objects.
        num_groups: Number of groups.
    Returns:
        groups: List of clients in each group.
    """
    groups = [[] for _ in range(num_groups)]
    # random.shuffle(clients)
    # random.shuffle(clients)
    for c in clients:
        groups[c.group-1].append(c)
    return groups


def setup_clients(model, args):
    """Load train, test data and instantiate clients.
    Args:
        model: The shared ClientModel object for all clients.
        args: Args entered from the command.
    Returns:
        clients: List of all client objects.
        groups: List of clients in each group.
    """
    eval_set = "test" if not args.use_val_set else "val"
    train_data_dir = os.path.join("data", args.dataset, "data", "train")
    test_data_dir = os.path.join("data", args.dataset, "data", eval_set)

    data = read_data(train_data_dir, test_data_dir)
    users, groups, train_data, test_data = data

    clients = create_clients(
        users, groups, train_data, test_data, model, args)

    groups = group_clients(clients, args.num_groups)

    return clients, groups


def get_clients_info(clients):
    """Returns the ids, groups and num_samples for the given clients.
    Args:
        clients: List of Client objects.
    Returns:
        ids: List of client_ids for the given clients.
        groups: Map of {client_id: group_id} for the given clients.
        num_samples: Map of {client_id: num_samples} for the given
            clients.
    """
    ids = [c.id for c in clients]
    groups = {c.id: c.group for c in clients}
    num_samples = {c.id: c.num_samples for c in clients}
    return ids, groups, num_samples


def get_clients_dist(
        clients, display=False, max_num_clients=20, metrics_dir="metrics"):
    """Return the global data distribution of all clients.
    Args:
        clients: List of Client objects.
        display: Visualize data distribution when set to True.
        max_num_clients: Maximum number of clients to plot.
        metrics_dir: Directory to save metrics files.
    Returns:
        global_dist: List of num samples for each class.
        global_train_dist: List of num samples for each class in train set.
        global_test_dist: List of num samples for each class in test set.
    """
    global_train_dist = sum([c.train_sample_dist for c in clients])
    global_test_dist = sum([c.test_sample_dist for c in clients])
    global_dist = global_train_dist + global_test_dist

    if display:

        try:
            from metrics.visualization_utils import plot_clients_dist

            np.random.seed(0)
            rand_clients = np.random.choice(clients, max_num_clients)
            plot_clients_dist(clients=rand_clients,
                              global_dist=global_dist,
                              global_train_dist=global_train_dist,
                              global_test_dist=global_test_dist,
                              draw_mean=False,
                              metrics_dir=metrics_dir)

        except ModuleNotFoundError:
            pass

    return global_dist, global_train_dist, global_test_dist


def setup_middle_servers(server_model, merged_update, groups,args):
    """Instantiates middle servers based on given ServerModel objects.
    Args:
        server_model: A shared ServerModel object to store the middle
            server model.
        merged_update: A shared ServerModel object to merge updates
            from clients.
        groups: List of clients in each group.
    Returns:
        middle_servers: List of all middle servers.
    """
    num_groups = len(groups)
    middle_servers = [
        MiddleServer(g, server_model, merged_update, groups[g],args)
        for g in range(num_groups)]
    return middle_servers


def get_stat_writer_function(ids, groups, num_samples, args):
    def writer_fn(num_round, metrics, partition):
        metrics_writer.print_metrics(
            num_round, ids, metrics, groups, num_samples,
            partition, args.metrics_dir, "{}_{}_{}_{}_{}_{}_{}_{}".format(
                args.metrics_name, "stat", args.log_rank,args.function,args.a,args.beta,args.frequency,args.num_rounds))

    return writer_fn


def get_sys_writer_function(args):
    def writer_fn(num_round, ids, metrics, groups, num_samples):
        metrics_writer.print_metrics(
            num_round, ids, metrics, groups, num_samples,
            "train", args.metrics_dir, "{}_{}_{}".format(
                args.metrics_name, "sys", args.log_rank))

    return writer_fn


def print_stats(num_round, server, num_samples, writer, use_val_set, log_fp=None):
    train_stat_metrics = server.test_model(set_to_use="train")

    acc,loss=print_metrics(
        train_stat_metrics, num_samples, prefix="train_", log_fp=log_fp)
    writer(num_round, train_stat_metrics, "train")

    eval_set = "test" if not use_val_set else "val"
    test_stat_metrics = server.test_model(set_to_use=eval_set)
    acc,loss=print_metrics(
        test_stat_metrics, num_samples, prefix="{}_".format(eval_set), log_fp=log_fp)
    writer(num_round, test_stat_metrics, eval_set)
    return acc,loss


def print_metrics(metrics, weights, prefix="", log_fp=None):
    """Prints weighted averages of the given metrics.
    Args:
        metrics: Dict with client ids as keys. Each entry is a dict
            with the metrics of that client.
        weights: Dict with client ids as keys. Each entry is the weight
            for that client.
        prefix: String, "train_" or "test_".
        log_fp: File pointer for logs.
    """
    ordered_weights = [weights[c] for c in sorted(weights)]
    metric_names = metrics_writer.get_metrics_names(metrics)
    for metric in metric_names:
        ordered_metric = [metrics[c][metric] for c in sorted(metrics)]
        print("%s: %g, 10th percentile: %g, 50th percentile: %g, 90th percentile %g" \
              % (prefix + metric,
                 np.average(ordered_metric, weights=ordered_weights),
                 np.percentile(ordered_metric, 10),
                 np.percentile(ordered_metric, 50),
                 np.percentile(ordered_metric, 90)),
              file=log_fp, flush=True)

        if(prefix+metric== "test_accuracy"):
            acc=np.average(ordered_metric, weights=ordered_weights)

        elif (prefix == "test_" and metric == "loss"):
            loss=np.average(ordered_metric, weights=ordered_weights)

        else:
            acc=0
            loss=0
    return acc,loss
def kmeans(clients,num_groups):
    k=int(len(clients)/num_groups)#number of mass center
    # print("k==",k)
    #Initialize the mass center
    selected_users=random.sample(clients,int(k))

    n=len(clients[0].train_sample_dist)
    u = np.matrix(np.zeros((k, n)))
    u_index = 0
    cluster_assment = np.matrix(np.zeros((len(clients), 2)))
    for c in selected_users:
        distribution = c.train_sample_dist
        u[u_index, :] =distribution
        u_index+=1

    # print("u:",u)
    cluster_changed = True
    train_loop_counter = 0
    while cluster_changed and train_loop_counter<20:
        cluster_changed = False
        train_loop_counter += 1
        k_cluester_list=[[] for i in range(k)]
        j=0
        for c in clients:
            min_dist = np.inf
            min_dist2= np.inf
            best_cluster_index = -1
            best_cluster_index2 = -1
            for i in range(k):
                dist_diff_ = c.train_sample_dist-u[i,:]
                dist = np.linalg.norm(dist_diff_, ord=1)
                if dist < min_dist and len(k_cluester_list[i])<=num_groups :
                    min_dist2 = dist
                    best_cluster_index2 = i
                    if len(k_cluester_list[i])<num_groups:
                        min_dist = dist
                        best_cluster_index = i

            if best_cluster_index==-1:
                min_dist=min_dist2
                best_cluster_index=best_cluster_index2
            if cluster_assment[j, 0] != best_cluster_index:
                cluster_changed = True
            cluster_assment[j, :] = int(best_cluster_index), min_dist
            k_cluester_list[best_cluster_index].append(c)
            j+=1
            # print("loop:",train_loop_counter)
            # print(k_cluester_list)
            # print(cluster_assment)
            # i=0
        sum_length=0
        for u_index in range(k):
            distribution=0
            length=0
            for c in k_cluester_list[u_index]:
                distribution+=c.train_sample_dist
                length+=1
            sum_length+=length
                # print("length:",length)
                # i += 1
            u[u_index, :] = distribution/length
        # print("u", u)
        # print("sum:",sum_length)
    subgroups = [[] for _ in range(num_groups)]
    index=0
    for subgroup in subgroups:
        for cluster in k_cluester_list:
            subgroup.append(cluster[index])
        index+=1

    print("subgroups:",subgroups)


    return k_cluester_list,cluster_assment,subgroups
def Equal_kmeans(clients,num_groups):
    k = int(len(clients) / num_groups)
    size=int(len(clients)/k)
    selected_clients = random.sample(clients, int(len(clients)/k)*k)
    client_distribution=[c.train_sample_dist.tolist() for c in selected_clients]
    print("client distribution",client_distribution)
    k_cluester_list = [[] for i in range(k)]
    n=len(clients[0].train_sample_dist)
    X=np.array(client_distribution)
    print("lengh:",len(selected_clients))
    print("n_cluesters",k)
    print("size max",size)
    # clf = KMeansConstrained(n_clusters=k,n_init=1,precompute_distances=True,verbose=1)
    clf = KMeansConstrained(
    n_clusters = k,
    size_min = size,
    size_max = size,
    random_state = 0)
    clf.fit_predict(X)
    print("labels:",clf.labels_)
    index=0
    for label in clf.labels_:
        k_cluester_list[label].append(selected_clients[index])
        index+=1
    print("k_cluester_list:",k_cluester_list)
    subgroups = [[] for _ in range(num_groups)]
    index = 0
    for cluster in k_cluester_list:
        print("length:",len(cluster))
    for subgroup in subgroups:
        for cluster in k_cluester_list:
            if(index<len(cluster)):
                subgroup.append(cluster[index])
        index += 1

    return k_cluester_list,subgroups


if __name__ == "__main__":
    main()
