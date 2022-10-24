from abc import ABC, abstractmethod
import mxnet as mx
from mxnet import autograd, nd,init,gluon
import scipy
import numpy as np
import random
import copy
from datetime import datetime
from clustering.equal_groups import EqualGroupsKMeans
from baseline_constants import BYTES_WRITTEN_KEY, \
    BYTES_READ_KEY, LOCAL_COMPUTATIONS_KEY
import math
from k_means_constrained import KMeansConstrained

class Server(ABC):
    def __init__(self, server_model, merged_update):
        self.model = server_model
        self.merged_update = merged_update
        self.total_weight = 0

    @abstractmethod
    def select_clients(self, my_round, clients_per_group, base_dist,
                       display, metrics_dir):
        """Select clients_per_group clients from each group.
        Args:
            my_round: The current training round, used for
                random sampling.
            clients_per_group: Number of clients to select in
                each group.
            base_dist: Real data distribution, usually global_dist.
            display: Visualize data distribution when set to True.
            metrics_dir: Directory to save metrics files.
        Returns:
            selected_clients: List of clients being selected.
            client_info: List of (num_train_samples, num_test_samples)
                of selected clients.
        """
        return None

    @abstractmethod
    def train_model(self, my_round, num_epochs, batch_size):
        """Aggregate models after each synchronization.
        Args:
            my_round: The current training round, used for learning rate
                decay.
            num_epochs: Number of epochs when clients train on data.
            batch_size: Size of train data batches.
        Returns:
            metrics: Dict of metrics returned by the model.
            update: The model after training num_syncs synchronizations.
        """
        return None

    def merge_updates(self, weight, update):
        """Aggregate updates based on their weights.
        Args:
            weight: Weight for this update.
            update: The trained model.
        """
        merged_update_ = list(self.merged_update.get_params())# a list / merged update is a model
        current_update_ = list(update)
        num_params = len(merged_update_)

        self.total_weight += weight

        for p in range(num_params):
            merged_update_[p].set_data(
                merged_update_[p].data() +
                (weight * current_update_[p].data()))

    def update_model(self):
        """Update self.model with averaged merged update."""
        merged_update_ = list(self.merged_update.get_params())
        num_params = len(merged_update_)

        for p in range(num_params):
            merged_update_[p].set_data(
                merged_update_[p].data() / self.total_weight)

        self.model.set_params(self.merged_update.get_params())

        self.total_weight = 0
        self.merged_update.reset_zero()
    # def update_model_copy(self,model_copy,update):
    #     current_update_ = list(update)
    #     num_params = len(current_update_)
    #     model_copy.set_params(update)


    # def update_merged_model(self):
    #     merged_update_ = list(self.merged_update.get_params())
    #     num_params = len(merged_update_)
    #
    #     for p in range(num_params):
    #         merged_update_[p].set_data(
    #             merged_update_[p].data() / self.total_weight)
    #     self

    @abstractmethod
    def test_model(self, set_to_use):
        """Test self.model on all clients.
        Args:
            set_to_use: Dataset to test on, either "train" or "test".
        Returns:
            metrics: Dict of metrics returned by the model.
        """
        return None

    def save_model(self, log_dir):
        """Save self.model to specified directory.
        Args:
            log_dir: Directory to save model file.
        """
        self.model.save(log_dir)


class TopServer(Server):
    def __init__(self, server_model, merged_update, servers,args):
        self.middle_servers = []
        self.register_middle_servers(servers)
        self.selected_clients = []
        self.selected_mediators=[]
        self.biground=0
        self.args=args
        super(TopServer, self).__init__(server_model, merged_update)

    def register_middle_servers(self, servers):
        """Register middle servers.
        Args:
            servers: Middle servers to be registered.
        """
        if type(servers) == MiddleServer:
            servers = [servers]

        self.middle_servers.extend(servers)
    def select_mediators(self,my_round,num_mediators,base_dist,sampler="probalbility"):
        if sampler=="all":
            self.selected_mediators=self.middle_servers
        if sampler=="random":
            self.selected_mediators=self.random_sampling(my_round,num_mediators,base_dist,num_iter=1)
        if sampler=="probalbility":
            self.selected_mediators=self.probalbility_sampling(my_round,num_mediators,base_dist,num_iter=20)
        return self.selected_mediators

    def select_clients(self, my_round, clients_per_group, base_dist=None,
                       display=False, metrics_dir="metrics"):
        """Call middle servers to select clients."""
        selected_info = []
        self.selected_clients = []

        for s in self.selected_mediators:
            _ = s.select_clients(
                my_round, clients_per_group, base_dist,
                display, metrics_dir)
            clients, info = _
            self.selected_clients.extend(clients)
            selected_info.extend(info)

        return selected_info

    def train_model(self, my_round, num_epochs, batch_size,model_list,weight_list,log_fp):
        """Call middle servers to train their models and aggregate
        their updates."""
        sys_metrics = {}
        #calculate big_round

        for s in self.selected_mediators:
            s.set_model(self.model)
            s_sys_metrics, update = s.train_model(
                my_round, num_epochs, batch_size,model_list,self.args.ctx,weight_list,log_fp)
            self.merge_updates(s.num_selected_clients, update)

            sys_metrics.update(s_sys_metrics)
        frequency = self.args.frequency
        big_round=int(my_round / frequency - 0.1 + 1)

        self.update_model()
        print("big_round",big_round)
        print("last_round",self.biground)
        if (self.biground == big_round):
            pram = 0
        else:
            self.biground=big_round
            pram = self.model.get_params()

        #return self.update.cpu()
        return sys_metrics,pram

    def test_model(self, set_to_use="test"):
        """Call middle servers to test their models."""
        metrics = {}

        for middle_server in self.middle_servers:
            middle_server.set_model(self.model)
            s_metrics = middle_server.test_model(set_to_use)
            metrics.update(s_metrics)

        return metrics
    def get_dist_distance(self, clients, base_dist, use_distance="wasserstein"):
        """Return distance of the base distribution and the mean distribution.
        Args:
            clients: List of sampled clients.
            base_dist: Real data distribution, usually global_dist.
            use_distance: Distance metric to be used, could be:
                ["l1", "l2", "cosine", "js", "wasserstein"].
        Returns:
            distance: The distance of the base distribution and the mean
                distribution.
        """
        c_sum_samples_ = sum([c.train_sample_dist for c in clients])
        c_mean_dist_ = c_sum_samples_ / c_sum_samples_.sum()
        base_dist_ = base_dist / base_dist.sum()

        distance = np.inf
        if use_distance == "l1":
            dist_diff_ = c_mean_dist_ - base_dist_
            distance = np.linalg.norm(dist_diff_, ord=1)
        elif use_distance == "l2":
            dist_diff_ = c_mean_dist_ - base_dist_
            distance = np.linalg.norm(dist_diff_, ord=2)
        elif use_distance == "cosine":
            # The cosine distance between vectors u and v is defined as:
            #       1 - dot(u, v) / (norm(u, ord=2) * norm(v, ord=2))
            distance = scipy.spatial.distance.cosine(c_mean_dist_, base_dist_)
        elif use_distance == "js":
            distance = scipy.spatial.distance.jensenshannon(c_mean_dist_, base_dist_)
        elif use_distance == "wasserstein":
            distance = scipy.stats.wasserstein_distance(c_mean_dist_, base_dist_)

        return distance
    def random_sampling(self,my_round,num_mediators,base_dist,num_iter):
        """ randomly sample mediators """
        mediator_score_list = []
        np.random.seed(my_round)
        if num_iter==1:
            selected_mediators=np.random.choice(self.middle_servers,num_mediators,replace=False).tolist()
        elif num_iter>1:
            min_distance=100
            mediator_temp_dist=0
            clients_temp=[]
            for i in range(0,num_iter):
                mediator_tmp=np.random.choice(self.middle_servers,num_mediators,replace=False).tolist()
                for m in mediator_tmp:
                    mediator_temp_dist+=m.get_server_dist(m.clients)
                    clients_temp.extend(m.clients)
                distance_ =self.get_dist_distance(clients_temp, base_dist,use_distance="cosine")
                if distance_<min_distance:
                    selected_mediators=mediator_tmp
                    min_distance=distance_
        return selected_mediators
    def probalbility_sampling(self,my_round,num_mediators,base_dist,num_iter):
        assert num_iter > 1, "Invalid num_iter=%s (num_iter>1)" % num_iter
        np.random.seed(my_round)
        min_distance_ = 100

        prob_ = np.array([1. / len(self.middle_servers)] * len(self.middle_servers))

        while num_iter > 0:
            mediator_temp_dist=0
            clients_temp=[]
            rand_mediator_idx_ = np.random.choice(
                range(len(self.middle_servers)), num_mediators, p=prob_, replace=False)
            mediator_tmp = np.take(self.middle_servers, rand_mediator_idx_).tolist()

            for m in mediator_tmp:
                mediator_temp_dist += m.get_server_dist(m.clients)
                clients_temp.extend(m.clients)
            distance_ = self.get_dist_distance(clients_temp, base_dist, use_distance="cosine")


            if distance_ < min_distance_:
                min_distance_ = distance_
                selected_mediators = mediator_tmp
                # update probability of sampled clients
                prob_[rand_mediator_idx_] += 1. / len(self.middle_servers)
                prob_ /= prob_.sum()

            num_iter -= 1

        return selected_mediators
        # for m in self.middle_servers:
        #     score = 1-m.get_server_scores(m.clients, base_dist, use_distance="cosine")
        #     mediator_score_list.append(score)
        # mediator_scores=np.array(mediator_score_list)
        # p=mediator_scores/(mediator_scores.sum())
        # print("pr=",p)
        # selected_mediators = np.random.choice(
        #     self.middle_servers, num_mediators, replace=False,p=p.ravel()).tolist()
        # #self.selected_mediators=np.random.choice(self.middle_servers,num_mediators,replace=False).tolist()
        # return selected_mediators

class MiddleServer(Server):
    def __init__(self, server_id, server_model, merged_update, clients_in_group,args):
        self.server_id = server_id
        self.clients = []
        self.register_clients(clients_in_group)
        self.selected_clients = []
        self.args=args
        self.full=0
        self.subgroups=[]
        self.selected_subgroups=[]
        self.acc_test_list=[]
        self.loss_test_list=[]
        super(MiddleServer, self).__init__(server_model, merged_update)

    def register_clients(self, clients):
        """Register clients of this middle server.
        Args:
            clients: Clients to be registered.
        """
        if type(clients) is not list:
            clients = [clients]

        self.clients.extend(clients)

    def select_clients(self, my_round, clients_per_group, base_dist=None,
                       display=False, metrics_dir="metrics"):
        """Randomly select part of clients for this round."""
        online_clients = self.online(self.clients)
        num_clients = min(clients_per_group, len(online_clients))
        np.random.seed(my_round)

        self.selected_clients = np.random.choice(
            online_clients, num_clients, replace=False).tolist()
        self.selected_clients=online_clients
        #self.selected_clients=self.sort_clients(online_clients)

        # Measure the distance of base distribution and mean distribution
        # distance = self.get_dist_distance(self.selected_clients, base_dist)
        # print("Dist Distance on Middle Server %i:"
        #       % self.server_id, distance, flush=True)

        # Visualize distributions if needed
        server_dist = self.get_server_dist(self.selected_clients)
        distance = self.get_dist_distance(self.selected_clients, base_dist, use_distance="cosine")
        score = self.get_server_scores(self.selected_clients, base_dist, use_distance="wasserstein")
        print("All clients in Middle Server %i:" % self.server_id, len(self.clients))
        print("Number of clients trained on Middle Server %i:" % self.server_id, len(self.selected_clients))
        print("base distribution", base_dist)
        print("server distribution", server_dist)
        print("Dist Distance on Middle Server %i:"
              % self.server_id, distance, flush=True)
        print("Dist Score on Middle Server %i:"
              % self.server_id, score, flush=True)

        if display:
            from metrics.visualization_utils import plot_clients_dist

            plot_clients_dist(clients=self.selected_clients,
                              global_dist=base_dist,
                              draw_mean=True,
                              metrics_dir=metrics_dir)

        info = [(c.num_train_samples, c.num_test_samples)
                for c in self.selected_clients]

        return self.selected_clients, info

    def sort_clients(self,clients):
        for index in range(0,len(clients)-1):
            for index_2 in range(0,len(clients)-2):
                if clients[index_2].score < clients[index_2 + 1].score:
                    b = clients[index_2]
                    clients[index_2] = clients[index_2 + 1]
                    clients[index_2 + 1] = b
        return clients

    def get_server_dist(self,clients):
        return sum([c.train_sample_dist for c in clients])
    def get_server_sum(self,clients):
        return sum(c.num_train_samples for c in clients)

    def get_server_scores(self, clients, base_dist, use_distance="cosine"):
        c_sum_samples_ = sum([c.train_sample_dist for c in clients])
        c_mean_dist_ = c_sum_samples_ / c_sum_samples_.sum()
        base_dist_ = base_dist / base_dist.sum()
        if use_distance == "cosine":
            scores = scipy.spatial.distance.cosine(c_mean_dist_, base_dist)
        elif use_distance == "wasserstein":
            scores = scipy.stats.wasserstein_distance(c_mean_dist_, base_dist_/base_dist_.sum())
        elif use_distance=="projection":
            scores=np.dot(c_sum_samples_,base_dist) / np.linalg.norm(base_dist)
        return scores
    def get_dist_distance(self, clients, base_dist, use_distance="wasserstein"):
        """Return distance of the base distribution and the mean distribution.
        Args:
            clients: List of sampled clients.
            base_dist: Real data distribution, usually global_dist.
            use_distance: Distance metric to be used, could be:
                ["l1", "l2", "cosine", "js", "wasserstein"].
        Returns:
            distance: The distance of the base distribution and the mean
                distribution.
        """
        c_sum_samples_ = sum([c.train_sample_dist for c in clients])
        c_mean_dist_ = c_sum_samples_ / c_sum_samples_.sum()
        base_dist_ = base_dist / base_dist.sum()

        distance = np.inf
        if use_distance == "l1":
            dist_diff_ = c_mean_dist_ - base_dist_
            distance = np.linalg.norm(dist_diff_, ord=1)
        elif use_distance == "l2":
            dist_diff_ = c_mean_dist_ - base_dist_
            distance = np.linalg.norm(dist_diff_, ord=2)
        elif use_distance == "cosine":
            # The cosine distance between vectors u and v is defined as:
            #       1 - dot(u, v) / (norm(u, ord=2) * norm(v, ord=2))
            distance = scipy.spatial.distance.cosine(c_mean_dist_, base_dist_)
        elif use_distance == "js":
            distance = scipy.spatial.distance.jensenshannon(c_mean_dist_, base_dist_)
        elif use_distance == "wasserstein":
            distance = scipy.stats.wasserstein_distance(c_mean_dist_, base_dist_)

        return distance

    def train_model(self, my_round, num_epochs, batch_size,model_list,ctx,weight_list,log_fp):
        """Train self.model."""
        a=self.args.a
        beta=self.args.beta
        fr=self.args.fr
        grouper=self.args.grouper
        clients = self.selected_clients
        frequency=self.args.frequency
        clients, scores, ids=self.rank_clients(clients)
        clients_score_list1=[]
        clients_score_list2=[]

        for c in clients:
            clients_score_list1.append(c.score)
        # random.shuffle(clients)
        # for c in clients:
        #     clients_score_list2.append(c.score)
        print("clients_score_list1=",clients_score_list1)
        # print("clients_score_list2=",clients_score_list2)
        s_sys_metrics = {
            c.id: {BYTES_WRITTEN_KEY: 0,
                   BYTES_READ_KEY: 0,
                   LOCAL_COMPUTATIONS_KEY: 0}
            for c in clients}

        if my_round<=10000 :#begin to decouping
            big_round=int(my_round/frequency-0.1+1)#change frequency
            print("a=",a)
            if self.args.function=="linear":
                print("Linear")
                num_subgroups=beta*int(a*big_round+1)
                num_last=beta*int(a*(big_round-1)+1)
            if self.args.function=="log":
                print("Log")
                num_subgroups=beta*int(a*np.log(big_round)+1)
                if(big_round-1<=0):
                    num_last=0
                else:
                    num_last=beta*int(a*np.log(big_round-1)+1)
            if self.args.function=="exp":
                print("Exponential")
                num_subgroups=beta*int(a**big_round)
                num_last = beta*int(a**(big_round-1))
            # n0=beta*int(a*(my_round-1)+1)
            # print("n0:",n0)
            if num_subgroups >= len(clients):
                num_subgroups = int(len(clients))


            # k_cluester_list,cluster_assment,subgroups_k= self.kmeans(clients, num_subgroups)
            # a = datetime.now()
            if self.args.mode=="NaiveGSP":
                num_subgroups = beta
                if(my_round==1):
                    subgroups_2 = self.subgroups
                    subgroups_2 = self.random_group(clients, num_subgroups)
                    self.subgroups = subgroups_2
                else:

                    subgroups_2 = self.subgroups
            if self.args.mode == "NaiveICG":
                num_subgroups = beta
                if (my_round == 1):
                    subgroups_2 = self.subgroups
                    k_cluester_list2, subgroups_2 = self.Equal_kmeans(clients, num_subgroups, my_round,streaming=False)
                    self.subgroups = subgroups_2
                else:
                    subgroups_2 = self.subgroups
            if self.args.mode == "FedAVG":
                num_subgroups = int(len(clients))
                if (my_round == 1):
                    subgroups_2 = self.subgroups
                    k_cluester_list2, subgroups_2 = self.Equal_kmeans(clients, num_subgroups, my_round)
                    self.subgroups = subgroups_2
                else:
                    subgroups_2 = self.subgroups
            if self.args.mode == "NaiveICGD":
                subgroups_2=self.subgroups
                num_subgroups = beta
                if grouper=="random":
                    print("random")
                    subgroups_2=self.random_group(clients,num_subgroups)
                else:
                    if (my_round % frequency == 1 or frequency==1):#change frequency
                        k_cluester_list2,subgroups_2=self.Equal_kmeans(clients,num_subgroups,my_round)
                        self.subgroups=subgroups_2
                    else:
                        subgroups_2=self.subgroups
            if self.args.mode == "FedGSP":
                subgroups_2=self.subgroups
                if grouper=="random":
                    print("random")
                    subgroups_2=self.random_group(clients,num_subgroups)
                else:
                    if (my_round % frequency == 1 or frequency==1):#change frequency
                        k_cluester_list2,subgroups_2=self.Equal_kmeans(clients,num_subgroups,my_round)
                        self.subgroups=subgroups_2
                    else:
                        subgroups_2=self.subgroups
            # b = datetime.now()
            print(subgroups_2)
            # print("Time:",(b-a))
            # print("cluester assignment",cluster_assment)
            # print("k_cluester_list",k_cluester_list)

            #num_subgroups=1
            print("num_subgroups:",num_subgroups)
            print("my round:",my_round)
            print("big round:",big_round)
            # subgroups=[[] for _ in range(num_subgroups)]
            # subgroup_list = []
            # segment = int(len(clients) / num_subgroups) + 1
            # index = 0
            # for i in range(1, segment):
            #     np.random.seed(i)
            #     L0 = list(range(1, num_subgroups + 1))
            #     if(i%2==1):
            #         # L = random.sample(range(1, num_subgroups + 1), num_subgroups)
            #         L=L0
            #     else:
            #         L=list(reversed(L0))
            #     subgroup_list.extend(L)
            # rest_len=len(clients) - (segment - 1) * num_subgroups
            # print("rest len：",rest_len)
            # if rest_len!=0:
            #     L2 = random.sample(range(1, num_subgroups+1), rest_len)
            #     subgroup_list.extend(L2)
            # for c in clients:
            #     c_group=subgroup_list[index]
            #     subgroups[c_group-1].append(c)
            #     index+=1
            # print("subgroup_list:",subgroup_list)
            # print("subgroup:",subgroups)
        # s_sys_metrics = {
        #     c.id: {BYTES_WRITTEN_KEY: 0,
        #            BYTES_READ_KEY: 0,
        #            LOCAL_COMPUTATIONS_KEY: 0}
        #     for c in clients}
            m=int(num_subgroups*fr)
            print()
            if(my_round%frequency==1 or frequency==1):#change frequency
                selected_groups=random.sample(subgroups_2,m) ##change a little bit
                self.selected_subgroups=selected_groups
            else:
                selected_groups=self.selected_subgroups
            print("selected groups:",selected_groups)
            print("len selected groups:",len(selected_groups))

            # if (num_subgroups == num_last and my_round % 3 != 0):
            #     self.full+=0
            # else:
            #     self.full+=1
            # print("full times:",self.full)
            index=0
            for g in selected_groups:
                #model_copy=self.model
                model_copy=self.model.get_params()
                model_copy2=self.model
                weight_=0
                random.shuffle(g)
                for c in g:
                    c.model.set_params(model_copy) #set client model
                    if(my_round%frequency!=1 and frequency!=1):#change frequency
                        # if (big_round >= 100 and big_round<200  and my_round % frequency != 1 ):
                        #     print("Original Data", c.data_stored["y"])
                        #     updated_labels = c.update_feature(self.streaming_test, c.model, ctx)
                        #     print("Updated labels:",updated_labels)
                        comp, num_samples, update = c.train(
                            my_round,big_round,num_epochs, batch_size,ctx,finetune=True, model_global=model_copy2,model_list=model_list,weight_list=weight_list)
                    else:
                        comp, num_samples, update = c.train(
                            my_round,big_round,num_epochs, batch_size,ctx, finetune=False, model_global=model_copy2,model_list=model_list,weight_list=weight_list)
                        if (my_round == 1):
                            if(index==0):
                                data=c.test_data
                                data_x = data["x"]
                                data_y = data["y"]
                            else:
                                data = c.test_data
                                # stored_features = np.concatenate((self.new_features.asnumpy(), adapted_features.asnumpy()))
                                # stored_features = nd.array(stored_features).as_in_context(mx.gpu(device_id=ctx))
                                data_x=np.concatenate((data_x.asnumpy(), data["x"].asnumpy()))
                                data_x=nd.array(data_x).as_in_context(mx.gpu(device_id=ctx))
                                data_y = np.concatenate((data_y.asnumpy(), data["y"].asnumpy()))
                                data_y = nd.array(data_y).as_in_context(mx.gpu(device_id=ctx))
                                # data_x.extend(data["x"])
                                # data_y.extend(data["y"])
                            if (len(data_x) < 50):
                                max_size = len(data_x)
                            else:
                                max_size =50
                            self.streaming_test={
                                    "x": data_x[1:max_size],
                                    "y": data_y[1:max_size]}
                            print("streaming data", self.streaming_test)
                            print("max size:", max_size, file=log_fp, flush=True)
                    index+=1
                    model_copy=update
                    weight_+=num_samples
                    #self.merge_updates(num_samples, update)
                    s_sys_metrics[c.id][BYTES_READ_KEY] += c.model.size
                    s_sys_metrics[c.id][BYTES_WRITTEN_KEY] += c.model.size
                    s_sys_metrics[c.id][LOCAL_COMPUTATIONS_KEY] = comp
                    #model_copy=self.update_merged_model()
                self.merge_updates(weight_,update)
            self.update_model()

            # fine-tune  using test-set
            # if(big_round==100 and my_round%frequency==1):
            #     print("====================begin finetune=========================",file=log_fp, flush=True)
            #     finetune_epoch=10
            #     for i in range(finetune_epoch):
            #         comp, update = self.model.train(
            #             self.streaming_test, my_round, num_epochs, batch_size, finetune=False)
            #         # test
            #         metric_t = self.model.test(self.streaming_test)
            #         test_acc = metric_t["accuracy"]
            #         print("fine epoch:",finetune_epoch,"accuracy in test batch:", test_acc,file=log_fp, flush=True)
            #     print("====================END finetune=========================", file=log_fp, flush=True)
            # if (big_round > 100):
            #     #use sematics
            #     comp, update = self.model.train(
            #         self.streaming_test, my_round, num_epochs, batch_size, finetune=True)

















            update = self.model.get_params()
            # metric_t=self.model.test(self.streaming_test)
            # test_acc=metric_t["accuracy"]
            # loss=metric_t["loss"]
            # self.acc_test_list.append(test_acc)
            # self.loss_test_list.append(loss)
            # print("accuracy in test set:", test_acc)
            # print("accuracy in test batch:", self.acc_test_list,file=log_fp, flush=True)
            # print("loss in test batch:", self.loss_test_list, file=log_fp, flush=True)

            return s_sys_metrics, update
        else:
            print("else:",my_round)
            # k=len(clients)
            # random.seed(1)
            # c_seq=random.sample(range(0,len(clients)),len(clients))
            # print("c_seq",c_seq)
            # while k>=1:
            #     c=clients[c_seq[len(clients)-k]]
            #     c.set_model(self.model)
            #     comp, num_samples, update = c.train(
            #               my_round, num_epochs, batch_size)
            #     self.merge_updates(num_samples, update)
            #     s_sys_metrics[c.id][BYTES_READ_KEY] += c.model.size
            #     s_sys_metrics[c.id][BYTES_WRITTEN_KEY] += c.model.size
            #     s_sys_metrics[c.id][LOCAL_COMPUTATIONS_KEY] = comp
            #     k-=1
            #     if my_round <= 300:
            #         self.update_model()
            # if my_round>1200:
            #     self.update_model()
            # update=self.model.get_params()
            # return s_sys_metrics, update

            for c in clients:
                c.set_model(self.model)
                comp, num_samples, update = c.train(
                    my_round, num_epochs, batch_size)
                self.merge_updates(num_samples, update)

                s_sys_metrics[c.id][BYTES_READ_KEY] += c.model.size
                s_sys_metrics[c.id][BYTES_WRITTEN_KEY] += c.model.size
                s_sys_metrics[c.id][LOCAL_COMPUTATIONS_KEY] = comp
                if my_round<=500:
                    self.update_model()
            # if my_round>=120:
            #     self.update_model()
            update = self.model.get_params()
            return s_sys_metrics, update

    def test_model(self, set_to_use="test"):
        """Test self.model on online clients."""
        s_metrics = {}

        for client in self.online(self.clients):
            client.set_model(self.model)
            c_metrics = client.test(set_to_use)
            s_metrics[client.id] = c_metrics

        return s_metrics

    def set_model(self, model):
        """Set the model data to specified model.
        Args:
            model: The specified model.
        """
        self.model.set_params(model.get_params())

    def online(self, clients):
        """Return clients that are online.
        Args:
            clients: List of all clients registered at this
                middle server.
        Returns:
            online_clients: List of all online clients.
        """
        online_clients = clients
        assert len(online_clients) != 0, "No client available."
        return online_clients

    @property
    def num_clients(self):
        """Return the number of all clients registered at this
        middle server."""
        if not hasattr(self, "_num_clients"):
            self._num_clients = len(self.clients)

        return self._num_clients

    @property
    def num_selected_clients(self):
        """Return the number of selected clients."""
        return len(self.selected_clients)

    @property
    def num_samples(self):
        """Return the total number of samples for self.clients."""
        if not hasattr(self, "_num_samples"):
            self._num_samples = sum([c.num_samples for c in self.clients])

        return self._num_samples

    @property
    def num_train_samples(self):
        """Return the total number of train samples for
        self.clients."""
        if not hasattr(self, "_num_train_samples"):
            self._num_train_samples = sum([c.num_train_samples
                                           for c in self.clients])

        return self._num_train_samples

    @property
    def num_test_samples(self):
        """Return the total number of test samples for
        self.clients."""
        if not hasattr(self, "_num_test_samples"):
            self._num_test_samples = sum([c.num_test_samples
                                          for c in self.clients])

        return self._num_test_samples

    @property
    def train_sample_dist(self):
        """Return the distribution of train data for
        self.clients."""
        if not hasattr(self, "_train_sample_dist"):
            self._train_sample_dist = sum([c.train_sample_dist
                                           for c in self.clients])

        return self._train_sample_dist

    @property
    def test_sample_dist(self):
        """Return the distribution of test data for
        self.clients."""
        if not hasattr(self, "_test_sample_dist"):
            self._test_sample_dist = sum([c.test_sample_dist
                                          for c in self.clients])

        return self._test_sample_dist

    @property
    def sample_dist(self):
        """Return the distribution of overall data for
        self.clients."""
        if not hasattr(self, "_sample_dist"):
            self._sample_dist = self.train_sample_dist + self.test_sample_dist

        return self._sample_dist

    def brief(self, log_fp):
        """Briefly summarize the statistics of this middle server"""
        print("[Group %i] Number of clients: %i, number of samples: %i, "
              "number of train samples: %s, number of test samples: %i, "
              % (self.server_id, self.num_clients, self.num_samples,
                 self.num_train_samples, self.num_test_samples),
              file=log_fp, flush=True, end="\n")
        # print("sample distribution:", list(self.sample_dist.astype("int64")))

    def rank_clients(self,clients):
        scores = []
        ids = []
        # base_dist_ = base_dist / base_dist.sum()
        index = 0
        for index in range(0, len(clients) - 1):
            for index_2 in range(0, len(clients) - 2):
                if clients[index_2].score < clients[index_2 + 1].score:
                    b = clients[index_2]
                    clients[index_2] = clients[index_2 + 1]
                    clients[index_2 + 1] = b
        for c in clients:
            scores.append(c.score)
            ids.append(c.id)
        return clients, scores, ids
    # def k_group(self,clients,k):
    def Equal_kmeans(self,clients,num_groups,round,streaming=True):
        ct=streaming
        k = int(len(clients) / num_groups)
        size=int(len(clients)/k)
        selected_clients=random.sample(clients,int(size*k)) # make sure it is divisible
        for c in selected_clients:
            straming_data=c.Get_streaming_data(round)

        client_distribution=[c.train_sample_dist.tolist() for c in selected_clients]
        # print("client distribution",client_distribution)

        k_cluester_list = [[] for i in range(k)]
        n=len(self.train_sample_dist)
        X=np.array(client_distribution)
        print("lengh:",len(selected_clients))
        print("n_clusters",k)
        print("size max",size)
        # clf = KMeansConstrained(n_clusters=k,n_init=1,precompute_distances=True,verbose=1)
        clf = KMeansConstrained(
        n_clusters = k,
        size_min = size,
        size_max = size,
        random_state = 0,
        verbose=1,
        n_init=1)
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
            random.shuffle(cluster)
            print("length:",len(cluster))
        for subgroup in subgroups:
            for cluster in k_cluester_list:
                if(index<len(cluster)):
                    subgroup.append(cluster[index])
            index += 1

        return k_cluester_list,subgroups
    def random_group(self,clients,num_groups):
        # random.shuffle(clients)
        groups = [[] for _ in range(num_groups)]
        i=0
        for c in clients:
            k=i%(num_groups)
            groups[k].append(c)
            i+=1
        return groups
    def kmeans(self,clients,num_groups):
        k=int(len(clients)/num_groups)#number of mass center
        # print("k==",k)
        #Initialize the mass center
        selected_users=random.sample(clients,int(k))
        n=len(self.train_sample_dist)
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










