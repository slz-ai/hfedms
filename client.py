import warnings
import numpy as np
import random
import heapq
import time
from mxnet import nd
import pandas as pd
import scipy
import mxnet as mx
from mxnet import autograd, nd,init,gluon
from mxnet.gluon import nn
import copy
from mxnet.gluon import data as gdata, loss as gloss, utils as gutils
from baseline_constants import ACCURACY_KEY, INPUT_SIZE
from datetime import datetime
from utils.model_utils import batch_data,batch_data2
def get_array(params):
    w_a2 = list(params)
    list_a2 = []
    for p in range(len(w_a2)):
        list_a2.extend(w_a2[p].data().reshape(1, -1)[0].asnumpy().tolist())
    array_a2 = np.array(list_a2)
    return array_a2

class Client:
    def __init__(self, seed, client_id, group, train_data, test_data, model):
        self.seed = seed
        self.id = client_id
        self.group = group
        self._model = model
        self.previous_model=model.net
        self.raw_data=train_data
        self.trained_round=[]
        self.big_round=-1
        self.cali_round=0
        # self.MLP = nn.Sequential()
        # self.MLP.add(nn.Dense(2048, activation='relu'), nn.Dense(100))
        # self.MLP.initialize(init.Normal(sigma=0.01))


        self.Nc=nd.array([0 for _ in range(62)])
        self.Mean_Matrix=nd.zeros((62,100))
        self.Var_Vec=np.zeros(62)

        # print(train_data["x"][1])
        self.train_data = {
            "x": self.process_data(train_data["x"]),
            "y": self.process_data(train_data["y"])
        }
        self.test_data = {
            "x": self.process_data(test_data["x"]),
            "y": self.process_data(test_data["y"])
        }

        avg_list, _, _, self.store_lables, self.store_features = model.get_feature(self.train_data)
        _, _, _, self.new_lables, self.new_features = model.get_feature(self.train_data)
        self.Mean_Matrix=nd.array(avg_list)
        self.data_stored = {"x": self.store_features, "y": self.store_lables}
        self.data_compensated = {"x": self.store_features, "y": self.store_lables}
        self.data_new = {"x": self.new_features, "y": self.new_lables}
        self.data_new_all = {"x": self.new_features, "y": self.new_lables}
        self.streaming_data = self.Get_streaming_data(my_round=0)
        self.previous_data = self.streaming_data
    def Get_Data_in_Previous_Round(self,big_round):
        data = self.train_data
        data_x = data["x"]
        data_y = data["y"]
        seed=big_round+1
        mx.random.seed(seed)
        data_x=mx.random.shuffle(data_x)
        mx.random.seed(seed)
        data_y=mx.random.shuffle(data_y)
        if (len(data_x) < 50):
            max_size = len(data_x)
        else:
            max_size = 50
        shape_aug = gdata.vision.transforms.RandomResizedCrop(
            (28, 28), scale=(0.8, 1), ratio=(0.8, 1.2))
        augs = gdata.vision.transforms.Compose([
            gdata.vision.transforms.RandomBrightness(0.1), shape_aug])

        i = 1
        streaming_data = {
            "x": data_x[1:max_size],
            "y": data_y[1:max_size]
        }
        return streaming_data

    def Get_streaming_data(self,my_round):
        data = self.train_data
        data_x = data["x"]
        data_y = data["y"]
        seed = my_round + 1
        mx.random.seed(seed)
        data_x = mx.random.shuffle(data_x)
        mx.random.seed(seed)
        data_y = mx.random.shuffle(data_y)
        # print(len(data_x))
        if(len(data_x)<50):
            max_size=len(data_x)

        else:
            max_size=50
        shape_aug = gdata.vision.transforms.RandomResizedCrop(
            (28, 28), scale=(0.8, 1), ratio=(0.8, 1.2))
        augs = gdata.vision.transforms.Compose([
            gdata.vision.transforms.RandomBrightness(0.1),shape_aug])

        i=1
        for img in data_x[1:max_size]:
            # print("imagei",data_x[i])
            img=nd.array(img)
            img=img.reshape((28,28,1))
            img=augs(img)
            img=nd.array(img, ctx=self.model.ctx)
            img=img.reshape(-1)
            data_x[i]=img
            i=i+1
            # print("aug",data_x[i])


        streaming_data = {
                "x": data_x[1:max_size],
                "y": data_y[1:max_size]
            }
        if(my_round):
            self.previous_data=self.streaming_data
        self.streaming_data=streaming_data
        return streaming_data

    def update_feature(self,data,model,ctx):
        """
        update client's semantic features
        Args:
            data: The raw data to be extracted
            model:The model used to extract the feature
        """
        mean_list, var_list, num_list, labels, features = model.get_feature(data)
        stored_features = np.concatenate((self.data_stored["x"].asnumpy(), features.asnumpy()))
        stored_features = nd.array(stored_features).as_in_context(mx.gpu(device_id=ctx))
        stored_labels = np.concatenate((self.data_stored["y"].asnumpy(), labels.asnumpy()))
        stored_labels = nd.array(stored_labels).as_in_context(mx.gpu(device_id=ctx))
        self.data_stored = {
            "x": stored_features,
            "y": stored_labels}

        return stored_labels


    def concat_dataset(self,datasetA,datasetB,ctx):
        """To Concat 2 semantic dataset
            Args:
                datasetA
                datasetB
                ctx:device
            return:
                datasetC: new dataset
        """
        feature_A = datasetA["x"]
        feature_B=datasetB["x"]
        label_A=datasetA["y"]
        label_B=datasetB["y"]
        concat_features = np.concatenate((feature_A.asnumpy(), feature_B.asnumpy()))

        concat_features = nd.array(concat_features).as_in_context(mx.gpu(device_id=ctx))
        concat_labels = np.concatenate((label_A.asnumpy(), label_B.asnumpy()))
        concat_labels = nd.array(concat_labels).as_in_context(mx.gpu(device_id=ctx))
        if (ctx == -1):
            concat_features=concat_features.as_in_context(mx.cpu())
            concat_labels=concat_labels.as_in_context(mx.cpu())
        datasetC={"x": concat_features, "y": concat_labels}
        return datasetC,concat_features,concat_labels
    def get_class_mean_matrix(self,dataset):
        temp_list = [[] for i in range(62)]
        avg_list = []
        features = dataset["x"]
        labels=dataset["y"]
        j=0
        f_len = len(features[0])
        for label in labels:
            # print(int(label.asnumpy()[0]))
            temp_list[int(label.asnumpy()[0])].append(features[j])
            j=j+1

        for c in  temp_list:
            # print("temp_list",j,len(c))
            if(len(c)!=0):
                feature_avg=0
                for feature in c:
                    feature_avg+=feature
                feature_avg=(feature_avg/len(c))
            else:
                feature_avg=nd.zeros(f_len)
            avg_list.append(feature_avg.asnumpy())
        return avg_list




    def train(self, my_round,big_round, num_epochs, batch_size,ctx,streaming=True,finetune=False,model_global=None,model_list=None,weight_list=None):
        """Trains on self.model using one batch of train_data.
        Args:
            my_round: The current training round, used for learning rate
                decay.
            num_epochs: Number of epochs when clients train on data.
            batch_size: Size of train data batches.
        Returns:
            comp: Number of FLOPs executed in training process.
            num_samples: Number of train samples on this client.
            update: Trained model params.
        """
        # nd.waitall()
        # a = datetime.now()
        # train的同时，考虑他所处的big_round.
        # 外部读取model_list
        # 取出model,更新，最后反映到update上面去.
        # 根据train_history/权重_dictionary 取出需要的数据切片
        # 模型+数据进行训练
        # 一个循环，每个一个batch_size. 50条现在，150条previous.
        #
        data = self.train_data
        streaming_data = self.streaming_data
        if(big_round!=self.big_round):#new big round
            self.cali_round = 0
            self.big_round=big_round
            self.trained_round.append(big_round)
            print("trained round",self.trained_round)
            print("data_new_all",self.data_new_all)
            #prepare semantic features
            # self.store_features = self.store_features.as_in_context(mx.gpu(device_id=ctx))
            index=0
            distance_list=[]
            # list_temp=list(map(distance_list,zip(range(len(distance_list)), test)))

            #comm A
            # get big dataset
            data_stored,stored_features,stored_labels=self.concat_dataset(self.data_new_all,self.data_compensated,ctx)
            # adapted_features = self.data_compensated["x"]
            # stored_features = np.concatenate((self.new_features.asnumpy(), adapted_features.asnumpy()))
            # stored_features = nd.array(stored_features).as_in_context(mx.gpu(device_id=ctx))
            # stored_labels = np.concatenate((self.new_lables.asnumpy(), self.data_stored["y"].asnumpy()))
            # stored_labels = nd.array(stored_labels).as_in_context(mx.gpu(device_id=ctx))
            self.Mean_Matrix=self.get_class_mean_matrix(data_stored)
            self.Mean_Matrix=nd.array(self.Mean_Matrix).as_in_context(mx.gpu(device_id=ctx))
            if(ctx==-1):
                self.Mean_Matrix=self.Mean_Matrix.as_in_context(mx.cpu())
            print("mean matrix：", self.Mean_Matrix)
            # comm A
            #start
            print("data_store", self.data_stored)
            t1=time.time()
            #get n_c most representive senmatics
            for features in stored_features:
                label = stored_labels[index]
                mean = self.Mean_Matrix[label]
                bias = features - mean
                distance = nd.dot(bias[0], bias[0])
                distance_list.append(distance)
                index += 1
            t2=time.time()
            # print("caculate distance time:",t2-t1)

            tmp = list(map(list,zip(range(len(distance_list)), distance_list)))
            small = sorted(tmp, key=lambda x: x[1], reverse=False)
            small,_=zip(*small)
            small=list(small)
            small=small[:len(distance_list)]
            t3=time.time()
            # print("get small index time:",t3-t2)
            print("min index",small)
            stored_x=stored_features[small]
            stored_y=stored_labels[small]
            random.seed(my_round*11)
            random.shuffle(stored_x)
            random.seed(my_round * 11)
            random.shuffle(stored_y)
            threshold=min(len(stored_y),200)
            print("threshold",threshold)
            self.data_stored = {"x": stored_x[:threshold], "y": stored_y[:threshold]}
            print("data stored after",self.data_stored)
            self.data_new_all={"x": None, "y": None}
            #end

            # self.data_stored = {"x": self.new_features, "y": self.new_lables}

        print("trained_round", self.trained_round)


        #augmentation

        if(streaming==True):
            # if(finetune==True):
            #     comp, update = self.model.train(
            #         previous_data, my_round, num_epochs, batch_size, finetune=finetune,
            #         previous_model=model_global.net)
            if(finetune==False):
                self.model.net.collect_params().reset_ctx(mx.cpu())
                model_global.net.collect_params().reset_ctx(mx.cpu())
                nd.waitall()  #
                begin = time.time()
                comp, update = self.model.train(
                    streaming_data, my_round, num_epochs, batch_size,finetune=finetune,previous_model=model_global.net)
                nd.waitall()  #
                end = time.time()
                print("Time for full training:",end-begin)
                mean_list_new, var_list_new, num_list_new, labels, features_new = self.model.get_feature(streaming_data)
                self.new_features = features_new
                self.new_lables = labels
                self.data_new = {"x": self.new_features, "y": self.new_lables}
                if (self.data_new_all["x"] is None):
                    self.data_new_all = {"x": self.new_features, "y": self.new_lables}
                else:
                    self.data_new_all,_,_ = self.concat_dataset(self.data_new_all, self.data_new, ctx)
                print("data new all after",self.data_new_all)
            #train streaming data
            #commB
            # self.MLP.collect_params().reset_ctx(mx.gpu(device_id=ctx))
            if (finetune == True):
                self.cali_round +=1
                print("cali_round",self.cali_round)
                if (len(self.trained_round) > 1):
                    last_roud=self.trained_round[-2]
                    print("this round",big_round)
                    print("last round",last_roud)
                    #get previous model
                    model_pre=model_list[last_roud]
                    if(ctx!=-1):
                        model_pre.net.collect_params().reset_ctx(mx.gpu(device_id=ctx))
                    nd.waitall()  #
                    begin = time.time()
                    #get "previous semantics"
                    mean_list,var_list,num_list,lables,features_old=model_pre.get_feature(streaming_data)

                    #get "current semantics"
                    mean_list_new,var_list_new,num_list_new,labels,features_new=self.model.get_feature(streaming_data)
                    nd.waitall()  #
                    end = time.time()
                    print("T of extraction and mean:",end-begin)
                    self.new_features = features_new
                    self.new_lables = labels
                    self.data_new = {"x": self.new_features, "y": self.new_lables}
                    print("data new all 0:",self.data_new_all)
                    if(self.data_new_all["x"] is None):
                        self.data_new_all = {"x": self.new_features, "y": self.new_lables}
                    else:
                        self.data_new_all,_,_ = self.concat_dataset(self.data_new_all, self.data_new,ctx)
                    print("data new all after",self.data_new_all)
                    #caculate semantic drift
                    self.Mean_Matrix=(nd.array(mean_list_new)).as_in_context(mx.gpu(device_id=ctx))
                    class_bias=(nd.array(mean_list_new) - nd.array(mean_list)).as_in_context(mx.gpu(device_id=ctx))
                    if(ctx==-1):
                        self.Mean_Matrix=self.Mean_Matrix.as_in_context(mx.cpu())
                        class_bias=class_bias.as_in_context(mx.cpu())
                    model_pre.net.collect_params().reset_ctx(mx.cpu())
                    loss_b = gloss.SoftmaxCrossEntropyLoss()
                    index=0
                    #Compensate semantics
                    print("caliround:",self.cali_round)
                    if(self.cali_round == 1):
                        print("Compensate:true")
                        self.data_compensated=self.data_stored
                        for features in self.data_stored["x"]:
                            label=self.data_stored["y"][index]
                            bias=class_bias[label]
                            adapted_features=features+bias
                            self.data_compensated["x"][index]=adapted_features
                            self.data_compensated["y"][index] = label
                            index+=1
                        nd.waitall()  #
                        end = time.time()
                        print("==============compensate time======================",end-begin)
                    else:
                        nd.waitall()  #
                        end = time.time()
                        print("Compensate:false")
                    trainer = gluon.Trainer(self.model.net.output.collect_params(), 'sgd', {'learning_rate': 0.01})
                    seed = my_round * 13
                    #Construct calibration set
                    adapted_features = self.data_compensated["x"]
                    stored_features = np.concatenate((self.new_features.asnumpy(), adapted_features.asnumpy()))
                    stored_features = nd.array(stored_features).as_in_context(mx.gpu(device_id=ctx))
                    stored_labels = np.concatenate((self.new_lables.asnumpy(), self.data_stored["y"].asnumpy()))
                    stored_labels = nd.array(stored_labels).as_in_context(mx.gpu(device_id=ctx))
                    # data_finetune = {"x": stored_features, "y": stored_labels}
                    data_finetune = {"x": self.new_lables, "y": self.new_lables}
                    # print("data finetuneA",data_finetune)
                    data_finetune,finetune_feature,finetune_label = self.concat_dataset(self.data_new, self.data_compensated,ctx)
                    # print("data finetuneB",data_finetune)
                    # Calibration
                    #cpu test
                    nd.waitall()  #
                    begin1 = time.time()
                    self.model.net.collect_params().reset_ctx(mx.cpu())
                    for i in range(3):
                        nd.waitall()  #
                        begin = time.time()
                        total=0
                        loss_total = 0
                        for batched_x, batched_y in batch_data(data_finetune, batch_size, seed):
                            input_data = batched_x
                            target_data = batched_y
                            num_batch = len(batched_y)
                            total += len(input_data)
                            with autograd.record():
                                y_hats = self.model.net.output(input_data)
                                ls = loss_b(y_hats, target_data)
                                ls.backward()
                            trainer.step(num_batch)
                            loss_total += nd.sum(ls)
                        loss = loss_total / total
                        print("-----finetune epoch---", i)
                        print("ls:", loss)
                        nd.waitall()  #
                        end = time.time()
                        print("==========================")
                        print("cali-time per epoch:",end-begin)
                    nd.waitall()
                    end1 = time.time()
                    print("cali-time:", end1 - begin1)
            #commB
             #

            update = self.model.get_params()
            comp=0

        else:
            comp, update = self.model.train(
                data, my_round, num_epochs, batch_size)

        nd.waitall()

        return comp, self.num_train_samples, update


    def test(self, set_to_use="test"):
        """Tests self.model on self.test_data.
        Args:
            set_to_use: Set to test on. Should be in ["train", "test"].
        Returns:
            metrics: Dict of metrics returned by the model.
        """
        assert set_to_use in ["train", "test", "val"]
        if set_to_use == "train":
            data = self.train_data
        elif set_to_use == "test" or set_to_use == "val":
            data = self.test_data
        return self.model.test(data)

    def set_model(self, model):
        """Set the model data to specified model.
        Args:
            model: The specified model.
        """
        self.model.set_params(model.get_params())

    @property
    def num_train_samples(self):
        """Return the number of train samples for this client."""
        if not hasattr(self, "_num_train_samples"):
            self._num_train_samples = len(self.train_data["y"])

        return self._num_train_samples

    @property
    def num_test_samples(self):
        """Return the number of test samples for this client."""
        if not hasattr(self, "_num_test_samples"):
            self._num_test_samples = len(self.test_data["y"])

        return self._num_test_samples

    @property
    def num_samples(self):
        """Return the number of train + test samples for this client."""
        if not hasattr(self, "_num_samples"):
            self._num_samples = self.num_train_samples + self.num_test_samples

        return self._num_samples

    @property
    def train_sample_dist(self,streaming=True):#记得改回True
        """Return the distribution of train data for this client."""
        if not hasattr(self, "_train_sample_dist"):
            if(streaming==True):
                labels=self.streaming_data["y"]
            else:
                labels = self.train_data["y"]
            labels = labels.asnumpy().astype("int64")
            dist = np.bincount(labels)
            # align to num_classes
            num_classes = self.model.num_classes
            self._train_sample_dist = np.concatenate(
                (dist, np.zeros(num_classes - len(dist))))

        return self._train_sample_dist



    @property
    def test_sample_dist(self):
        """Return the distribution of test data for this client."""
        if not hasattr(self, "_test_sample_dist"):
            labels = self.test_data["y"]
            labels = labels.asnumpy().astype("int64")
            dist = np.bincount(labels)
            # align to num_classes
            num_classes = self.model.num_classes
            self._test_sample_dist = np.concatenate(
                (dist, np.zeros(num_classes - len(dist))))

        return self._test_sample_dist

    def client_score(self, base_dist, score_cal="cosine"):
        """Return the score of overall data for this client."""
        if score_cal == "cosine":
            self.score = 1-scipy.spatial.distance.cosine(self.train_sample_dist, base_dist)
        elif score_cal == "wasserstein":
            train_sample_dist_ = self.train_sample_dist / self.train_sample_dist.sum()
            base_dist_=base_dist/base_dist.sum()
            self.score = scipy.stats.wasserstein_distance(train_sample_dist_, base_dist_)
        elif score_cal == "js":
            distance = scipy.spatial.distance.jensenshannon(self.train_sample_dist, base_dist)
        elif score_cal=="projection":
            self.score=np.dot(self.train_sample_dist, base_dist) / np.linalg.norm(base_dist)
        return self.score
    @property
    def sample_dist(self):
        """Return the distribution of overall data for this client."""
        if not hasattr(self, "_sample_dist"):
            self._sample_dist = self.train_sample_dist + self.test_sample_dist

        return self._sample_dist

    @property
    def model(self):
        """Returns this client reference to model being trained"""
        return self._model

    @model.setter
    def model(self, model):
        warnings.warn("The current implementation shares the model among all clients."
                      "Setting it on one client will effectively modify all clients.")
        self._model = model

    def process_data(self, data):
        """Convert train data and test data to NDArray objects with
        specified context.
        Args:
            data: List of train vectors or labels.
        Returns:
            nd_data: Format NDArray data with specified context.
        """

        return nd.array(data, ctx=self.model.ctx)
