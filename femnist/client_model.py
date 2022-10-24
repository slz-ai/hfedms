import mxnet as mx
from mxnet.gluon import loss as gloss

from baseline_constants import ACCURACY_KEY, INPUT_SIZE
from model import Model
from utils.model_utils import build_net
from mxnet import autograd, nd,init,gluon
from mxnet.gluon import nn
import numpy as np
def batch_data(data, batch_size, seed):
    """Return batches of data as an iterator.
    Args:
        data: A dict := {"x": NDArray, "y": NDArray} (on one client).
        batch_size: Number of samples in a batch data.
        seed: The random number seed.
    Returns:
        batched_x: A batch of features of length: batch_size.
        batched_y: A batch of labels of length: batch_size.
    """
    data_x = data["x"]
    data_y = data["y"]

    # randomly shuffle data
    mx.random.seed(seed)
    data_x = mx.random.shuffle(data_x)
    mx.random.seed(seed)
    data_y = mx.random.shuffle(data_y)

    # loop through mini-batches
    for i in range(0, len(data_x), batch_size):
        l = i
        r = min(i + batch_size, len(data_y))
        batched_x = data_x[l:r]
        batched_y = data_y[l:r]
        yield batched_x, batched_y

class ClientModel(Model):
    def __init__(self, seed, dataset, model_name, ctx, lr, num_classes):
        self.dataset = dataset
        self.model_name = model_name
        self.num_classes = num_classes

        super(ClientModel, self).__init__(seed, lr, ctx)
    def create_model(self):
        # Build a simple cnn network
        net = build_net(
            self.dataset, self.model_name, self.num_classes, self.ctx, self.seed)

        # Use softmax cross-entropy loss
        loss = gloss.SoftmaxCrossEntropyLoss()

        # Create trainer
        trainer = mx.gluon.Trainer(
            params=net.collect_params(),
            optimizer=self.optimizer,
            optimizer_params={"learning_rate": self.lr}
        )

        return net, loss, trainer

    def train(self, data, my_round, num_epochs, batch_size, lr_factor=1.0,finetune=False,previous_model=None,weight=1):  # train all model/just classifier
        """
        Train the model using a batch of data.
        Args:
            data: Dict of the form {"x": NDArray, "y": NDArray}.
            my_round: The current training round, used for learning rate
                decay.
            num_epochs: Number of epochs when clients train on data.
            batch_size: Size of train data batches.
            lr_factor: Decay factor for learning rate.
        Returns:
            comp: Number of FLOPs computed while training given data.
            update: Trained model params.
        """
        # Decay learning rate
        # add trainer settings

        self.trainer.set_learning_rate(
            self.lr * (lr_factor ** my_round)*weight)
        # print("finetune or not:",finetune)
        if(finetune==True):
            self.trainer = mx.gluon.Trainer(
                params=self.net.collect_params(select='.*dense2'),
                optimizer=self.optimizer,
                optimizer_params={"learning_rate": self.lr}
            )
        else:
            self.trainer = mx.gluon.Trainer(
                params=self.net.collect_params(),
                optimizer=self.optimizer,
                optimizer_params={"learning_rate": self.lr}
            )


        # Train on data for epochs
        for i in range(num_epochs):
            seed = my_round * 11 + i
            # print("previous:",previous_model)
            self.run_epochs(seed, data, batch_size,previous_model)


        # Wait to avoid running out of GPU memory
        nd.waitall()

        update = self.get_params()
        comp = num_epochs * len(data["y"]) * self.flops_per_sample
        return comp, update



    def test(self, data):
        # Process train data and labels before inference
        x_vecs = self.preprocess_x(data["x"])
        labels = self.preprocess_y(data["y"])

        # Model inference
        output = self.net(x_vecs)

        # Calculate accuracy and loss
        acc = (output.argmax(axis=1) == labels).mean().asscalar()
        loss = self.loss(output, labels).mean().asscalar()
        return {ACCURACY_KEY: acc, "loss": loss}

    def preprocess_x(self, raw_x_batch):
        return raw_x_batch.reshape((-1, *INPUT_SIZE))

    def preprocess_y(self, raw_y_batch):
        return raw_y_batch
    def get_feature(self,data):
        x_vecs = self.preprocess_x(data["x"])
        labels = self.preprocess_y(data["y"])
        features= self.net.features(x_vecs)
        f_len=len(features[0])
        # print("flen:",f_len)
        temp_list=[[] for i in range(62)]
        avg_list=[]
        var_list=[]
        num_list=[]
        varience_list=nd.array(62)
        j=0
        for label in labels:
            # print(int(label.asnumpy()[0]))
            temp_list[int(label.asnumpy()[0])].append(features[j])
            j=j+1
        # print("labels",labels)
        j=0
        # print("temp_lengh:",len(temp_list))
        for c in  temp_list:
            # print("temp_list",j,len(c))
            if(len(c)!=0):
                feature_avg=0
                feature_var=0
                for feature in c:
                    feature_avg+=feature
                feature_avg=(feature_avg/len(c))
                for feature in c:
                    diff=feature-feature_avg
                    feature_var+=nd.dot(feature,feature.T)
                feature_var-=len(c)*nd.dot(feature_avg,feature_avg.T)
                if(len(c)>=2):
                    feature_var=(feature_var/(len(c)-1)).asnumpy()[0]
                else:
                    feature_var=0
                # print("avg:",feature_avg)
                # print("var:",feature_var)
            else:
                feature_avg=nd.zeros(f_len)
                feature_var=0

            j+=1
            avg_list.append(feature_avg.asnumpy())
            var_list.append(feature_var)
            num_list.append(len(c))


        # print("avg_list",avg_list)
        # print("var_list",var_list)
        return avg_list,var_list,num_list,labels,features
    # def generate_features(self,data):