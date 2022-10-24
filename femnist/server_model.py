import os
import mxnet as mx
from mxnet import autograd, nd,init,gluon
from mxnet import init, nd
from mxnet.gluon import loss as gloss
from utils.model_utils import build_net
from baseline_constants import ACCURACY_KEY, INPUT_SIZE
from utils.model_utils import batch_data
from  mxnet.gluon.loss import *
class ServerModel:

    def __init__(self, client_model, dataset, model_name, num_classes, ctx):
        self.client_model = client_model
        self.dataset = dataset
        self.model_name = model_name
        self.num_classes = num_classes
        self.ctx = ctx
        self.lr=0.01
        self.create_model()
        self.loss = gloss.SoftmaxCrossEntropyLoss()

    def optimizer(self):
        """Optimizer to be used, the default is SGD optimizer."""
        if self._optimizer is None:
            self._optimizer = "sgd"

        return self._optimizer
    def create_model(self):
        """Build and initialize the server model. If self.client_model is
        given, this ServerModel object is created for the server model,
        otherwise for the merged update. The server model will be synchronized
        with the client model, and the merged update will be initialized to zero.
        """
        self.net = build_net(
            self.dataset, self.model_name, self.num_classes, self.ctx)

        if self.client_model:
            self.set_params(self.client_model.get_params())
        else:
            self.net.initialize(
                init.Zero(), ctx=self.ctx, force_reinit=True)
        self.trainer = mx.gluon.Trainer(
            params=self.net.collect_params(),
            optimizer="sgd",
            optimizer_params={"learning_rate": 0.01}
        )

    def reset_zero(self):
        """Reset the model data to zero, usually used to reset the merged update.
        Note that force reinit the model data with:
            self.net.initialize(
                init.Zero(), ctx=self.ctx, force_reinit=True)
        will leads to high cpu usage.
        """
        self.set_params([])

    def set_params(self, model_params):
        """Set the model data to the specified data. If an empty list is given,
        the model data will be set to zero.
        Args:
            model_params: The specified model data.
        """
        source_params = list(model_params)
        target_params = list(self.get_params())
        num_params = len(target_params)

        for p in range(num_params):
            if source_params:
                data = source_params[p].data()
            else:
                data = nd.zeros(target_params[p].shape, ctx=self.ctx)
            target_params[p].set_data(data)

    def get_params(self):
        """Return current model data.
        Returns:
            params: The current model data.
        """
        return self.net.collect_params().values()

    def save(self, log_dir):
        """Saves the server model to:
            {log_dir}/{model_name}.params
        """
        self.net.save_parameters(
            os.path.join(log_dir, self.model_name + ".params"))

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
                optimizer="sgd",
                optimizer_params={"learning_rate": 0.01}
            )




        # Train on data for epochs
        for i in range(num_epochs):
            seed = my_round * 11 + i
            # print("previous:",previous_model)
            self.run_epochs(seed, data, batch_size,previous_model)


        # Wait to avoid running out of GPU memory
        nd.waitall()

        update = self.get_params()
        comp = 0
        return comp, update
    def run_epochs(self, seed, data, batch_size,previous_model):
        for batched_x, batched_y in batch_data(data, batch_size, seed):
            input_data = self.preprocess_x(batched_x)
            target_data = self.preprocess_y(batched_y)


            num_batch = len(batched_y)

            # Set MXNET_ENFORCE_DETERMINISM=1 to avoid difference in
            # calculation precision.
            if(previous_model==None):
                print("none")

            else:
                logits_previous=previous_model(input_data)
            with autograd.record():
                y_hats = self.net(input_data)
                # y_presentation=self.net.features(input_data)
                # print("batch feature:",y_presentation)
                ls1= self.loss(y_hats, target_data)
                # print("y_hats:",y_hats)
                # print("logtis_previous",logits_previous)
                if(previous_model!=None):
                    ls2=KLDivLoss(from_logits=True)(mx.nd.log_softmax(y_hats),mx.nd.softmax(logits_previous))
                else:
                    ls2=0
                ls=ls1+0.1*ls2
                # print("ls1:",ls1)
                # print("ls2:",ls2)
                ls.backward()

            self.trainer.step(num_batch)
    def test(self, data):
        # Process train data and labels before inference
        x_vecs = self.preprocess_x(data["x"])
        labels = self.preprocess_y(data["y"])

        # Model inference
        output = self.net(x_vecs)

        # Calculate accuracy and loss
        acc = (output.argmax(axis=1) == labels).mean().asscalar()
        loss = self.loss(output, labels).mean().asscalar()
        return {"accuracy": acc, "loss": loss}
    def preprocess_x(self, raw_x_batch):
        return raw_x_batch.reshape((-1, *INPUT_SIZE))

    def preprocess_y(self, raw_y_batch):
        return raw_y_batch
