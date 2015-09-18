import numpy as np
from numpy.random import *
import theano
from theano import tensor as T
from kuconnect.ldnn import *
from kuconnect.optim import *
import time
import sys
import os
import numpy as np, h5py
from theano import sparse as sp

class RNNModel:
    def __init__(self, n_out, n_in, configuration):
        self.configuration = configuration
        u = T.matrix(dtype=theano.config.floatX)
        y = T.ivector()

        hiddens = self.configuration["n_hidden"]
        acts = self.configuration["activation"]
        drates = self.configuration["drates"]

        print "...building the model"
        
        self.ldnn = LDNN()

        if acts[0].startswith("bi"):
            self.ldnn.add_bidirectional_input_layer(u,
                    dropout_rate=drates[0])
        else:
            self.ldnn.add_input_layer(u,
                    dropout_rate=drates[0])

        numout = n_out if "feedback" in acts[0] else None

        self.ldnn.add_layer(n_in, hiddens[0],
            dropout_rate=drates[1],
            activation=acts[0],
            bias=True, n_out=numout)
        
        for i in xrange(len(hiddens) - 1):
            self.ldnn.add_layer(hiddens[i], hiddens[i + 1], dropout_rate=drates[i+2],
                activation = acts[i+1], bias=True)
        
        self.ldnn.connect_output(n_out, recurrent=self.configuration["recout"])
        cost = self.ldnn.get_cost(y)
        params = self.ldnn.get_params()
        gparams = T.grad(cost, params)

        if self.configuration["norm"] != 0:
            gparams = clip_norms(gparams, params, self.configuration["norm"])

        lr = self.configuration["lr"]
        optim = self.configuration["opt"]

        if optim == "rmsprop":
            updates = rmsprop(params, gparams, lr)
        elif optim == "adagrad":
            updates = adagrad(params, gparams, lr)
        elif optim == "adam":
            updates = adam(params, gparams)
        else:#sgd
            updates = sgd(params, gparams, lr)

        self.train_model = theano.function(inputs=[u, y],
            outputs=cost,
            updates=updates,
            allow_input_downcast=True,
        )

        self.predict_model = theano.function([u, y],
            outputs=[self.ldnn.output_layer.loss(y), self.ldnn.output_layer.y_pred],
            allow_input_downcast=True)

    def train(self, dsetdat):
        tcost = 0
        for Xdset, Xdsetmsk, ydset, ydsetmsk in zip(*dsetdat):
            x,y = Xdset[0], np.nonzero(ydset[0])[1]
            cost = self.train_model(x,y)
            tcost += cost
        _, preds = self.predict(dsetdat)

        return tcost, preds
    
    def predict(self, dsetdat):
        ecost, rnn_last_predictions = 0, []
        for Xdset, Xdsetmsk, ydset, ydsetmsk in zip(*dsetdat):
            x,y = Xdset[0], np.nonzero(ydset[0])[1]
            cost, preds = self.predict_model(x,y)
            ecost += cost
            rnn_last_predictions.append(preds)
        return ecost, rnn_last_predictions
