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
import argparse
from theano import sparse as sp

class RNNModel:
    def __init__(self, configuration):
        self.configuration = configuration
        u = T.matrix(dtype=theano.config.floatX)
        y = T.ivector()
        if self.configuration["mean"]:
            I = T.imatrix('I')
        else:
            I = T.ivector('I')
        n_in = self.configuration["n_in"]
        n_out = self.configuration["n_out"]
        hiddens = self.configuration["hidden"]
        acts = self.configuration["activation"]
        drates = self.configuration["drates"]
        biases = self.configuration["bias"]

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
            bias=bool(biases[0]), n_out=numout)
        for i in xrange(len(hiddens) - 1):
            indices = I if "pool" in acts[i+1] else None
            sys.stdout.flush()
            self.ldnn.add_layer(hiddens[i], hiddens[i + 1], dropout_rate=drates[i+2],
                activation = acts[i+1], bias=bool(biases[0]), indices=indices)

        self.ldnn.connect_output(n_out, compile_predict=False, recurrent=self.configuration["recout"])
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

        self.train_model = theano.function(inputs=[u, y, I],
            outputs=[cost, self.ldnn.output_layer.d_error(y)],
            updates=updates,
            allow_input_downcast=True,
        )

        self.predict_model = theano.function(inputs=[u, y, I],
            outputs=[self.ldnn.output_layer.error(y), self.ldnn.output_layer.y_pred],
            allow_input_downcast=True)

    def train(self, trnX, trnY, trnI, trnIndx, devX, devY, devI, devIndx):
        print "...training the model"

        train_size = trnIndx[-1][1]

        for i in xrange(1, self.configuration["epoch"] + 1):
            losses = []
            errors = []
            start = time.time()

            meanpool = self.configuration["mean"]
            s_y = 0
            for ix in xrange(len(trnIndx)):
                s, e = trnIndx[ix]

                if meanpool:
                    loss, err = self.train_model(trnX[s:e, :], trnY[s_y:(s_y+trnI[ix].shape[0])], trnI[ix])
                else:
                    loss, err = self.train_model(trnX[s:e, :],
                        trnY[s_y:(s_y+trnI[ix].shape[0])], trnI[ix][:, 1] - 1)
                s_y += trnI[ix].shape[0]
                curr = e - s
                ratio = float(curr) / train_size
                losses.append(loss * ratio)
                errors.append(err * ratio)

            end = time.time()
            trn_time = end - start

            print "Epoch: %i Loss: %.6f Error: %.6f Time: %.6f seconds" % (i, np.sum(losses), np.sum(errors), trn_time)
            sys.stdout.flush()

            devErr, dev_time = self.test(devX, devY, devI, devIndx)
            print "Dev Err: %.6f Time: %.6f seconds" % (np.sum(devErr), dev_time)
            sys.stdout.flush()

    def test(self, testX, testY, testI, testIndx):
        test_size = testIndx[-1][1]
        testErr = []
        self.last_predictions = []
        start = time.time()
        s_y = 0

        meanpool = self.configuration["mean"]

        for ix in xrange(len(testIndx)):
            s, e = testIndx[ix]
            if meanpool:
                err, preds = self.predict_model(testX[s:e,:],
                    testY[s_y:(s_y+testI[ix].shape[0])], testI[ix])
            else:
                err, preds = self.predict_model(testX[s:e, :],
                        testY[s_y:(s_y+testI[ix].shape[0])], testI[ix][:, 1] - 1)
            s_y += testI[ix].shape[0]
            self.last_predictions.append(preds)
            curr = e - s
            ratio = float(curr) / test_size
            testErr.append(err * ratio)
            end = time.time()
        return np.sum(testErr), end - start
