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

class RNNModel:
    def __init__(self, configuration):
        self.configuration = configuration
        u = T.matrix(dtype=theano.config.floatX)
        y = T.ivector()
        n_in = self.configuration["n_in"]
        n_out = self.configuration["n_out"]
        hiddens = self.configuration["hidden"]
        acts = self.configuration["activation"]
        drates = self.configuration["drates"]
        biases = self.configuration["bias"]

        print "...building the model"
        
        self.ldnn = LDNN()
        self.ldnn.add_input_layer(u,
            dropout_rate=drates[0])
        self.ldnn.add_layer(n_in, hiddens[0],
            dropout_rate=drates[1],
            activation=acts[0],
            bias=biases[0])
        
        for i in xrange(len(hiddens) - 1):
            self.ldnn.add_layer(hiddens[i], hiddens[i + 1], dropout_rate=drates[i+2],
                activation = acts[i+1], bias=bool(biases[0]))
        
        self.ldnn.connect_output(n_out)
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
            outputs=[cost, self.ldnn.output_layer.d_error(y)],
            updates=updates,
            allow_input_downcast=True)

        self.predict_model = theano.function(inputs=[u, y],
            outputs=[self.ldnn.output_layer.error(y)],
            allow_input_downcast=True)

    def train(self, trainX, trainY, devX, devY):
        print "...training the model"
        self.sizef = lambda X: np.sum(map(lambda a: len(a), X))
        trainSize = self.sizef(trainY)
        devSize = self.sizef(devY)

        for i in xrange(1, self.configuration["epoch"] + 1):
            losses = []
            errors = []
            start = time.time()
            
            for j in xrange(len(trainY)):
                loss, err = self.train_model(np.asarray(trainX[j]), trainY[j])
                curr = float(len(trainY[j]))
                ratio = curr / trainSize
                losses.append(loss * ratio)
                errors.append(err * ratio)
            
            devErr = []
            for j in xrange(len(devY)):
                err = self.predict_model(devX[j], devY[j])
                curr = float(len(devY[j]))
                ratio = curr / devSize
                devErr.append(err[0] * ratio)
 
            end = time.time()
            print "Epoch: %i Loss: %.6f Error: %.6f Dev Err: %.6f Time: %.6f seconds" % (i, np.sum(losses), np.sum(errors), np.sum(devErr), end - start)
            sys.stdout.flush()
    
    def test(self, testX, testY):
        testSize = self.sizef(testX)
        testErr = []
        for i in xrange(len(testY)):
                err = self.predict_model(testX[i], testY[i])
                curr = float(len(testY[i]))
                ratio = curr / testSize
                testErr.append(err[0] * ratio)
        return np.sum(testErr)
        
def get_arg_parser():
    parser = argparse.ArgumentParser(prog="nerrnn")
    parser.add_argument("--data", default="", help="data path")
    parser.add_argument("--hidden", default=[100], type=int, nargs='+', help="number of neurons in each hidden layer")
    parser.add_argument("--activation", default=["tanh"], nargs='+', help="activation function for hidden layer : sigmoid, tanh, relu, lstm, gru")
    parser.add_argument("--drates", default=[0, 0], nargs='+', type=float, help="dropout rates")
    parser.add_argument("--bias", default=[0], nargs='+', type=int, help="bias on/off for layer")
    parser.add_argument("--opt", default="rmsprop", help="optimization method: sgd, rmsprop, adagrad, adam")
    parser.add_argument("--epoch", default=50, type=int, help="number of epochs")
    parser.add_argument("--lr", default=0.005, type=float, help="learning rate")
    parser.add_argument("--norm", default=5, type=float, help="Threshold for clipping norm of gradient")
    parser.add_argument("--truncate", default=-1, type=int, help="backward step size")

    return parser

def get_ner_data(filename):
    data = np.load(filename)
    return data["trn"][0], data["trn"][1], data["dev"][0], data["dev"][1], data["tst"][0], data["tst"][1]

def nerexp(args):
    trainX, trainY, devX, devY, tstX, tstY = get_ner_data(args["data"])
    args["n_in"] = trainX[0].shape[1]
    args["n_out"] = 17
    rnn = RNNModel(args)
    rnn.train(trainX, trainY, devX, devY)
    tstErr = rnn.test(tstX, tstY)
    print "Test Err: %.6f" % tstErr

if __name__ == "__main__":
    parser = get_arg_parser()
    args = vars(parser.parse_args())
    print args
        
    nerexp(args)
