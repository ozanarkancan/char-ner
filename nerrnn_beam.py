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
from beam_tree import *
from featchar import *

class RNNModel:
    def __init__(self, configuration, X, Y):
        self.configuration = configuration
        u = T.matrix(dtype=theano.config.floatX)
        y = T.ivector()
        s_index = T.lscalar()
        e_index = T.lscalar()

        self.shared_x = theano.shared(np.asarray(X, dtype=theano.config.floatX), borrow=True)
        self.shared_y = T.cast(theano.shared(np.asarray(Y, dtype=theano.config.floatX), borrow=True), 'int32')
        
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

        self.train_model = theano.function(inputs=[s_index, e_index],
            outputs=[cost, self.ldnn.output_layer.d_error(y)],
            updates=updates,
            allow_input_downcast=True,
            givens={
                u: self.shared_x[s_index:e_index, :],
                y: self.shared_y[s_index:e_index]
            }
        )

        self.predict_model = theano.function([u, y],
            outputs=[self.ldnn.output_layer.error(y), self.ldnn.output_layer.y_pred],
            allow_input_downcast=True)

        """
        self.predict_model = theano.function([s_index, e_index],
            outputs=[self.ldnn.output_layer.error(y), self.ldnn.output_layer.y_pred],
            allow_input_downcast=True,
            givens={
                u: self.shared_x[s_index:e_index, :],
                y: self.shared_y[s_index:e_index]
            })
        """

    def train(self, trnX, trnY, trainIndx, dev, dvec, lblenc, featfunc):
        print "...training the model"
        
        train_size = trainIndx[-1][1]

        for i in xrange(1, self.configuration["epoch"] + 1):
            # self.shared_x = theano.shared(np.asarray(trnX, dtype=theano.config.floatX), borrow=True)
            # self.shared_y = T.cast(theano.shared(np.asarray(trnY, dtype=theano.config.floatX), borrow=True), 'int32')
            losses = []
            errors = []
            start = time.time()
            
            for s, e in trainIndx:
                loss, err = self.train_model(s, e)
                curr = e - s
                ratio = float(curr) / train_size
                losses.append(loss * ratio)
                errors.append(err * ratio)
            
            end = time.time()
            trn_time = end - start

            print "Epoch: %i Loss: %.6f Error: %.6f Time: %.6f seconds" % (i, np.sum(losses), np.sum(errors), trn_time)
            sys.stdout.flush()

            devErr, dev_time = self.test(dev, dvec, lblenc, featfunc)
            print "Dev Err: %.6f Time: %.6f seconds" % (np.sum(devErr), dev_time)
            sys.stdout.flush()
    
    def test(self, tst, dvec, lblenc, featfunc):
        #test_size = testIndx[-1][1]
        testErr = []
        self.last_predictions = []
	
        start = time.time()
	total_char = 0
	total_err = 0
	for sent in tst:
        #for s, e in testIndx:
		self.ldnn.reset_memory()
            	beam_tree = BeamTree(self.configuration["beam"])
		memory = [(m[0], m[0].get_value()) for m in self.ldnn.memo]
		root = Node(memory)
		beam_tree.beams.append(root)
		
		gold = lblenc.transform(sent["tseq"])
		#print "GOLD: ", gold
		for ci, c in enumerate(sent["cseq"]):
			for n in beam_tree.beams:
				for m in n.memory:
					m[0].set_value(m[1])
				xdict = featfunc(ci, sent, tseq_pred=lblenc.inverse_transform(n.preds) if len(n.preds) else [])
				xvec = dvec.transform(xdict)
				pred, probs = self.ldnn.predict(xvec)
				pred = pred[0]
				#print "Pred: ", pred
				probs = probs[0]
				#print "Probs: ", probs
				memory = [(m[0], m[0].get_value()) for m in self.ldnn.memo]
				for j in xrange(self.configuration["n_out"]):
					node = Node(memory, n.prob * probs[j], n.preds + [j])
					beam_tree.cands.append(node)
			beam_tree.prune()
			#beam_tree.print_tree()
		best = beam_tree.beams[0]
		#print "GOLD: ", gold
		#print "BEST: ", best.preds
		
		self.last_predictions.append(best.preds)
		gold = lblenc.transform(sent["tseq"])
		preds = np.array(best.preds)
		total_char += float(len(gold))
		total_err += np.sum(gold != preds)
		#print "ERR: ", np.sum(gold != preds)

        end = time.time()
        return total_err / total_char, end - start
        
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
