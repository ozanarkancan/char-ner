import copy
from itertools import *
import random, numpy as np
from utils import *
import argparse
from nerrnn import RNNModel

def get_arg_parser():
    parser = argparse.ArgumentParser(prog="nerrnn")
    
    parser.add_argument("--data", default="", help="data path")
    parser.add_argument("--hidden", default=[100], type=int, nargs='+',
        help="number of neurons in each hidden layer")
    parser.add_argument("--activation", default=["tanh"], nargs='+',
        help="activation function for hidden layer : sigmoid, tanh, relu, lstm, gru")
    parser.add_argument("--drates", default=[0, 0], nargs='+', type=float,
        help="dropout rates")
    parser.add_argument("--bias", default=[0], nargs='+', type=int,
        help="bias on/off for layer")
    parser.add_argument("--opt", default="rmsprop", help="optimization method: sgd, rmsprop, adagrad, adam")
    parser.add_argument("--epoch", default=50, type=int, help="number of epochs")
    parser.add_argument("--lr", default=0.005, type=float, help="learning rate")
    parser.add_argument("--norm", default=5, type=float, help="Threshold for clipping norm of gradient")
    parser.add_argument("--truncate", default=-1, type=int, help="backward step size")
    
    return parser

if __name__ == '__main__':
    trn, dev, tst = get_sents()

    for d in (trn,dev,tst):
        for sent in d:
            extend_sent(sent)

    dvec = DictVectorizer(dtype=np.float32, sparse=False)
    lblenc = LabelEncoder()
    dvec.fit(get_cfeatures(wi, ci, sent)  for sent in trn for c,wi,ci in zip(sent['cseq'],sent['wiseq'],count(0)))
    lblenc.fit([t for sent in trn for t in sent['tseq']])
    print dvec.get_feature_names()
    print lblenc.classes_

    nf = len(dvec.get_feature_names())
    nc = len(lblenc.classes_)
    print '# of sents: ', map(len, (trn,dev,tst))
    print '# of feats: ', nf 
    print '# of lbls: ', nc

    Xtrn = dvec.transform(get_cfeatures(wi, ci, sent)  for sent in trn for c,wi,ci in zip(sent['cseq'],sent['wiseq'],count(0)))
    Xdev = dvec.transform(get_cfeatures(wi, ci, sent)  for sent in dev for c,wi,ci in zip(sent['cseq'],sent['wiseq'],count(0)))
    Xtst = dvec.transform(get_cfeatures(wi, ci, sent)  for sent in tst for c,wi,ci in zip(sent['cseq'],sent['wiseq'],count(0)))

    print Xtrn.shape, Xdev.shape

    ytrn = lblenc.transform([t for sent in trn for t in sent['tseq']])
    ydev = lblenc.transform([t for sent in dev for t in sent['tseq']])
    ytst = lblenc.transform([t for sent in tst for t in sent['tseq']])

    print ytrn.shape, ydev.shape

    trnIndx = get_sent_indx(trn)
    devIndx = get_sent_indx(dev)
    tstIndx = get_sent_indx(tst)
    
    print len(trnIndx), len(devIndx)
    
    parser = get_arg_parser()
    args = vars(parser.parse_args())
  
    args["n_in"] = Xtrn.shape[1]
    args["n_out"] = len(np.unique(ytrn))
    
    print args
    
    rnn = RNNModel(args, Xtrn, ytrn)
    rnn.train(trnIndx, Xdev, ydev, devIndx)
    tstErr, tstTime = rnn.test(Xtst, ytst, tstIndx)

    print "Test Err: %.6f Time: %.6f" % (tstErr, tstTime)
