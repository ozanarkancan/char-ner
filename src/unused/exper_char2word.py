import copy
from itertools import *
import random, numpy as np
from utils import get_sents, get_sent_indx
from sklearn.feature_extraction import DictVectorizer
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import confusion_matrix, classification_report
import argparse
from nerrnn_char2word import RNNModel
from featchar import *
import featchar
import sys

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
    parser.add_argument("--fepoch", default=50, type=int, help="number of epochs")
    parser.add_argument("--sample", default=False, action='store_true', help="sample 100 from trn, 10 from dev")
    parser.add_argument("--feat", default='basic', help="feat func to use")
    parser.add_argument("--lr", default=0.005, type=float, help="learning rate")
    parser.add_argument("--norm", default=5, type=float, help="Threshold for clipping norm of gradient")
    parser.add_argument("--truncate", default=-1, type=int, help="backward step size")
    parser.add_argument("--recout", default=0, type=int, help="Recurrent Output Layer")
    parser.add_argument("--mean", default=False, type=bool,
        help="pool or mean pool")
    
    return parser

def print_conmat(y_true, y_pred, lblenc):
    print '\t'.join(['bos'] + list(lblenc.classes_))
    conmat = confusion_matrix(y_true,y_pred, labels=lblenc.transform(lblenc.classes_))
    for r,clss in zip(conmat,lblenc.classes_):
        print '\t'.join([clss] + list(map(str,r)))

if __name__ == '__main__':
    trn, dev, tst = get_sents()
    parser = get_arg_parser()
    args = vars(parser.parse_args())
    print args

    if args['sample']:
        trn = random.sample(trn,100)
        dev = random.sample(dev,2)
        tst = random.sample(tst,2)

    featfunc = getattr(featchar,'get_cfeatures_'+args['feat'])

    for d in (trn,dev,tst):
        for sent in d:
            sent.update({
                'cseq': get_cseq(sent), 
                'wiseq': get_wiseq(sent), 
                'tseq': get_tseq2(sent)})
                #'tseq': get_tseq1(sent)})

    dvec = DictVectorizer(dtype=np.float32, sparse=False)
    lblenc = LabelEncoder()
    dvec.fit(featfunc(ci, sent)  for sent in trn for ci,c in enumerate(sent['cseq']))
    lblenc.fit([t for sent in trn for t in sent['ts']])
    print dvec.get_feature_names()
    print lblenc.classes_

    nf = len(dvec.get_feature_names())
    nc = len(lblenc.classes_)
    print '# of sents: ', list(map(len, (trn,dev,tst)))
    print '# of feats: ', nf 
    print '# of lbls: ', nc

    Xtrn = dvec.transform(featfunc(ci, sent)  for sent in trn for ci,c in enumerate(sent['cseq']))
    Xdev = dvec.transform(featfunc(ci, sent)  for sent in dev for ci,c in enumerate(sent['cseq']))
    Xtst = dvec.transform(featfunc(ci, sent)  for sent in tst for ci,c in enumerate(sent['cseq']))

    print Xtrn.shape, Xdev.shape

    ytrn = lblenc.transform([t for sent in trn for t in sent['ts']])
    ydev = lblenc.transform([t for sent in dev for t in sent['ts']])
    ytst = lblenc.transform([t for sent in tst for t in sent['ts']])

    print ytrn.shape, ydev.shape

    trnIndx = get_sent_indx(trn)
    devIndx = get_sent_indx(dev)
    tstIndx = get_sent_indx(tst)
    
    print len(trnIndx), len(devIndx)

    trnI = [sent_word_indx(sent) for sent in trn]
    devI = [sent_word_indx(sent) for sent in dev]
    tstI = [sent_word_indx(sent) for sent in tst]

    print len(trnI), len(devI)
  
    args["n_in"] = Xtrn.shape[1]
    args["n_out"] = len(np.unique(ytrn))
    
    from biloueval import bilouEval2
    rnn = RNNModel(args)
    for e in xrange(args['fepoch']):
        rnn.train(Xtrn, ytrn, trnI, trnIndx, Xdev, ydev, devI, devIndx)
        y_true, y_pred = ydev, list(chain.from_iterable(rnn.last_predictions))
        print_conmat(y_true, y_pred, lblenc)

        lts = [sent['ts'] for sent in dev]
        lts_pred = []
        for sent, ipred in zip(dev, rnn.last_predictions):
            ts_pred = lblenc.inverse_transform(ipred)
            lts_pred.append(ts_pred)

        print 'f1: ', bilouEval2(lts, lts_pred)
        sys.stdout.flush()
        # sent_indx = random.sample(range(len(dev)), 10)
    tstErr, tmp = rnn.test(Xtst, ytst, tstI, tstIndx)
    lts = [sent['ts'] for sent in tst]
    lts_pred = []
    for sent, ipred in zip(tst, rnn.last_predictions):
        ts_pred = lblenc.inverse_transform(ipred)
        lts_pred.append(ts_pred)
        
    print "Test Err: %.6f" % tstErr
    print 'f1: ', bilouEval2(lts, lts_pred)
