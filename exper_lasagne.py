import copy
from itertools import *
import random, numpy as np
from utils import get_sents, get_sent_indx
from sklearn.feature_extraction import DictVectorizer
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import confusion_matrix, classification_report
import argparse
from featchar import *
import featchar
import lasagne
from biloueval import bilouEval2
import theano.tensor as T
import theano

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
    parser.add_argument("--feat", default='basic_seg', help="feat func to use")
    parser.add_argument("--lr", default=0.005, type=float, help="learning rate")
    parser.add_argument("--norm", default=5, type=float, help="Threshold for clipping norm of gradient")
    parser.add_argument("--truncate", default=-1, type=int, help="backward step size")
    
    return parser

def print_conmat(y_true, y_pred, tseqenc):
    print '\t'.join(['bos'] + list(tseqenc.classes_))
    conmat = confusion_matrix(y_true,y_pred, labels=tseqenc.transform(tseqenc.classes_))
    for r,clss in zip(conmat,tseqenc.classes_):
        print '\t'.join([clss] + list(map(str,r)))

def print_pred_info(dset, int_predL, charenc, wordenc, tfunc):
    y_true = charenc.transform([t for sent in dset for t in sent['ts']])
    y_pred = list(chain.from_iterable(int_predL))
    print_conmat(y_true, y_pred, charenc)

    lts = [sent['ts'] for sent in dset]
    lts_pred = []
    for sent, ipred in zip(dset, int_predL):
        tseq_pred = charenc.inverse_transform(ipred)
        tseqgrp_pred = get_tseqgrp(sent['wiseq'],tseq_pred)
        ts_pred = tfunc(tseqgrp_pred)
        lts_pred.append(ts_pred)

    y_true = wordenc.transform([t for ts in lts for t in ts])
    y_pred = wordenc.transform([t for ts in lts_pred for t in ts])
    print_conmat(y_true, y_pred, wordenc)

    print 'f1: {:3.4f} {:3.4f} {:3.4f} {:3.4f}'.format(bilouEval2(lts, lts_pred))


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
    dvec.fit(featfunc(ci, sent)  for sent in trn for ci,c in enumerate(sent['cseq']))
    tseqenc = LabelEncoder()
    tseqenc.fit([t for sent in trn for t in sent['tseq']])
    tsenc = LabelEncoder()
    tsenc.fit([t for sent in trn for t in sent['ts']])
    print dvec.get_feature_names()
    print tseqenc.classes_
    print tsenc.classes_

    nf = len(dvec.get_feature_names())
    nc = len(tseqenc.classes_)
    ntrnsent, ndevsent, ntstsent = list(map(len, (trn,dev,tst)))
    print '# of sents trn, dev, tst: ', ntrnsent, ndevsent, ntstsent
    print 'NF:', nf, 'NC:', nc


    """
    Xdev = dvec.transform(featfunc(ci, sent)  for sent in dev for ci,c in enumerate(sent['cseq']))
    Xdev = dvec.transform(featfunc(ci, sent)  for sent in dev for ci,c in enumerate(sent['cseq']))
    Xtst = dvec.transform(featfunc(ci, sent)  for sent in tst for ci,c in enumerate(sent['cseq']))

    ytrn = tseqenc.transform([t for sent in trn for t in sent['tseq']])
    ydev = tseqenc.transform([t for sent in dev for t in sent['tseq']])
    ytst = tseqenc.transform([t for sent in tst for t in sent['tseq']])
    """

  
    # NETWORK params
    MAX_LENGTH = max(len(sent['cseq']) for sent in trn)
    MIN_LENGTH = min(len(sent['cseq']) for sent in trn)
    print 'maxlen:', MAX_LENGTH, 'minlen:', MIN_LENGTH
    N_BATCH = None
    N_HIDDEN = 100
    GRAD_CLIP = 10
    LEARNING_RATE = .001
    # end NETWORK params

    for sent in trn:
        Xsent = dvec.transform([featfunc(ci, sent) for ci,c in enumerate(sent['cseq'])])
        padding = MAX_LENGTH - Xsent.shape[0]
        Xsent = np.pad(Xsent,((0,padding),(0,0)),mode='constant')
        print Xsent
        print Xsent.shape
        break
    # NETWORK
    l_in = lasagne.layers.InputLayer(shape=(N_BATCH, MAX_LENGTH, nf))
    l_mask = lasagne.layers.InputLayer(shape=(N_BATCH, MAX_LENGTH))
    l_forward = lasagne.layers.RecurrentLayer(
        l_in, N_HIDDEN, mask_input=l_mask, grad_clipping=GRAD_CLIP,
        W_in_to_hid=lasagne.init.HeUniform(),
        W_hid_to_hid=lasagne.init.HeUniform(),
        nonlinearity=lasagne.nonlinearities.rectify)
    l_backward = lasagne.layers.RecurrentLayer(
        l_in, N_HIDDEN, mask_input=l_mask, grad_clipping=GRAD_CLIP,
        W_in_to_hid=lasagne.init.HeUniform(),
        W_hid_to_hid=lasagne.init.HeUniform(),
        nonlinearity=lasagne.nonlinearities.rectify, backwards=True)
    l_sum = lasagne.layers.ConcatLayer([l_forward, l_backward])
    # Our output layer is a simple dense connection, with 1 output unit
    l_out = lasagne.layers.DenseLayer(
        l_sum, num_units=nc, nonlinearity=lasagne.nonlinearities.softmax)

    target_values = T.vector('target_output', dtype='int64')
    network_output = lasagne.layers.get_output(l_out)
    predicted_values = network_output
    cost = lasagne.objectives.categorical_crossentropy(predicted_values, target_values).mean()

    all_params = lasagne.layers.get_all_params(l_out)
    # Compute SGD updates for training
    print("Computing updates ...")
    updates = lasagne.updates.adagrad(cost, all_params, LEARNING_RATE)
    # Theano functions for training and computing cost
    print("Compiling functions ...")
    train = theano.function([l_in.input_var, target_values, l_mask.input_var],
                            cost, updates=updates)
    """
    compute_cost = theano.function(
        [l_in.input_var, target_values, l_mask.input_var], cost)
    # end NETWORK
    
    for e in xrange(args['fepoch']):
        rnn.train(Xtrn, ytrn, trnIndx, Xdev, ydev, devIndx)
        print_pred_info(dev, rnn.last_predictions, tseqenc, tsenc, get_ts2)

    tstErr, tmp = rnn.test(Xtst, ytst, tstIndx)
    print "Test Err: %.6f" % tstErr
    print_pred_info(tst, rnn.last_predictions, tseqenc, tsenc, get_ts2)
    """

