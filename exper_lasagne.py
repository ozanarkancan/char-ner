import copy, sys, time
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
    y_true = charenc.transform([t for sent in dset for t in sent['tseq']])
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

    print 'f1: {:3.4f} {:3.4f} {:3.4f} {:3.4f}'.format(*bilouEval2(lts, lts_pred))

def one_hot(labels, n_classes):
    '''
    Converts an array of label integers to a one-hot matrix encoding
    :parameters:
        - labels : np.ndarray, dtype=int
            Array of integer labels, in {0, n_classes - 1}
        - n_classes : int
            Total number of classes
    :returns:
        - one_hot : np.ndarray, dtype=bool, shape=(labels.shape[0], n_classes)
            One-hot matrix of the input
    '''
    one_hot = np.zeros((labels.shape[0], n_classes)).astype(bool)
    one_hot[range(labels.shape[0]), labels] = True
    return one_hot

def make_batches(X, length, batch_size=50):
    '''
    Convert a list of matrices into batches of uniform length
    :parameters:
        - X : list of np.ndarray
            List of matrices
        - length : int
            Desired sequence length.  Smaller sequences will be padded with 0s,
            longer will be truncated.
        - batch_size : int
            Mini-batch size
    :returns:
        - X_batch : np.ndarray
            Tensor of time series matrix batches,
            shape=(n_batches, batch_size, length, n_features)
        - X_mask : np.ndarray
            Mask denoting whether to include each time step of each time series
            matrix
    '''
    n_batches = len(X)//batch_size
    X_batch = np.zeros((n_batches, batch_size, length, X[0].shape[1]),
                       dtype=theano.config.floatX)
    X_mask = np.zeros(X_batch.shape, dtype=np.bool)
    for b in range(n_batches):
        for n in range(batch_size):
            X_m = X[b*batch_size + n]
            X_batch[b, n, :X_m.shape[0]] = X_m[:length]
            X_mask[b, n, :X_m.shape[0]] = 1
    return X_batch, X_mask

def batch_dset(dset, dvec, tseqenc, mlen, bsize):
    XL, yL = [], []
    for i, sent in enumerate(dset):
        Xsent = dvec.transform([featfunc(ci, sent) for ci,c in enumerate(sent['cseq'])])
        XL.append(Xsent)
        yL.append(one_hot(tseqenc.transform([t for t in sent['tseq']]), nc))
    X, Xmsk = make_batches(XL, mlen, batch_size=bsize)
    Xmsk = Xmsk[:,:,:,0]
    y, ymsk = make_batches(yL, mlen, batch_size=bsize)
    return X, Xmsk, y, ymsk


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



    # NETWORK params
    MAX_LENGTH = max(len(sent['cseq']) for sent in trn)
    MIN_LENGTH = min(len(sent['cseq']) for sent in trn)
    print 'maxlen:', MAX_LENGTH, 'minlen:', MIN_LENGTH
    N_BATCH = 50
    N_HIDDEN = 500
    GRAD_CLIP = 7
    LEARNING_RATE = .001
    # end NETWORK params

    Xtrn, Xtrnmsk, ytrn, ytrnmsk = batch_dset(trn, dvec, tseqenc, MAX_LENGTH, N_BATCH)
    Xdev, Xdevmsk, ydev, ydevmsk = batch_dset(dev, dvec, tseqenc, MAX_LENGTH, N_BATCH)
    print 'Xtrn, Xtrnmsk, ytrn, ytrnmsk', Xtrn.shape, Xtrnmsk.shape, ytrn.shape, ytrnmsk.shape
    print 'Xdev, Xdevmsk, ydev, ydevmsk', Xdev.shape, Xdevmsk.shape, ydev.shape, ydevmsk.shape

    """
    XtrnL, ytrnL = [], []
    for i, sent in enumerate(trn[:100]):
        Xsent = dvec.transform([featfunc(ci, sent) for ci,c in enumerate(sent['cseq'])])
        XtrnL.append(Xsent)
        ytrnL.append(one_hot(tseqenc.transform([t for t in sent['tseq']]), nc))
    Xtrn, Xmsk = make_batches(XtrnL, MAX_LENGTH, batch_size=N_BATCH)
    Xmsk = Xmsk[:,:,:,0]
    ytrn, ytrnmsk = make_batches(ytrnL, MAX_LENGTH, batch_size=N_BATCH)
    """

    # NETWORK
    l_in = lasagne.layers.InputLayer(shape=(N_BATCH, MAX_LENGTH, nf))
    #  batchsize, seqlen, _ = l_inp.input_var.shape # symbolic ref to input_var shape
    l_mask = lasagne.layers.InputLayer(shape=(N_BATCH, MAX_LENGTH))
    l_forward = lasagne.layers.RecurrentLayer(
        l_in, N_HIDDEN, mask_input=l_mask, grad_clipping=GRAD_CLIP,
        # W_in_to_hid=lasagne.init.HeUniform(gain='relu'),
        # W_hid_to_hid=lasagne.init.HeUniform(gain='relu'),
        nonlinearity=lasagne.nonlinearities.rectify)
    print 'l_forward:', lasagne.layers.get_output_shape(l_forward)
    l_backward = lasagne.layers.RecurrentLayer(
        l_in, N_HIDDEN, mask_input=l_mask, grad_clipping=GRAD_CLIP,
        # W_in_to_hid=lasagne.init.HeUniform(gain='relu'),
        # W_hid_to_hid=lasagne.init.HeUniform(gain='relu'),
        nonlinearity=lasagne.nonlinearities.rectify, backwards=True)
    print 'l_backward:', lasagne.layers.get_output_shape(l_backward)
    # l_sum = lasagne.layers.ConcatLayer([l_forward, l_backward])
    l_sum = lasagne.layers.ElemwiseSumLayer([l_forward, l_backward])
    print 'l_sum:', lasagne.layers.get_output_shape(l_sum)
    # Our output layer is a simple dense connection, with 1 output unit
    l_reshape = lasagne.layers.ReshapeLayer(l_sum, (-1, N_HIDDEN))
    print 'l_reshape:', lasagne.layers.get_output_shape(l_reshape)
    l_rec_out = lasagne.layers.DenseLayer(
        l_reshape, num_units=nc, nonlinearity=lasagne.nonlinearities.softmax)
    print 'l_rec_out:', lasagne.layers.get_output_shape(l_rec_out)
    l_out = lasagne.layers.ReshapeLayer(l_rec_out, (N_BATCH, MAX_LENGTH, nc))
    print 'l_out:', lasagne.layers.get_output_shape(l_out)
    # l_out = lasagne.layers.ReshapeLayer(l_soft_out, (N_BATCH, MAX_LENGTH))
    print lasagne.layers.get_output_shape(l_out)


    input = T.tensor3('input')
    target_output = T.tensor3('target_output')
    out_mask = T.tensor3('mask')

    def cost(output):
         return -T.sum(out_mask*target_output*T.log(output))/T.sum(out_mask)

    cost_train = cost(lasagne.layers.get_output(l_out, deterministic=False))
    cost_eval = cost(lasagne.layers.get_output(l_out, deterministic=True))

    all_params = lasagne.layers.get_all_params(l_out, trainable=True)
    # Compute SGD updates for training
    print("Computing updates ...")
    # updates = lasagne.updates.adagrad(cost_train, all_params, LEARNING_RATE)
    updates = lasagne.updates.adam(cost_train, all_params, LEARNING_RATE)
    # Theano functions for training and computing cost
    print("Compiling functions ...")
    train = theano.function([l_in.input_var, target_output, l_mask.input_var, out_mask],
                            cost_train, updates=updates)
    compute_cost = theano.function(
        [l_in.input_var, target_output, l_mask.input_var, out_mask], cost_eval)
    predict = theano.function([l_in.input_var, l_mask.input_var], lasagne.layers.get_output(l_out, deterministic=True))
    # end NETWORK
    for e in range(50):
        start_time = time.time()
        for b in range(Xtrn.shape[0]):
            train(Xtrn[b], ytrn[b], Xtrnmsk[b], ytrnmsk[b])
        end_time = time.time()
        print 'training seconds:', end_time - start_time
        
        start_time = time.time()
        rnn_last_predictions = []
        for b in range(Xdev.shape[0]):
            pred = predict(Xdev[b], Xdevmsk[b])
            predictions = np.argmax(pred*ydevmsk[b], axis=-1).flatten()
            sentLens = Xdevmsk[b].sum(axis=-1)
            for i, slen in enumerate(sentLens):
                rnn_last_predictions.append(predictions[i*MAX_LENGTH:i*MAX_LENGTH+slen])
        end_time = time.time()
        print 'predicting seconds:', end_time - start_time

        print_pred_info(dev, rnn_last_predictions, tseqenc, tsenc, get_ts2)
    """
    truth = np.argmax(ytrn*ytrnmsk[0], axis=-1).flatten()
    n_time_steps = np.sum(ytrnmsk)/ytrnmsk.shape[-1]
    error = np.sum(predictions != truth)/float(n_time_steps)
    print error
    """
    """
    sentLens = Xtrnmsk[0].sum(axis=-1)
    for i, slen in enumerate(sentLens):
        print predictions[i*MAX_LENGTH:i*MAX_LENGTH+slen]
    """
    """
    
    for e in xrange(args['fepoch']):
        rnn.train(Xtrn, ytrn, trnIndx, Xdev, ydev, devIndx)
        print_pred_info(dev, rnn.last_predictions, tseqenc, tsenc, get_ts2)

    tstErr, tmp = rnn.test(Xtst, ytst, tstIndx)
    print "Test Err: %.6f" % tstErr
    print_pred_info(tst, rnn.last_predictions, tseqenc, tsenc, get_ts2)
    """

