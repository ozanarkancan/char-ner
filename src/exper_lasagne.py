import copy, sys, time, logging, datetime
from itertools import *
import random, numpy as np
from utils import get_sents, get_sent_indx, sample_sents
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
from lazrnn import RDNN

LOG_DIR = 'logs'

def get_arg_parser():
    parser = argparse.ArgumentParser(prog="lazrnn")
    
    parser.add_argument("--n_hidden", default=[100], nargs='+', type=int, help="number of neurons in each hidden layer")
    parser.add_argument("--activation", default='rectify', help="activation function for hidden layer : sigmoid, tanh, rectify")
    # parser.add_argument("--drates", default=[0, 0], nargs='+', type=float, help="dropout rates")
    # parser.add_argument("--bias", default=[0], nargs='+', type=int, help="bias on/off for layer")
    parser.add_argument("--opt", default="adam", help="optimization method: sgd, rmsprop, adagrad, adam")
    parser.add_argument("--ltype", default="recurrent", help="layer type: recurrent lstm")
    parser.add_argument("--n_batch", default=50, type=int, help="batch size")
    # parser.add_argument("--epoch", default=50, type=int, help="number of epochs")
    parser.add_argument("--fepoch", default=50, type=int, help="number of epochs")
    # parser.add_argument("--sample", default=False, action='store_true', help="sample 100 from trn, 10 from dev")
    parser.add_argument("--sample", default=[], nargs='+', type=int, help="num of sents to sample from trn, dev in the order of K")
    parser.add_argument("--feat", default='basic_seg', help="feat func to use")
    parser.add_argument("--lr", default=0.005, type=float, help="learning rate")
    parser.add_argument("--grad_clip", default=-1, type=float, help="clip gradient messages in recurrent layers if they are above this value")
    parser.add_argument("--norm", default=2, type=float, help="Threshold for clipping norm of gradient")
    parser.add_argument("--truncate", default=-1, type=int, help="backward step size")
    parser.add_argument("--recout", default=False, action='store_true', help="use recurrent output layer")
    parser.add_argument("--log", default='das_auto', help="log file name")
    parser.add_argument("--sorted", default=1, type=int, help="sort datasets before training and prediction")
    
    return parser

def print_conmat(y_true, y_pred, tseqenc):
    print '\t'.join(['bos'] + list(tseqenc.classes_))
    conmat = confusion_matrix(y_true,y_pred, labels=tseqenc.transform(tseqenc.classes_))
    for r,clss in zip(conmat,tseqenc.classes_):
        print '\t'.join([clss] + list(map(str,r)))

def pred_info(dset, int_predL, charenc, wordenc, tfunc):
    y_true = charenc.transform([t for sent in dset for t in sent['tseq']])
    y_pred = list(chain.from_iterable(int_predL))
    cerr = np.sum(y_true!=y_pred)/float(len(y_true))

    lts = [sent['ts'] for sent in dset]
    lts_pred = []
    for sent, ipred in zip(dset, int_predL):
        tseq_pred = charenc.inverse_transform(ipred)
        tseqgrp_pred = get_tseqgrp(sent['wiseq'],tseq_pred)
        ts_pred = tfunc(tseqgrp_pred)
        lts_pred.append(ts_pred)

    y_true = wordenc.transform([t for ts in lts for t in ts])
    y_pred = wordenc.transform([t for ts in lts_pred for t in ts])
    werr = np.sum(y_true!=y_pred)/float(len(y_true))
    wacc, pre, recall, f1 = bilouEval2(lts, lts_pred)
    return cerr, werr, wacc, pre, recall, f1

def print_pred_info(dset, int_predL, charenc, wordenc, tfunc, epoch, dset_name):
    y_true = charenc.transform([t for sent in dset for t in sent['tseq']])
    y_pred = list(chain.from_iterable(int_predL))
    print_conmat(y_true, y_pred, charenc)
    err = np.sum(y_true!=y_pred)/len(y_true)

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

    print 'f1\t{}\t{}\t{:3.4f}\t{:3.4f}\t{:3.4f}\t{:3.4f}'.format(dset_name, epoch, *bilouEval2(lts, lts_pred))
    print 

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
    nf = len(dvec.get_feature_names())
    sent_batches = [dset[i:i+bsize] for i in range(0, len(dset), bsize)]
    X_batches, Xmsk_batches, y_batches, ymsk_batches = [], [], [], []
    for batch in sent_batches:
        X_batch = np.zeros((len(batch), mlen, nf),dtype=theano.config.floatX)
        Xmsk_batch = np.zeros((len(batch), mlen),dtype=np.bool)
        y_batch = np.zeros((len(batch), mlen, nc),dtype=theano.config.floatX)
        ymsk_batch = np.zeros((len(batch), mlen, nc),dtype=np.bool)
        for si, sent in enumerate(batch):
            Xsent = dvec.transform([featfunc(ci, sent) for ci,c in enumerate(sent['cseq'])]) # nchar x nf
            ysent = one_hot(tseqenc.transform([t for t in sent['tseq']]), nc) # nchar x nc
            nchar = Xsent.shape[0]
            X_batch[si,:nchar,:] = Xsent
            Xmsk_batch[si,:nchar] = True
            y_batch[si,:nchar,:] = ysent
            ymsk_batch[si,:nchar,:] = True
        X_batches.append(X_batch)
        Xmsk_batches.append(Xmsk_batch)
        y_batches.append(y_batch)
        ymsk_batches.append(ymsk_batch)
    return X_batches, Xmsk_batches, y_batches, ymsk_batches


if __name__ == '__main__':
    parser = get_arg_parser()
    args = vars(parser.parse_args())

    # logger setup
    logger = logging.getLogger()
    logger.setLevel(logging.DEBUG)
    shandler = logging.StreamHandler()
    shandler.setLevel(logging.INFO)
    # 
    lparams = ['ltype','activation','n_hidden','opt','lr','norm','recout']
    param_log_name = ','.join(['{}:{}'.format(p,args[p]) for p in lparams])
    base_log_name = '{:%d-%m-%y+%H:%M:%S}={}'.format(datetime.datetime.now(), param_log_name if args['log'] == 'das_auto' else args['log'])
    ihandler = logging.FileHandler('{}/{}.info'.format(LOG_DIR,base_log_name), mode='w')
    ihandler.setLevel(logging.INFO)
    dhandler = logging.FileHandler('{}/{}.debug'.format(LOG_DIR,base_log_name), mode='w')
    dhandler.setLevel(logging.DEBUG)
    logger.addHandler(shandler);logger.addHandler(ihandler);logger.addHandler(dhandler);
    # end logger setup

    # print args
    for k,v in sorted(args.iteritems()):
        logger.info('{}:\t{}'.format(k,v))

    trn, dev, tst = get_sents()

    if len(args['sample']):
        trn_size = args['sample'][0]*1000
        dev_size = args['sample'][1]*1000
        trn = sample_sents(trn,trn_size)
        dev = sample_sents(dev,dev_size)

    featfunc = getattr(featchar,'get_cfeatures_'+args['feat'])

    for d in (trn,dev,tst):
        for sent in d:
            sent.update({
                'cseq': get_cseq(sent), 
                'wiseq': get_wiseq(sent), 
                'tseq': get_tseq2(sent)})
                #'tseq': get_tseq1(sent)})

    if args['sorted']:
        trn = sorted(trn, key=lambda sent: len(sent['cseq']))
        dev = sorted(dev, key=lambda sent: len(sent['cseq']))

    dvec = DictVectorizer(dtype=np.float32, sparse=False)
    dvec.fit(featfunc(ci, sent)  for sent in trn for ci,c in enumerate(sent['cseq']))
    tseqenc = LabelEncoder()
    tseqenc.fit([t for sent in trn for t in sent['tseq']])
    tsenc = LabelEncoder()
    tsenc.fit([t for sent in trn for t in sent['ts']])
    logger.info(dvec.get_feature_names())
    logger.info(tseqenc.classes_)
    logger.info(tsenc.classes_)

    nf = len(dvec.get_feature_names())
    nc = len(tseqenc.classes_)
    ntrnsent, ndevsent, ntstsent = list(map(len, (trn,dev,tst)))
    logger.info('# of sents trn, dev, tst: {} {} {}'.format(ntrnsent, ndevsent, ntstsent))
    logger.info('NF: {} NC: {}'.format(nf, nc))



    # NETWORK params
    MAX_LENGTH = max(len(sent['cseq']) for sent in chain(trn,dev))
    MIN_LENGTH = min(len(sent['cseq']) for sent in chain(trn,dev))
    logger.info('maxlen: {} minlen: {}'.format(MAX_LENGTH, MIN_LENGTH))
    # end NETWORK params

    trndat = batch_dset(trn, dvec, tseqenc, MAX_LENGTH, args['n_batch'])
    devdat = batch_dset(dev, dvec, tseqenc, MAX_LENGTH, args['n_batch'])
    # tstdat = batch_dset(tst, dvec, tseqenc, MAX_LENGTH, args['n_batch'])


    rdnn = RDNN(nc, nf, MAX_LENGTH, **args)
    for e in range(1,args['fepoch']+1):
        # trn
        start_time = time.time()
        # mcost, pred = rdnn.sing(trndat, 'train')
        m1cost = rdnn.train(trndat)
        mcost, pred = rdnn.predict(trndat)
        end_time = time.time()
        mtime = end_time - start_time
        cerr, werr, wacc, pre, recall, f1 = pred_info(trn, pred, tseqenc, tsenc, get_ts2)
        logger.info(('{:<5} {:<5} ' + ('{:>10} '*8)).format('dset','epoch','mcost', 'mtime', 'cerr', 'werr', 'wacc', 'pre', 'recall', 'f1'))
        logger.info(('{:<5} {:<5d} ' + ('{:>10.4f} '*8)).format('trn',e,mcost, mtime, cerr, werr, wacc, pre, recall, f1))
        # end trn
        
        # dev
        start_time = time.time()
        # mcost, pred = rdnn.sing(devdat, 'predict')
        mcost, pred = rdnn.predict(devdat)
        end_time = time.time()
        mtime = end_time - start_time
        cerr, werr, wacc, pre, recall, f1 = pred_info(dev, pred, tseqenc, tsenc, get_ts2)
        logger.info(('{:<5} {:<5d} ' + ('{:>10.4f} '*8)).format('dev',e,mcost, mtime, cerr, werr, wacc, pre, recall, f1))
        # end dev
        logger.info('')
