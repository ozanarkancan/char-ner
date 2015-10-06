import logging
import argparse
import sys, time, datetime
from itertools import *
import random, numpy as np

from sklearn.feature_extraction import DictVectorizer
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import confusion_matrix, classification_report

import theano
import lasagne
import theano.tensor as T

from featchar import *
import featchar
from utils import get_sents, get_sent_indx, sample_sents
from utils import ROOT_DIR
from biloueval import bilouEval2, bilou_post_correct
from score import conlleval
from encoding import io2bio
from lazrnn import RDNN, RDNN_Dummy, extract_rnn_params
from nerrnn import RNNModel

LOG_DIR = '{}/logs'.format(ROOT_DIR)
random.seed(0)
rng = np.random.RandomState(1234567)
lasagne.random.set_rng(rng)

def get_arg_parser():
    parser = argparse.ArgumentParser(prog="lazrnn")
    
    parser.add_argument("--rnn", default='lazrnn', choices=['dummy','lazrnn','nerrnn'], help="which rnn to use")
    parser.add_argument("--activation", default=['bi-relu'], nargs='+', help="activation function for hidden layer : sigmoid, tanh, rectify")
    parser.add_argument("--n_hidden", default=[100], nargs='+', type=int, help="number of neurons in each hidden layer")
    parser.add_argument("--recout", default=0, type=int, help="use recurrent output layer")
    parser.add_argument("--drates", default=[0, 0], nargs='+', type=float, help="dropout rates")
    parser.add_argument("--opt", default="adam", help="optimization method: sgd, rmsprop, adagrad, adam")
    parser.add_argument("--lr", default=0.001, type=float, help="learning rate")
    parser.add_argument("--norm", default=2, type=float, help="Threshold for clipping norm of gradient")
    parser.add_argument("--n_batch", default=20, type=int, help="batch size")
    parser.add_argument("--fepoch", default=300, type=int, help="number of epochs")
    parser.add_argument("--patience", default=-1, type=int, help="how patient the validator is")
    parser.add_argument("--feat", default='basic_seg', help="feat func to use")
    parser.add_argument("--grad_clip", default=-1, type=float, help="clip gradient messages in recurrent layers if they are above this value")
    parser.add_argument("--truncate", default=-1, type=int, help="backward step size")
    parser.add_argument("--log", default='das_auto', help="log file name")
    parser.add_argument("--sorted", default=1, type=int, help="sort datasets before training and prediction")

    parser.add_argument("--gbudget", default=10, type=int, help="num of parameter combinations")
    parser.add_argument("--gfepoch", default=300, type=int, help="number of epochs")
    parser.add_argument("--gpatience", default=-1, type=int, help="how patient the validator is")
    parser.add_argument("--gsample", default=0, type=int, help="num of sents to sample from trn in the order of K for grid search")
    
    return parser


class Feat(object):

    def __init__(self, featfunc):
        self.dvec = DictVectorizer(dtype=np.float32, sparse=False)
        self.tseqenc = LabelEncoder()
        self.tsenc = LabelEncoder()
        self.featfunc = featfunc

    def fit(self, trn):
        self.dvec.fit(self.featfunc(ci, sent)  for sent in trn for ci,c in enumerate(sent['cseq']))
        self.tseqenc.fit([t for sent in trn for t in sent['tseq']])
        self.tsenc.fit([t for sent in trn for t in sent['ts']])
        self.feature_names = self.dvec.get_feature_names()
        self.ctag_classes = self.tseqenc.classes_
        self.wtag_classes = self.tsenc.classes_
        logging.info(self.feature_names)
        logging.info(self.ctag_classes)
        logging.info(self.wtag_classes)
        self.NF = len(self.feature_names)
        self.NC = len(self.ctag_classes)
        logging.info('NF: {} NC: {}'.format(self.NF, self.NC))

    def transform(self, sent):
        Xsent = self.dvec.transform([self.featfunc(ci, sent) for ci,c in enumerate(sent['cseq'])]) # nchar x nf
        ysent = self.one_hot(self.tseqenc.transform([t for t in sent['tseq']]), self.NC) # nchar x nc
        return Xsent, ysent

    def one_hot(self, labels, n_classes):
        one_hot = np.zeros((labels.shape[0], n_classes)).astype(bool)
        one_hot[range(labels.shape[0]), labels] = True
        return one_hot

class Batcher(object):

    def __init__(self, batch_size, feat):
        self.batch_size = batch_size
        self.feat = feat

    def get_batches(self, dset):
        nf = self.feat.NF 
        sent_batches = [dset[i:i+self.batch_size] for i in range(0, len(dset), self.batch_size)]
        X_batches, Xmsk_batches, y_batches, ymsk_batches = [], [], [], []
        for batch in sent_batches:
            mlen = max(len(sent['cseq']) for sent in batch)
            X_batch = np.zeros((len(batch), mlen, nf),dtype=theano.config.floatX)
            Xmsk_batch = np.zeros((len(batch), mlen),dtype=np.bool)
            y_batch = np.zeros((len(batch), mlen, self.feat.NC),dtype=theano.config.floatX)
            ymsk_batch = np.zeros((len(batch), mlen, self.feat.NC),dtype=np.bool)
            for si, sent in enumerate(batch):
                Xsent, ysent = self.feat.transform(sent)
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

class Reporter(object):

    def __init__(self, feat, tfunc):
        self.feat = feat
        self.tfunc = tfunc

    def report(self, dset, pred):
        y_true = self.feat.tseqenc.transform([t for sent in dset for t in sent['tseq']])
        y_pred = list(chain.from_iterable(pred))
        cerr = np.sum(y_true!=y_pred)/float(len(y_true))

        char_conmat_str = self.get_conmat_str(y_true, y_pred, self.feat.tseqenc)

        lts = [sent['ts'] for sent in dset]
        lts_pred = []
        for sent, ipred in zip(dset, pred):
            tseq_pred = self.feat.tseqenc.inverse_transform(ipred)
            tseqgrp_pred = get_tseqgrp(sent['wiseq'],tseq_pred)
            ts_pred = self.tfunc(tseqgrp_pred)
            lts_pred.append(io2bio(ts_pred)) # changed

        y_true = self.feat.tsenc.transform([t for ts in lts for t in ts])
        y_pred = self.feat.tsenc.transform([t for ts in lts_pred for t in ts])
        werr = np.sum(y_true!=y_pred)/float(len(y_true))

        word_conmat_str = self.get_conmat_str(y_true, y_pred, self.feat.tsenc)

        # wacc, pre, recall, f1 = bilouEval2(lts, lts_pred)
        (wacc, pre, recall, f1), conll_print = conlleval(lts, lts_pred)
        return cerr, werr, wacc, pre, recall, f1, conll_print, char_conmat_str, word_conmat_str

    def get_conmat_str(self, y_true, y_pred, lblenc): # NOT USED
        str_list = []
        str_list.append('\t'.join(['bos'] + list(lblenc.classes_)))
        conmat = confusion_matrix(y_true,y_pred, labels=lblenc.transform(lblenc.classes_))
        for r,clss in zip(conmat,lblenc.classes_):
            str_list.append('\t'.join([clss] + list(map(str,r))))
        return '\n'.join(str_list) + '\n'

class Validator(object):

    def __init__(self, trn, dev, batcher, reporter, log_lvl='info'):
        self.trn = trn
        self.dev = dev
        self.trndat = batcher.get_batches(trn) 
        self.devdat = batcher.get_batches(dev) 
        self.reporter = reporter
        self.logfunc = getattr(logging,log_lvl)

    def validate(self, rdnn, fepoch, patience=-1):
        self.logfunc('training the model...')
        dbests = {'trn':(1,0.), 'dev':(1,0.)}
        for e in range(1,fepoch+1): # foreach epoch
            self.logfunc(('{:<5} {:<5} ' + ('{:>10} '*10)).format('dset','epoch','mcost', 'mtime', 'cerr', 'werr', 'wacc', 'pre', 'recall', 'f1', 'best', 'best'))
            for funcname, ddat, datname in zip(['train','predict'],[self.trndat,self.devdat],['trn','dev']):
                start_time = time.time()
                mcost, pred = getattr(rdnn, funcname)(ddat)
                end_time = time.time()
                mtime = end_time - start_time
                
                # cerr, werr, wacc, pre, recall, f1 = self.reporter.report(getattr(self.trn, pred) # find better solution for getattr
                cerr, werr, wacc, pre, recall, f1, conll_print, char_conmat_str, word_conmat_str =\
                        self.reporter.report(getattr(self, datname), pred) # find better solution for getattr
                if f1 > dbests[datname][1]: dbests[datname] = (e,f1)
                self.logfunc(('{:<5} {:<5d} ' + ('{:>10.4f} '*9)+'{:>10d}')\
                        .format(datname,e,mcost, mtime, cerr, werr, wacc, pre, recall, f1, dbests[datname][1],dbests[datname][0]))
                logging.debug('')
                logging.debug(conll_print)
                logging.debug(char_conmat_str)
                logging.debug(word_conmat_str)
                logging.debug('')
            if patience > 0 and e - dbests['dev'][0] > patience:
                self.logfunc('sabir tasti.')
                break
            self.logfunc('')
        return dbests['dev'][1]
            
def valid_file_name(s):
    return "".join(i for i in s if i not in "\"\/ &*?<>|[]()'")

def main():
    parser = get_arg_parser()
    args = vars(parser.parse_args())

    if args['rnn'] == 'nerrnn':
        args['n_batch'] = 1

    # logger setup
    import socket
    host = socket.gethostname().split('.')[0]
    logger = logging.getLogger()
    logger.setLevel(logging.DEBUG)
    shandler = logging.StreamHandler()
    shandler.setLevel(logging.INFO)
    lparams = ['rnn', 'feat', 'activation', 'n_hidden', 'recout', 'opt','n_batch', 'gsample', 'gfepoch','gpatience','gbudget']
    param_log_name = ','.join(['{}:{}'.format(p,args[p]) for p in lparams])
    param_log_name = valid_file_name(param_log_name)
    base_log_name = '{}:{},{}'.format(host, theano.config.device, param_log_name if args['log'] == 'das_auto' else args['log'])
    # base_log_name = '{:%d-%m-%y+%H:%M:%S}:{},{}'.format(datetime.datetime.now(), theano.config.device, param_log_name if args['log'] == 'das_auto' else args['log'])
    ihandler = logging.FileHandler('{}/{}.info'.format(LOG_DIR,base_log_name), mode='w')
    ihandler.setLevel(logging.INFO)
    dhandler = logging.FileHandler('{}/{}.debug'.format(LOG_DIR,base_log_name), mode='w')
    dhandler.setLevel(logging.DEBUG)
    logger.addHandler(shandler);logger.addHandler(ihandler);logger.addHandler(dhandler);
    # end logger setup

    # print args
    for k,v in sorted(args.iteritems()):
        logger.info('{}:\t{}'.format(k,v))
    logger.info('{}:\t{}'.format('base_log_name',base_log_name))

    trn, dev, tst = get_sents('eng','bio')


    # TODO
    ctag2wtag_func = get_ts3
    wtag2ctag_func = get_tseq3

    for d in (trn,dev,tst):
        for sent in d:
            sent.update({
                'cseq': get_cseq(sent), 
                'wiseq': get_wiseq(sent), 
                'tseq': wtag2ctag_func(sent)})

    if args['sorted']:
        trn = sorted(trn, key=lambda sent: len(sent['cseq']))
        dev = sorted(dev, key=lambda sent: len(sent['cseq']))

    ntrnsent, ndevsent, ntstsent = list(map(len, (trn,dev,tst)))
    logger.info('# of sents trn, dev, tst: {} {} {}'.format(ntrnsent, ndevsent, ntstsent))

    MAX_LENGTH = max(len(sent['cseq']) for sent in chain(trn,dev))
    MIN_LENGTH = min(len(sent['cseq']) for sent in chain(trn,dev))
    logger.info('maxlen: {} minlen: {}'.format(MAX_LENGTH, MIN_LENGTH))

    featfunc = getattr(featchar,'get_cfeatures_'+args['feat'])
    feat = Feat(featfunc)
    feat.fit(trn)

    batcher = Batcher(args['n_batch'], feat)
    reporter = Reporter(feat, ctag2wtag_func)

    if args['gsample']>0:
        trn_sample_size = args['gsample']*1000
        trn_sample = sample_sents(trn,trn_sample_size)
        gvalidator = Validator(trn_sample, dev, batcher, reporter, log_lvl='debug')
    else:
        gvalidator = Validator(trn, dev, batcher, reporter, log_lvl='debug')

    # select rnn
    if args['rnn'] == 'dummy':
        RNN = RDNN_Dummy
    elif args['rnn'] == 'lazrnn':
        RNN = RDNN
    elif args['rnn'] == 'nerrnn':
        RNN = RNNModel
    else:
        raise Exception
    # end select rnn

    norm_list = [0.1, 1, 10, 100]
    lr_list = [1.e-1, 1.e-2, 1.e-3, 1.e-4]
    rnn_params = extract_rnn_params(args)

    optim_params = random.sample(list(product(norm_list,lr_list)),args['gbudget'])
    res_list = []

    for norm, lr in optim_params:
        rnn_params['norm'] = norm
        rnn_params['lr'] = lr
        rdnn = RNN(feat.NC, feat.NF, rnn_params)
        start_time = time.time()
        f1 = gvalidator.validate(rdnn, args['gfepoch'], args['gpatience'])
        end_time = time.time()
        mtime = end_time - start_time
        res_list.append((f1,(norm,lr)))
        logging.info('f1: {}\tnorm: {}\tlr: {:.2g}\tsecs: {:.2f}'.format(f1,norm,lr,mtime))

    (f1,(norm,lr)) = max(res_list)
    logging.info('f1: {}\tnorm: {}\tlr: {:.2g} :best'.format(f1,norm,lr))
    rnn_params['norm'] = norm
    rnn_params['lr'] = lr
    validator = Validator(trn, dev, batcher, reporter, log_lvl='info')
    rdnn = RNN(feat.NC, feat.NF, rnn_params)
    f1 = validator.validate(rdnn, args['fepoch'], args['patience'])

    # lr: scipy.stats.expon.rvs(loc=0.0001,scale=0.1,size=100)
    # norm: scipy.stats.expon.rvs(loc=0, scale=5,size=10)

if __name__ == '__main__':
    main()
