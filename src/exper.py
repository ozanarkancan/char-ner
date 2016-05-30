import copy
import logging
import argparse
import time
from itertools import chain
import random, numpy as np

import theano
import lasagne
from sklearn.metrics import confusion_matrix

from dataset import Dset
import featchar, decoder
import rep
from utils import valid_file_name
from utils import LOG_DIR, MODEL_DIR
from score import conlleval
from lazrnn import RDNN, RDNN_Dummy

random.seed(7)
np.random.seed(7)
rng = np.random.RandomState(1234567)
lasagne.random.set_rng(rng)

def get_args():
    parser = argparse.ArgumentParser(prog="lazrnn")
    
    parser.add_argument("--rnn", default='lazrnn', choices=['dummy','lazrnn'], help="which rnn to use")
    parser.add_argument("--rep", default='std', choices=['std','nospace','spec'], help="which representation to use")
    parser.add_argument("--activation", default='bi-lstm', help="activation function for hidden layer: bi-relu bi-lstm bi-tanh")
    parser.add_argument("--fbmerge", default='concat', choices=['concat','sum'], help="how to merge forward backward layer outputs")
    parser.add_argument("--n_hidden", default=[128], nargs='+', type=int, help="number of neurons in each hidden layer")
    parser.add_argument("--recout", default=0, type=int, help="use recurrent output layer")
    parser.add_argument("--batch_norm", default=0, type=int, help="whether to use batch norm between deep layers")
    parser.add_argument("--drates", default=[0, 0], nargs='+', type=float, help="dropout rates")
    parser.add_argument("--opt", default="adam", help="optimization method: sgd, rmsprop, adagrad, adam")
    parser.add_argument("--lr", default=0.001, type=float, help="learning rate")
    parser.add_argument("--norm", default=1, type=float, help="Threshold for clipping norm of gradient")
    parser.add_argument("--n_batch", default=32, type=int, help="batch size")
    parser.add_argument("--fepoch", default=600, type=int, help="number of epochs")
    parser.add_argument("--sample", default=0, type=int, help="num of sents to sample from trn in the order of K")
    parser.add_argument("--feat", default='basic', help="feat func to use")
    parser.add_argument("--emb", default=0, type=int, help="embedding layer size")
    parser.add_argument("--gclip", default=0, type=float, help="clip gradient messages in recurrent layers if they are above this value")
    parser.add_argument("--truncate", default=-1, type=int, help="backward step size")
    parser.add_argument("--log", default='das_auto', help="log file name")
    parser.add_argument("--sorted", default=1, type=int, help="sort datasets before training and prediction")
    parser.add_argument("--in2out", default=0, type=int, help="connect input & output")
    parser.add_argument("--lang", default='eng', help="ner lang")
    parser.add_argument("--charset", default=[], nargs='+', type=str, help="additional dset names for training charset")
    parser.add_argument("--shuf", default=1, type=int, help="shuffle the batches.")
    parser.add_argument("--tagging", default='bio', choices=['io','bio'], help="tag scheme to use")
    parser.add_argument("--decoder", default=1, type=int, help="use decoder to prevent invalid tag transitions")
    parser.add_argument("--breaktrn", default=0, type=int, help="break trn sents to subsents")
    parser.add_argument("--captrn", default=500, type=int, help="consider sents lt this as trn")
    parser.add_argument("--fbias", default=0., type=float, help="forget gate bias")
    parser.add_argument("--eps", default=1e-8, type=float, help="epsilon for adam")
    parser.add_argument("--gnoise", default=False, action='store_true', help="adding time dependent noise to the gradients")
    parser.add_argument("--cdrop", default=0., type=float, help="char dropout rate")
    parser.add_argument("--level", default='char', choices=['char','word'], help="char/word level")

    parser.add_argument("--save", default='', help="save param values to file")
    parser.add_argument("--load", default='', help="load a pretrained model")

    args = vars(parser.parse_args())
    args['drates'] = args['drates'] if any(args['drates']) else [0]*(len(args['n_hidden'])+1)

    return args



class Batcher(object):

    def __init__(self, batch_size, feat):
        self.batch_size = batch_size
        self.feat = feat

    def get_batches(self, dset):
        nf = self.feat.NF 
        sent_batches = [dset[i:i+self.batch_size] for i in range(0, len(dset), self.batch_size)]
        batches = []
        for batch in sent_batches:
            mlen = max(len(sent['x']) for sent in batch)
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
            batches.append((X_batch, Xmsk_batch, y_batch, ymsk_batch))
        return batches

class Reporter(object):

    def __init__(self, dset, feat):
        self.feat = feat
        self.tfunc = rep.get_ts_bio if dset.level == 'char' else lambda x,y: y
        # self.tdecoder = decoder.ViterbiDecoder(dset.trn, feat) if dset.level == 'char' else decoder.MaxDecoder(dset.trn, feat)
        self.tdecoder = decoder.ViterbiDecoder(dset.trn, feat) if dset.level == 'char' else decoder.WDecoder(dset.trn, feat)

    def report_yerr(self, dset, pred):
        pred = [np.argmax(p, axis=-1).flatten() for p in pred]
        y_true = self.feat.yenc.transform([t for sent in dset for t in sent['y']])
        y_pred = list(chain.from_iterable(pred))
        yerr = np.sum(y_true!=y_pred)/float(len(y_true))

        return yerr, 0, 0, 0

    def report(self, dset, pred):
        pred = [self.tdecoder.decode(s, p) for s, p in zip(dset, pred)]
        y_true = self.feat.yenc.transform([t for sent in dset for t in sent['y']])
        y_pred = list(chain.from_iterable(pred))
        yerr = np.sum(y_true!=y_pred)/float(len(y_true))

        # char_conmat_str = self.get_conmat_str(y_true, y_pred, self.feat.tseqenc)

        lts = [sent['ts'] for sent in dset]
        lts_pred = []
        for sent, ipred in zip(dset, pred):
            tseq_pred = self.feat.yenc.inverse_transform(ipred)
            # tseqgrp_pred = get_tseqgrp(sent['wiseq'],tseq_pred)
            ts_pred = self.tfunc(sent['wiseq'],tseq_pred)
            lts_pred.append(ts_pred) # changed

        """
        y_true = self.feat.tsenc.transform([t for ts in lts for t in ts])
        y_pred = self.feat.tsenc.transform([t for ts in lts_pred for t in ts])
        werr = np.sum(y_true!=y_pred)/float(len(y_true))

        word_conmat_str = self.get_conmat_str(y_true, y_pred, self.feat.tsenc)
        """

        # wacc, pre, recall, f1 = bilouEval2(lts, lts_pred)
        (wacc, pre, recall, f1), conll_print = conlleval(lts, lts_pred)
        logging.debug('')
        logging.debug(conll_print)
        # logging.debug(char_conmat_str)
        # logging.debug(word_conmat_str)
        logging.debug('')
        return yerr, pre, recall, f1

    def get_conmat_str(self, y_true, y_pred, lblenc):
        str_list = []
        str_list.append('\t'.join(['bos'] + list(lblenc.classes_)))
        conmat = confusion_matrix(y_true,y_pred, labels=lblenc.transform(lblenc.classes_))
        for r,clss in zip(conmat,lblenc.classes_):
            str_list.append('\t'.join([clss] + list(map(str,r))))
        return '\n'.join(str_list) + '\n'

class Validator(object):

    def __init__(self, dset, batcher, reporter):
        self.dset = dset
        self.trndat = batcher.get_batches(dset.trn) 
        self.devdat = batcher.get_batches(dset.dev) 
        self.tstdat = batcher.get_batches(dset.tst) 
        self.reporter = reporter
        self.batcher = batcher

    def validate(self, rdnn, argsd):
        logging.info('training the model...')
        dbests = {'trn':(1,0.), 'dev':(1,0.), 'tst':(1,0.)}

        for e in range(1,argsd['fepoch']+1): # foreach epoch
            logging.info(('{:<5} {:<5} {:>12} ' + ('{:>10} '*7)).format('dset','epoch','mcost', 'mtime', 'yerr', 'pre', 'recall', 'f1', 'best', 'best'))
            """ training """
            if argsd['cdrop'] > 0: # TODO
                trndat = self.batcher.get_batches(self.dset.trn) 
            trndat = copy.copy(self.trndat)
            if argsd['shuf']:
                random.shuffle(trndat) 

            start_time = time.time()
            mcost = rdnn.train(trndat)
            end_time = time.time(); mtime = end_time - start_time
            logging.info(('{:<5} {:<5d} {:>12.4e} {:>10.4f}').format('trn0',e,mcost, mtime))
            """ end training """

            """ predictions """
            for ddat, datname, dset in zip([self.trndat,self.devdat, self.tstdat],['trn','dev','tst'], [self.dset.trn, self.dset.dev, self.dset.tst]):
            # for ddat, datname, dset in zip([self.devdat, self.tstdat],['dev','tst'], [self.dev, self.tst]):
                start_time = time.time()
                mcost, pred = rdnn.predict(ddat)
                pred = [p for b in pred for p in b]
                end_time = time.time()
                mtime = end_time - start_time
                
                if datname=='trn':
                    yerr, pre, recall, f1 = self.reporter.report_yerr(dset, pred) 
                else:
                    yerr, pre, recall, f1 = self.reporter.report(dset, pred) 
                
                if f1 > dbests[datname][1]:
                    dbests[datname] = (e,f1)
                    if argsd['save'] and datname == 'dev': # save model to file
                        rnn_param_values = rdnn.get_param_values()
                        np.savez('{}/{}'.format(MODEL_DIR, argsd['save']), argsd=argsd, rnn_param_values=rnn_param_values)

                logging.info(('{:<5} {:<5d} {:>12.4e} ' + ('{:>10.4f} '*6)+'{:>10d}')\
                    .format(datname, e, mcost, mtime, yerr, pre, recall, f1, dbests[datname][1], dbests[datname][0]))
            """ end predictions """
            logging.info('')



def setup_logger(args):
    LPARAMS = ['activation', 'n_hidden', 'fbmerge', 'drates',
        'recout','decoder', 'opt','lr','norm','gclip','truncate','n_batch', 'shuf',
        'breaktrn', 'captrn', 'emb','lang', 'fbias']
    import socket
    host = socket.gethostname().split('.')[0]
    logger = logging.getLogger()
    logger.setLevel(logging.DEBUG)
    shandler = logging.StreamHandler()
    shandler.setLevel(logging.INFO)
    param_log_name = ','.join(['{}:{}'.format(p,args[p]) for p in LPARAMS])
    param_log_name = valid_file_name(param_log_name)
    base_log_name = '{}:{},{}'.format(host, theano.config.device, param_log_name if args['log'] == 'das_auto' else args['log'])
    ihandler = logging.FileHandler('{}/{}.info'.format(LOG_DIR,base_log_name), mode='w')
    ihandler.setLevel(logging.INFO)
    dhandler = logging.FileHandler('{}/{}.debug'.format(LOG_DIR,base_log_name), mode='w')
    dhandler.setLevel(logging.DEBUG)
    logger.addHandler(shandler);logger.addHandler(ihandler);logger.addHandler(dhandler);

    # print args
    for k,v in sorted(args.iteritems()):
        logger.info('{}:\t{}'.format(k,v))
    logger.info('{}:\t{}'.format('base_log_name',base_log_name))

def main():
    args = get_args()
    setup_logger(args)

    dset = Dset(**args)
    feat = featchar.Feat(args['feat'])
    feat.fit(dset, xdsets=[Dset(dname) for dname in args['charset']])

    batcher = Batcher(args['n_batch'], feat)
    reporter = Reporter(dset, feat)

    validator = Validator(dset, batcher, reporter)

    RNN = RDNN_Dummy if args['rnn'] == 'dummy' else RDNN
    rdnn = RNN(feat.NC, feat.NF, args)
    validator.validate(rdnn, args)

if __name__ == '__main__':
    main()
