import os
import copy
from itertools import *
import random

__author__ = 'Onur Kuru'
file_abspath = os.path.abspath(__file__)
ROOT_DIR = os.path.abspath(os.path.join(file_abspath, os.pardir, os.pardir))
DATA_DIR = '{}/data'.format(ROOT_DIR)
WSTART = '/w'
WEND = 'w/'

def read_sents(file):
    a,b,c,d = [],[],[],[]
    sentences = []
    with open(file) as src:
        for l in src:
            if len(l.strip()):
                w, pt, ct, t = l.strip().split('\t')
                a.append(w);b.append(pt);
                c.append(ct);d.append(t);
            else: # emtpy line
                if len(a):
                    sentences.append({'ws':a,'ts':d,'tsg':copy.deepcopy(d),\
                            'pts':b,'cts':c})
                a,b,c,d = [],[],[],[]
    return sentences

def read_sents_deu(file):
    a,b,c,d,e = [],[],[],[], []
    sentences = []
    with open(file) as src:
        for l in src:
            if len(l.strip()):
                w, lem, pt, ct, t = l.strip().split('\t')
                a.append(w);b.append(pt);
                c.append(ct);d.append(t);e.append(lem)
            else: # emtpy line
                if len(a):
                    sentences.append({'ws':a,'ts':d,'tsg':copy.deepcopy(d),\
                            'pts':b,'cts':c,'lems':e})
                a,b,c,d = [],[],[],[]
    return sentences

def read_sents_spa(file):
    a,b,c,d = [],[],[],[]
    sentences = []
    with open(file) as src:
        for l in src:
            if len(l.strip()):
                w, pt, t = l.strip().split('\t')
                a.append(w);b.append(pt);d.append(t);
            else: # emtpy line
                if len(a):
                    sentences.append({'ws':a,'ts':d,'tsg':copy.deepcopy(d),\
                            'pts':b})
                a,b,c,d = [],[],[],[]
    return sentences

def get_sents(lang='eng', enc='bilou'):
    if lang=='eng':
        return read_sents('%s/%s/train.%s'%(DATA_DIR,lang,enc)),\
                read_sents('%s/%s/testa.%s'%(DATA_DIR,lang,enc)),\
                read_sents('%s/%s/testb.%s'%(DATA_DIR,lang,enc))
    elif lang=='spa':
        return read_sents_spa('%s/%s/train.%s'%(DATA_DIR,lang,enc)), read_sents('%s/%s/testa.%s'%(DATA_DIR,lang,enc)), read_sents('%s/%s/testb.%s'%(DATA_DIR,lang,enc))
    else:
        raise Exception

def sample_sents(sents, n, min_ws_len=None, max_ws_len=None):
    pp = sents
    if min_ws_len: pp = ifilter(lambda x: len(x['ws']) >= min_ws_len, pp)
    if max_ws_len: pp = ifilter(lambda x: len(x['ws']) <= max_ws_len, pp)
    return random.sample(list(pp), n)

def get_sent_indx(dset):
    start = 0
    indexes = []
    for sent in dset:
        indexes.append((start,start+len(sent['cseq'])))
        start += len(sent['cseq'])
    return indexes

def get_sent_indx_word(dset):
    start = 0
    indexes = []
    for sent in dset:
        indexes.append((start,start+len(sent['ws'])))
        start += len(sent['ws'])
    return indexes

if __name__ == '__main__':
    trn = read_sents_deu('data/deu/train.iob')

    sent = trn[0]
    for w in sent['ws']:
        print [c for c in w]
