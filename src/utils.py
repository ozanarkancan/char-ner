import copy
from itertools import *
import random, numpy as np
from sklearn.feature_extraction import DictVectorizer
from sklearn.preprocessing import LabelEncoder

__author__ = 'Onur Kuru'
import os
file_abspath = os.path.abspath(__file__)
ROOT_DIR = os.path.abspath(os.path.join(file_abspath, os.pardir, os.pardir))
DATA_DIR = '{}/data'.format(ROOT_DIR)

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

def get_cfeatures(wi, ci, sent):
    return {'c':sent['cseq'][ci]}


def extend_sent2(sent):
    cseq, tseq, wiseq = [], [], []
    wi = 0
    for w, t in zip(sent['ws'], sent['ts']):
        if t == 'O': tp, ttype = 'O', 'O'
        else: tp, ttype = t.split('-')
        if ttype == 'ORG': ttype = 'G' + ttype
        ttype = ttype[0].lower()

        cseq.extend([c for c in w+' '])
        wiseq.extend([wi for c in w+' '])
        wi+=1
        if tp == 'U':
            for c in w:
                tseq.append(ttype)
            tseq.append(ttype+'-l')
        elif tp == 'B':
            for c in w:
                tseq.append(ttype)
            tseq.append(ttype)
        elif tp == 'L':
            for c in w:
                tseq.append(ttype)
            tseq.append(ttype+'-l')
        elif tp == 'I':
            for c in w:
                tseq.append(ttype)
            tseq.append(ttype)
        else: # O
            for c in w:
                tseq.append(ttype)
            tseq.append(ttype+'-l')
    sent['cseq'] = cseq
    sent['tseq'] = tseq
    sent['wiseq'] = wiseq

def extend_sent(sent):
    cseq, tseq, wiseq, ciseq = [], [], [], []
    wi, ci = 0, 0
    for w, t in zip(sent['ws'], sent['ts']):
        if t == 'O': tp, ttype = 'O', 'O'
        else: tp, ttype = t.split('-')
        # ttype = ttype[0]

        cseq.extend([c for c in w])
        wiseq.extend([wi for c in w])
        cseq.append(' ') # for space
        wiseq.append(-1) # for space
        # wiseq.append(wi) # for space
        wi+=1
        if tp == 'U':
            if len(w) > 1:
                tseq.append('b-'+ttype)
                for c in w[1:-1]:
                    tseq.append('i-'+ttype)
                tseq.append('l-'+ttype)
            else:
                tseq.append('u-'+ttype)
            tseq.append('o') # for space
        elif tp == 'B':
            tseq.append('b-'+ttype)
            for c in w[1:]:
                tseq.append('i-'+ttype)
            tseq.append('i-'+ttype) # for space
        elif tp == 'L':
            for c in w[:-1]:
                tseq.append('i-'+ttype)
            tseq.append('l-'+ttype)
            tseq.append('o') # for space
        elif tp == 'I':
            for c in w:
                tseq.append('i-'+ttype)
            tseq.append('i-'+ttype) # for space
        else: # O
            for c in w:
                tseq.append('o')
            tseq.append('o') # for space
    sent['cseq'] = cseq[:-1]
    sent['tseq'] = tseq[:-1]
    sent['wiseq'] = wiseq[:-1]

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

def tseq2ts(sent):
    cindx = [[t[1] for t in g] for k,g in groupby(izip(sent['wiseq'],count(0)),lambda x:x[0]) if k!=-1]
    assert len(sent['ws']) == len(cindx)
    tcseq = [[sent['tpredseq'][i] for i in ind] for ind in cindx]
    ts = []
    for tseq in tcseq:
        ttype = tseq[0].split('-')[1] if '-' in tseq[0] else 'O'  # different decision can be used
        if tseq[0].startswith('b') and tseq[-1].startswith('i'):
            ts.append('B-'+ttype)
        elif tseq[0].startswith('i') and tseq[-1].startswith('i'):
            ts.append('I-'+ttype)
        elif tseq[0].startswith('i') and tseq[-1].startswith('l'):
            ts.append('L-'+ttype)
        elif tseq[0].startswith('b') and tseq[-1].startswith('l') or tseq[0].startswith('u'):
            ts.append('U-'+ttype)
        elif len(tseq)==1 and tseq[0].startswith('l'):
            ts.append('L-'+ttype)
        elif len(tseq)==1 and tseq[0].startswith('b'):
            ts.append('B-'+ttype)
        else:
            ts.append('O')
    return ts

if __name__ == '__main__':
    print ROOT_DIR
    pass

