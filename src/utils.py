import os
import copy
from itertools import *
import random

import encoding

__author__ = 'Onur Kuru'
file_abspath = os.path.abspath(__file__)
ROOT_DIR = os.path.abspath(os.path.join(file_abspath, os.pardir, os.pardir))
DATA_DIR = '{}/data'.format(ROOT_DIR)
SRC_DIR = '{}/src'.format(ROOT_DIR)
SCRIPTS_DIR = '{}/scripts'.format(ROOT_DIR)

WSTART = '/w'
WEND = 'w/'

def valid_file_name(s):
    return "".join(i for i in s if i not in "\"\/ &*?<>|[]()'")

def get_sents(lang='eng'):
    trn,dev,tst = map(read_sents, ['{}/{}/{}.bio'.format(DATA_DIR,lang,dset) for dset in ('train','testa','testb')])
    trn = filter(lambda sent: len(sent['ws'])<1000,trn)
    return trn,dev,tst 

def read_sents(file, delim='\t'):
    a = []
    sentences = []
    with open(file) as src:
        for l in src:
            if len(l.strip()):
                a.append(l.strip().split(delim))
            else: # emtpy line
                if len(a):
                    ws = [el[0] for el in a]
                    ts = [el[-1].upper() for el in a]
                    sentences.append({'ws':ws,'ts':ts})
                a = []
    return sentences


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

def get_subsents(sent):
    subsents = []
    cursubsent = {'ws':[],'ts':[]}
    for w,t in zip(sent['ws'],sent['ts']):
        cursubsent['ws'].append(w)
        cursubsent['ts'].append(t)
        if w == '.' and t == 'O':
            subsents.append(cursubsent)
            cursubsent = {'ws':[],'ts':[]}
    if len(cursubsent['ws']) > 0:
        subsents.append(cursubsent)
    return subsents

if __name__ == '__main__':
    trn,dev,tst = get_sents('ned')
    print list(islice(reversed(sorted(len(sent['ws']) for sent in trn)),10))
    print list(islice(reversed(sorted(len(sent['ws']) for sent in dev)),10))
    print list(islice(reversed(sorted(len(sent['ws']) for sent in tst)),10))
    trn1 = []
    for sent in trn:
        if len(sent['ws']) > 1000:
            print ' '.join(sent['ws'])
        trn1.extend(get_subsents(sent))
    print list(islice(reversed(sorted(len(sent['ws']) for sent in trn1)),10))
    print len(trn), len(trn1)
