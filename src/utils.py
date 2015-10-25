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
    return map(read_sents, ['{}/{}/{}.bio'.format(DATA_DIR,lang,dset) for dset in ('train','testa','testb')])

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
                    ts = [el[-1] for el in a]
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

if __name__ == '__main__':
    trn,dev,tst = get_sents('eng')
    sents = sample_sents(trn,10)
    sent = sents[0]
    print sent['ws']
    print sent['ts']
    print encoding.any2io(sent['ts'])
