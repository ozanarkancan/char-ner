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
    enc = 'latin1' if lang in ['eng','deu','spa','ned', 'ita'] else 'utf-8'
    trn,dev,tst = map(read_sents, ['{}/{}/{}.bio'.format(DATA_DIR,lang,dset) for dset in ('train','testa','testb')], [enc for i in range(3)])
    trn = filter(lambda sent: len(sent['ws'])<1000,trn)
    return trn,dev,tst 

def read_sents(file, enc, delim='\t'):
    a = []
    sentences = []
    with open(file) as src:
        for l in src:
            if len(l.strip()):
                a.append(l.strip().split(delim))
            else: # emtpy line
                if len(a):
                    ws = [el[0].decode(enc) for el in a] # TODO
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

def top10(lang):
    langs = ['eng', 'deu', 'spa', 'ned', 'tr', 'cze', 'ger', 'arb']
    for l in langs:
        trn,dev,tst = get_sents(l)
        print l, top10(l)
    return list(islice(reversed(sorted(len(sent['ws']) for sent in trn)),20))
    # return sum(1 for sent in trn if len(sent['ws'])>50)

def ff(o_list):
    N = len(o_list)
    k = 5
    if N > 5:
        ii = range(0,N-k,k)
        ll = [o_list[i:i+k] for i in ii[:-1]]
        return ll + [o_list[ii[-1]:]]
    else:
        return [o_list]

def break2subsents(sent):
    phrases = []
    for k,g in groupby(sent['ts'],lambda x:x=='O'):
        g = list(g)
        if k:
            phrases.extend(ff(g))
        else:
            phrases.append(g)
    """ for p in phrases:
        print p,p[0]=='O' """

    tt = [p[0]=='O' for p in phrases]
    ii =  [i+1 for i,(t1,t2) in enumerate(zip(tt,tt[1:])) if t1==t2]
    # print ii
    # print
    pi = [0]+ii+[len(phrases)]
    subsents=[]
    tphrases = [[t for t in chain.from_iterable(phrases[s:e])] for s,e in zip(pi,pi[1:])]
    cnt = count(0)
    for tphrase in tphrases:
        nsent = {}
        nsent['ws'] = [sent['ws'][cnt.next()] for t in tphrase]
        nsent['ts'] = tphrase
        subsents.append(nsent)
    return subsents

if __name__ == '__main__':
    trn,dev,tst = get_sents('ita')
    sent = random.choice(trn)
    for w,t in zip(sent['ws'],sent['ts']):
        print w,t
        for c in w:
            print c
        print 

    
