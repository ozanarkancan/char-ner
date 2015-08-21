from utils import get_sents
import random
from itertools import *
# from future_builtins import map, filter, zip
from collections import Counter
from sklearn.feature_extraction import DictVectorizer
from sklearn.preprocessing import LabelEncoder
import numpy as np
import string


def get_tseq1(sent):
    """ majority """
    tseq = []
    for w, t in zip(sent['ws'],sent['ts']):
        tp, sep, ttype = (t, '', '') if t == 'O' else (t.split('-')[0], '-', t.split('-')[1])
        tseq.extend([''.join([tp.lower(),sep,ttype]) for c in w])

        # handle space
        if tp == 'B' or tp == 'I':
            tseq.append('i-'+ttype)
        else: # U, L, O
            tseq.append('o')
    return tseq[:-1]

def get_tseq2(sent):
    tseq = []
    for w, t in zip(sent['ws'], sent['ts']):
        tp, sep, ttype = (t, '', '') if t == 'O' else (t.split('-')[0], '-', t.split('-')[1])
        if tp == 'U':
            if len(w) > 1:
                tseq.append('b-'+ttype)
                for c in w[1:-1]:
                    tseq.append('i-'+ttype)
                tseq.append('l-'+ttype)
            else:
                tseq.append('u-'+ttype)
        elif tp == 'B':
            tseq.append('b-'+ttype)
            for c in w[1:]:
                tseq.append('i-'+ttype)
        elif tp == 'L':
            for c in w[:-1]:
                tseq.append('i-'+ttype)
            tseq.append('l-'+ttype)
        else: # I O
            for c in w:
                tseq.append(tp.lower()+sep+ttype)

        # handle space
        if tp == 'B' or tp == 'I':
            tseq.append('i-'+ttype)
        else: # U, L, O
            tseq.append('o')
    return tseq[:-1]

def get_tseqgrp(wiseq, tseq):
    tgroup = [[e[0] for e in g] for k, g in groupby(enumerate(wiseq),lambda x: x[1] !=-1) if k]
    tseqgrp = [[tseq[ti] for ti in ts] for ts in tgroup]
    return tseqgrp

def get_ts1(tseqgrp):
    return [Counter(tseq).most_common(1)[0][0].upper() for tseq in tseqgrp]

def get_ts2(tseqgrp):
    ts = []
    for tseq in tseqgrp:
        if tseq[0].startswith('b-') and tseq[-1].startswith('i-'): # B
            ts.append(tseq[0].upper())
        elif tseq[0].startswith('i-') and tseq[-1].startswith('i-'): # I
            ts.append(tseq[0].upper())
        elif tseq[0].startswith('i-') and tseq[-1].startswith('l-'): # L
            ts.append(tseq[-1].upper())
        elif tseq[0].startswith('b-') and tseq[-1].startswith('l-'): # U
            tp, ttype = tseq[0].split('-')
            ts.append('U-'+ttype)
        elif tseq[0].startswith('u-'): # U
            ts.append(tseq[0].upper())
        else: # 
            ts.append(Counter(tseq).most_common(1)[0][0].upper())
    return ts

def get_cseq(sent):
    return [c for c in ' '.join(sent['ws'])]

def get_wiseq(sent):
    wiseq = []
    for wi,w in enumerate(sent['ws']):
        wiseq.extend([wi for c in w])
        wiseq.append(-1)
    return wiseq[:-1]

def get_wfeatures(wi, sent):
    return {'w':sent['ws'][wi]}

def get_cfeatures_basic_seg_cap(ci, sent, tseq_pred=None):
    d = {}
    d['c'] = sent['cseq'][ci]

    # wstart
    if ci==0: d['wstart'] = 1
    if ci>0:
        d['wstart'] = sent['wiseq'][ci-1] == -1


    # wend
    if ci==(len(sent['cseq'])-1): d['wend'] = 1
    if ci<(len(sent['cseq'])-1):
        d['wend'] = sent['wiseq'][ci+1] == -1

    if sent['wiseq'][ci] == -1: d['isspace'] = 1

    # capitilization
    d['is_capital'] = sent['cseq'][ci].isupper()

    return d

def get_cfeatures_basic_seg(ci, sent, tseq_pred=None):
    d = {}
    d['c'] = sent['cseq'][ci]

    # wstart
    if ci==0: d['wstart'] = 1
    if ci>0:
        d['wstart'] = sent['wiseq'][ci-1] == -1

    # wend
    if ci==(len(sent['cseq'])-1): d['wend'] = 1
    if ci<(len(sent['cseq'])-1):
        d['wend'] = sent['wiseq'][ci+1] == -1

    if sent['wiseq'][ci] == -1: d['isspace'] = 1
    return d

def get_cfeatures_basic(ci, sent, tseq_pred=None):
    return {'c': sent['cseq'][ci]}

def get_cfeatures_simple_seg(ci, sent, tseq_pred=None):
    d = {}
    c = sent['cseq'][ci]
    if c in 'abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ':
        d['c'] = c.lower()
    else:
        d['c'] = 'not_letter'
    d['isupper'] = c.isupper()
    d['isdigit'] = c.isdigit()
    d['ispunc'] = c in string.punctuation

    # wstart
    if ci==0: d['wstart'] = 1
    if ci>0:
        d['wstart'] = sent['wiseq'][ci-1] == -1

    # wend
    if ci==(len(sent['cseq'])-1): d['wend'] = 1
    if ci<(len(sent['cseq'])-1):
        d['wend'] = sent['wiseq'][ci+1] == -1

    if sent['wiseq'][ci] == -1: d['isspace'] = 1
    return d

def get_cfeatures_just_tags(ci, sent, tseq_pred=None):
    d = {}
    # previous tag
    if ci == 0:
        d['sent_start'] = 1
    else: # ci > 0:
        d['t'] = sent['tseq'][ci-1]
    return d

def get_cfeatures_prev(ci, sent, tseq_pred=None):
    d = {}
    d['c'] = sent['cseq'][ci]

    # previous tag
    if ci == 0:
        d['sent_start'] = 1
    else: # ci > 0:
        if not tseq_pred is None:
            d['t'] = tseq_pred[ci-1]
        else:
            d['t'] = sent['tseq'][ci-1]

    # wstart
    if ci==0: d['wstart'] = 1
    if ci>0:
        d['wstart'] = sent['wiseq'][ci-1] == -1

    # wend
    if ci==(len(sent['cseq'])-1): d['wend'] = 1
    if ci<(len(sent['cseq'])-1):
        d['wend'] = sent['wiseq'][ci+1] == -1

    if sent['wiseq'][ci] == -1: d['isspace'] = 1
    return d

def sent_word_indx(sent):
    space_indx = [a for a,b in enumerate(sent['wiseq']) if b==-1]
    indx = [0] + space_indx + [len(sent['wiseq'])]
    arr = np.array([[a,b] for a,b in zip(indx,indx[1:])])
    arr[1:,0] += 1
    return arr

if __name__ == '__main__':
    trn, dev, tst = get_sents()
    for d in (trn,dev,tst):
        for sent in d:
            sent.update({
                'cseq': get_cseq(sent), 
                'wiseq': get_wiseq(sent), 
                'tseq': get_tseq2(sent)})

    sent = random.choice(dev)
    print [(a,b) for a,b in enumerate(sent['wiseq'])]
    print sent_word_indx(sent)
