import string
from itertools import *

import utils

def seg(ci, sent):
    d = {}
    # wstart
    if ci==0:
        d['wstart'] = 1
    elif ci>0:
        d['wstart'] = sent['wiseq'][ci-1] != sent['wiseq'][ci]

    # wend
    if ci==(len(sent['cseq'])-1):
        d['wend'] = 1
    elif ci<(len(sent['cseq'])-1):
        d['wend'] = sent['wiseq'][ci+1] != sent['wiseq'][ci]
    return d


def get_cfeatures_basic(ci, sent):
    return {'c': sent['cseq'][ci]}

def get_cfeatures_basic_seg(ci, sent):
    d = {}
    d['c'] = sent['cseq'][ci]

    d.update(seg(ci,sent)) # seg
    return d


def get_cfeatures_basic_seg_pos(ci, sent):
    d = {}
    d['c'] = sent['cseq'][ci]
    wi = sent['wiseq'][ci]
    d['pt'] = 'space_pt' if d['c'] == ' ' else sent['pts'][wi]

    d.update(seg(ci,sent)) # seg

    return d

def get_cfeatures_basic_seg_cap(ci, sent):
    d = {}
    d['c'] = sent['cseq'][ci]

    d.update(seg(ci,sent)) # seg
    # capitilization
    d['is_capital'] = sent['cseq'][ci].isupper()

    return d

def get_cfeatures_gen(ci, sent):
    c = sent['cseq'][ci]
    if c in string.ascii_letters:
        return {'c': c}
    elif c in (utils.WSTART, utils.WEND):
        return {'c': c}
    elif c == ' ':
        return {'c': 'space'}
    elif c in string.digits:
        return {'c': 'digit'}
    elif c in string.punctuation:
        return {'c': 'punc'}
    else:
        return {'c': 'other'}

def get_cfeatures_gen_seg(ci, sent):
    d = {}
    c = sent['cseq'][ci]
    if c in string.ascii_letters:
        d['c'] = c
    elif c in (utils.WSTART, utils.WEND):
        d['c'] = c
    elif c == ' ':
        d['c'] = 'space'
    elif c in string.digits:
        d['c'] = 'digit'
    elif c in string.punctuation:
        d['c'] = 'punc'
    else:
        d['c'] = 'other'

    d.update(seg(ci,sent)) # seg

    return d

def get_cfeatures_simple_seg(ci, sent):
    d = {}
    c = sent['cseq'][ci]
    if c in 'abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ':
        d['c'] = c.lower()
    else:
        d['c'] = 'not_letter'
    d['isupper'] = c.isupper()
    d['isdigit'] = c.isdigit()
    d['ispunc'] = c in string.punctuation

    d.update(seg(ci,sent)) # seg

    return d

if __name__ == '__main__':
    import rep
    trn, dev, tst = utils.get_sents('eng')
    sents = utils.sample_sents(trn, 5, 6, 8)

    repstd = rep.Repstd()
    for sent in sents:
        sent['cseq'] = repstd.get_cseq(sent)
        sent['wiseq'] = repstd.get_wiseq(sent)
        for ci,c in enumerate(sent['cseq']):
            featd = get_cfeatures_basic_seg_pos(ci, sent) 
            print featd['c'], featd['pt']
        print 
        

