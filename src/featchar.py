import string
from itertools import *

import utils


def get_cfeatures_basic(ci, sent):
    return {'c': sent['cseq'][ci]}

def get_cfeatures_basic_seg(ci, sent):
    d = {}
    d['c'] = sent['cseq'][ci]

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

    # if sent['wiseq'][ci] == -1: d['isspace'] = 1
    return d

def get_cfeatures_basic_seg_cap(ci, sent):
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

    # if sent['wiseq'][ci] == -1: d['isspace'] = 1

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

    # wstart
    if ci==0: d['wstart'] = 1
    if ci>0:
        d['wstart'] = sent['wiseq'][ci-1] == -1

    # wend
    if ci==(len(sent['cseq'])-1): d['wend'] = 1
    if ci<(len(sent['cseq'])-1):
        d['wend'] = sent['wiseq'][ci+1] == -1

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

