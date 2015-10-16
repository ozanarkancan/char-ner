import random
import numpy as np
from collections import Counter
from itertools import *

import encoding
from utils import get_sents

""" assumes sents are in iob format """



### SENT ###
def contains_consecutive_same_type(sent):
    encoding.iob2
    for t1,t2 in zip(sent['ts'], sent['ts'][1:]):
        if t1 != 'O' and t2 != 'O':
            t1_pos, t1_type = t1.split('-')
            t2_pos, t2_type = t2.split('-')
            if t1_type == t2_type:
                if (t1_pos,t2_pos) in [e for e in product(['U','L'],['B','U'])]:
                    return True
    return False

def num_of_phrases_sent(sent):
    ts = encoding.iob2bilou(sent['ts'])
    return sum(1 for t in ts if t.startswith('B') or t.startswith('U'))

### end SENT ###

### DSET ###

def stat_num_of_chars(dset):
    a = np.mean([len(''.join(sent['ws'])) for sent in dset])
    return np.mean(a), np.std(a)

def stat_num_of_words(dset):
    a = [len(sent['ws']) for sent in dset]
    return np.mean(a), np.std(a)

def num_of_consecutive_same_type(dset):
    return sum(1 for sent in dset for t in sent['ts'] if t.startswith('B'))

def entity_tagged_vocab(dset):
    return set(w for sent in dset for w,t in zip(sent['ws'],sent['ts']) if t != 'O')

def unk_perc(trn, dset):
    trn_vocab = entity_tagged_vocab(trn)
    cntr = Counter(w for sent in dset for w,t in zip(sent['ws'],sent['ts']) if t != 'O')
    z = sum(v for k,v in cntr.iteritems())
    nom = sum(v for k,v in cntr.iteritems() if not k in trn_vocab)
    return nom/float(z)

def num_of_phrases(dset):
    return sum(num_of_phrases_sent(sent) for sent in dset)

def io_ideal(dev,tst):
    from score import conlleval
    print 'io tagging ideal scores'
    for dset, dset_str in zip((dev,tst),('dev','tst')):
        ts_gold = [sent['ts'] for sent in dset]
        ts_pred = [encoding.io2iob(encoding.iob2io(sent['ts'])) for sent in dset]
        r1,r2 = conlleval(ts_gold, ts_pred)
        print '\t'.join([dset_str]+map(str,r1))

def vocab(dset):
    return set(w for sent in dset for w in sent['ws'])

### end DSET ###

if __name__ == '__main__':
    from tabulate import tabulate
    from score import conlleval
    """
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('lang')
    args = parser.parse_args()
    """

    langs = ['eng','deu','ned']
    dsetnames = ['trn','dev','tst']

    data = dict((lang,dict((dname,dset) for dname,dset in zip(dsetnames, get_sents(lang)))) for lang in langs)

    table = []
    for dname in dsetnames:
        table.append([dname]+map(len,[data[l][dname] for l in langs]))
    print tabulate(table,headers=['#sent']+langs)
    print

    table = []
    for l, dname in product(langs,('dev','tst')):
        dset = data[l][dname]
        ts_gold = [sent['ts'] for sent in dset]
        ts_pred = [encoding.io2iob(encoding.iob2io(sent['ts'])) for sent in dset]
        r1,r2 = conlleval(ts_gold, ts_pred)
        table.append([l+'-'+dname]+map(str,r1))
    print tabulate(table, headers=['io-ideal', 'wacc','pre','rec','f1'])
    print

    """
    print 'num of consecutive same type:'
    for dstr, dset in zip(('trn','dev','tst'),(trn,dev,tst)):
        print '{}\t{:.4f}'.format(dstr,num_of_consecutive_same_type(dset)/float(num_of_phrases(dset)))
    print

    a,b,c = map(vocab, (trn,dev,tst))
    print 'vocab'
    print 'trn\tdev\ttst'
    print '{}\t{}\t{}'.format(*map(len,(a,b,c)))
    print
    print 'unk\tdev\t{:.2f}'.format(len(b.difference(a)) / float(len(b)))
    print 'unk\ttst\t{:.2f}'.format(len(c.difference(a)) / float(len(c)))
    """

    """
    a,b,c = map(entity_tagged_vocab, (trn,dev,tst))
    print 'trn, dev, tst:', map(len,(a,b,c))
    print 'dev diff:', len(b.difference(a)) / float(len(b))
    print 'tst diff:', len(c.difference(a)) / float(len(c))

    print '-->', unk_perc(trn,dev)
    print '-->', unk_perc(trn,tst)
    """

