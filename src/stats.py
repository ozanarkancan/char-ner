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

"""
def unk_perc(trn, dset):
    trn_vocab = entity_tagged_vocab(trn)
    cntr = Counter(w for sent in dset for w,t in zip(sent['ws'],sent['ts']) if t != 'O')
    z = sum(v for k,v in cntr.iteritems())
    nom = sum(v for k,v in cntr.iteritems() if not k in trn_vocab)
    return nom/float(z)
"""

def num_of_phrases(dset):
    return sum(num_of_phrases_sent(sent) for sent in dset)

def io_ideal(dev,tst):
    from score import conlleval
    for dset in (dev,tst):
        ts_gold = [sent['ts'] for sent in dset]
        ts_pred = [encoding.io2iob(encoding.iob2io(sent['ts'])) for sent in dset]
        r1,r2 = conlleval(ts_gold, ts_pred)
        print r2

### end DSET ###

if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('lang')
    args = parser.parse_args()

    trn,dev,tst = get_sents(args.lang)

    io_ideal(dev,tst)

    """
    print 'onemli:', num_of_consecutive_same_type(trn)
    print 'onemli2:', num_of_consecutive_same_type(dev)
    print 'onemli3:', num_of_consecutive_same_type(tst)

    print num_of_phrases([sent for sent in trn if not contains_consecutive_same_type(sent)]) / float(num_of_phrases(trn))
    print num_of_phrases([sent for sent in dev if not contains_consecutive_same_type(sent)]) / float(num_of_phrases(dev))
    print num_of_phrases([sent for sent in tst if not contains_consecutive_same_type(sent)]) / float(num_of_phrases(tst))

    a,b,c = map(entity_tagged_vocab, (trn,dev,tst))
    print 'trn, dev, tst:', map(len,(a,b,c))
    print 'dev diff:', len(b.difference(a)) / float(len(b))
    print 'tst diff:', len(c.difference(a)) / float(len(c))

    print '-->', unk_perc(trn,dev)
    print '-->', unk_perc(trn,tst)
    """

