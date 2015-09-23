from utils import get_sents
import numpy as np
from collections import Counter
from featchar import get_tseq2, get_cseq
import random
from itertools import product
from collections import Counter

def stat_num_of_chars(dset):
    a = np.mean([len(''.join(sent['ws'])) for sent in dset])
    return np.mean(a), np.std(a)

def stat_num_of_words(dset):
    a = [len(sent['ws']) for sent in dset]
    return np.mean(a), np.std(a)

def char_tag_dist(trn):
    tcounts =  Counter(t for sent in trn for t in sent['tseq'])
    z = sum(tcounts.values())
    tpercs = sorted((tcount/float(z), t) for t,tcount in tcounts.iteritems())
    for tperc,t in tpercs:
        print '%.4f\t%s'%(tperc,t)

def contains_consecutive_same_type(sent):
    for t1,t2 in zip(sent['ts'], sent['ts'][1:]):
        if t1 != 'O' and t2 != 'O':
            t1_pos, t1_type = t1.split('-')
            t2_pos, t2_type = t2.split('-')
            if t1_type == t2_type:
                if (t1_pos,t2_pos) in [e for e in product(['U','L'],['B','U'])]:
                    return True
    return False

def num_of_phrases_sent(sent):
    return sum(1 if t.startswith('B') or t.startswith('U') else 0 for t in sent['ts'])

def num_of_consecutive_same_type(dset):
    c = 0
    for sent in dset:
        for t1,t2 in zip(sent['ts'], sent['ts'][1:]):
            if t1 != 'O' and t2 != 'O':
                t1_pos, t1_type = t1.split('-')
                t2_pos, t2_type = t2.split('-')
                if t1_type == t2_type:
                    c += (t1_pos,t2_pos) in [e for e in product(['U','L'],['B','U'])]
    return c

def entity_tagged_vocab(dset):
    return set(w for sent in dset for w,t in zip(sent['ws'],sent['ts']) if t != 'O')

def unk_perc(trn, dset):
    trn_vocab = entity_tagged_vocab(trn)
    cntr = Counter(w for sent in dset for w,t in zip(sent['ws'],sent['ts']) if t != 'O')
    z = sum(v for k,v in cntr.iteritems())
    nom = sum(v for k,v in cntr.iteritems() if not k in trn_vocab)
    return nom/float(z)

def num_of_phrases(dset):
    return sum(1 for sent in dset for t in sent['ts'] \
            if t.startswith('B') or t.startswith('U'))

if __name__ == '__main__':
    trn,dev,tst = get_sents()
    for sent in trn:
        sent['tseq'] = get_tseq2(sent)
        sent['cseq'] = get_cseq(sent)
        if len(sent['cseq']) == 512:
            print ' '.join(sent['ws'])

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
    print 'cseq len counts:'
    for k,v in sorted(Counter(len(sent['cseq']) for sent in trn).iteritems()):
        print '{}\t{}'.format(k,v)
    """
