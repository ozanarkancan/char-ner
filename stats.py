from utils import get_sents
import numpy as np
from collections import Counter
from featchar import get_tseq2
import random

class Stats:
    def __init__(self):
        trn,dev,tst = get_sents()
        print np.mean([len(' '.join(sent['ws'])) for sent in trn])
        print np.std([len(' '.join(sent['ws'])) for sent in trn])

def char_tag_dist(trn):
    tcounts =  Counter(t for sent in trn for t in sent['tseq'])
    z = sum(tcounts.values())
    tpercs = sorted((tcount/float(z), t) for t,tcount in tcounts.iteritems())
    for tperc,t in tpercs:
        print '%.4f\t%s'%(tperc,t)

if __name__ == '__main__':
    trn,dev,tst = get_sents()
    for sent in trn:
        sent['tseq'] = get_tseq2(sent)

    sent = trn[3]
    print sent['ts']
    # for a,b in zip(sent['ts'], sent['ts'][1:])

