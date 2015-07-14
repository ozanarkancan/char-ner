from utils import get_sents
import random
from itertools import *
from future_builtins import map, filter, zip
from collections import Counter


def w2seq(sent,src='tsg',dst='tseqg'):
    tseq, wiseq = [], []
    assert len(sent[src]) > 0
    for w, t, wi in zip(sent['ws'],sent[src],count(0)):
        tp, sep, ttype = (t, '', '') if t == 'O' else (t.split('-')[0], '-', t.split('-')[1])
        tseq.extend([''.join([tp.lower(),sep,ttype]) for c in w])
        wiseq.extend([wi for c in w])

        # handle space
        if tp == 'B' or tp == 'I':
            tseq.append('i-'+ttype)
        else: # U, L, O
            tseq.append('o')
        wiseq.append(-1)

    # discard last elements
    sent['cseq'] = [c for w in sent['ws'] for c in w+' '][:-1]
    sent[dst] = tseq[:-1]
    sent['wiseq'] = wiseq[:-1]

def seq2w(sent, src='tseqp', dst='tsp'):
    assert len(sent[src]) > 0
    tgroup = [[e[0] for e in g] for k, g in groupby(enumerate(sent['wiseq']),lambda x: x[1] !=-1) if k]
    tseqgrp = [[sent[src][ti] for ti in ts] for ts in tgroup]
    sent[dst] = [Counter(tseq).most_common(1)[0][0].upper() for tseq in tseqgrp]

if __name__ == '__main__':
    trn, dev, tst = get_sents()
    sent = random.choice(trn)
    print ' '.join(sent['ws']), ' '.join(sent['ts'])

    w2seq(sent)
    for c, ct in zip(sent['cseq'], sent['tseqg']):
        print c, ct

    seq2w(sent,src='tseqg',dst='tsp')
    print ' '.join(sent['tsg'])
    print ' '.join(sent['tsp'])
    # print [[e for e in v] for k,v in groupby(sent['ts'], lambda x: x=='O')]
