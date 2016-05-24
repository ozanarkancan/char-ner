from itertools import chain, groupby, izip
from collections import Counter

import utils

def get_phrases(ts):
    phrases, curphrase = [], []
    for t in ts:
        if t.startswith('B'):
            len(curphrase) and phrases.append(curphrase)
            curphrase = [t]
        elif t.startswith('I'):
            curphrase.append(t)
        elif t.startswith('O'):
            len(curphrase) and phrases.append(curphrase)
            curphrase = [t]
        else:
            raise Exception()
    len(curphrase) and phrases.append(curphrase)
    return phrases

class Repstd(object):

    def get_cseq(self, sent):
        return [c for c in ' '.join(sent['ws'])]

    def get_wiseq(self, sent):
        wiseq = []
        for wi,w in enumerate(sent['ws']):
            wiseq.extend([wi for c in w])
            wiseq.append(-1)
        return wiseq[:-1]

    def get_tseq(self, sent):
        tseq, ts = [], sent['ts']
        # ts = encoding.any2io(sent['ts'])
        for w, t, tnext in zip(sent['ws'],ts, chain(ts[1:],[None])):
            if t == 'O':
                tseq.extend('o' for c in w)
                tseq.append('o')
            else: # starts with B or I
                tp, ttype = t.split('-')
                tseq.extend('i-'+ttype.lower() for c in w)

                # handle space
                if tnext and tnext.startswith('I-') and ttype == tnext.split('-')[1]:
                    tseq.append('i-'+ttype.lower())
                else:
                    tseq.append('o')
        return tseq[:-1]

    def pprint(self, sent, *margs):
        args = [sent['cseq'],sent['tseq']]; args.extend(margs)
        mlen = max(len(e) for e in chain(*args))
        wgroup = [[e[0] for e in g] for k, g in groupby(enumerate(sent['wiseq']),lambda x: x[1] !=-1) if k]
        space_indx = [i for i,wi in enumerate(sent['wiseq']) if wi==-1]
        for cil, si in izip(wgroup, space_indx):
            for a in args:
                print ' '.join(map(('{:^%d}'%mlen).format, [a[ci] for ci in cil] + [a[si]]))


class Repnospace(object):

    def get_cseq(self, sent):
        return [c for c in ''.join(sent['ws'])]

    def get_wiseq(self, sent):
        return [wi for wi,w in enumerate(sent['ws']) for c in w]

    def get_tseq(self, sent): 
        return [sent['ts'][wi].lower() for wi,w in enumerate(sent['ws']) for c in w]

class Repspec(object):

    def get_cseq(self, sent):
        return [c for w in sent['ws'] for c in [utils.WSTART]+list(w)+[utils.WEND]]

    def get_wiseq(self, sent):
        return [wi for wi,w in enumerate(sent['ws']) for c in [utils.WSTART]+list(w)+[utils.WEND]]

    def get_tseq(self, sent):
        return [sent['ts'][wi].lower() for wi,w in enumerate(sent['ws']) for c in [utils.WSTART]+list(w)+[utils.WEND]]

    def pprint(self, sent, *margs):
        args = [sent['cseq'],sent['tseq']]; args.extend(margs)
        mlen = max(len(e) for e in chain(*args))
        wgroup = [[e[0] for e in g] for k, g in groupby(enumerate(sent['wiseq']), lambda x: x[1])]
        for cil in wgroup:
            for a in args:
                print ' '.join(map(('{:^%d}'%mlen).format, [a[ci] for ci in cil]))

def get_ts_io(wiseq, tseq):
    tgroup = [[e[0] for e in g] for k, g in groupby(enumerate(wiseq),lambda x: x[1]) if k >= 0]
    tseqgrp = [[tseq[ti] for ti in ts] for ts in tgroup]
    return [Counter(tseq1).most_common(1)[0][0].upper() for tseq1 in tseqgrp]

def get_ts_bio(wiseq, tseq):
# def get_ts():
    """
    cseq = ['a','b','c',' ','d','e']
    wiseq = [0,0,0,-1,1,1]
    tseq = ['i-per', 'i-per', 'i-per', 'o', 'i-per', 'i-per']
    """
    # print tabulate([cseq,wiseq,tseq])
    windxs = [group.next()[1] for k, group in groupby(((wi,i) for i, wi in enumerate(wiseq) if wi > -1), lambda x:x[0])]
    ts = []
    for i in windxs:
        if tseq[i] == 'o':
            ts.append('O')
        else:
            ttype = tseq[i].split('-')[1]
            if i == 0:
                ts.append('B-{}'.format(ttype.upper()))
            else:
                if tseq[i-1] == tseq[i]:
                    ts.append('I-{}'.format(ttype.upper()))
                else:
                    ts.append('B-{}'.format(ttype.upper()))
    return ts

def is_consec(sent):
    return any(t1.startswith('I-') and t2.startswith('B-') and t1.split('-')[1] == t2.split('-')[1]
            for t1,t2 in zip(sent['ts'],sent['ts'][1:]))

if __name__ == '__main__':
    trn,dev,tst = utils.get_sents('eng')
    rep = Repstd()
    print Counter(c for sent in trn for c in rep.get_cseq(sent))

