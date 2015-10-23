from itertools import *
from collections import Counter

from encoding import iob2io
import utils

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
        """ io """
        tseq = []
        for w, t, tnext in zip(sent['ws'],sent['ts'], chain(sent['ts'][1:],[None])):
            tp, sep, ttype = (t, '', 'O') if t == 'O' else (t.split('-')[0], '-', t.split('-')[1])
            tseq.extend(ttype.lower() for c in w)

            # handle space
            if t == tnext:
                tseq.append(ttype.lower())
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
        """ assumes sent['ts'] is in iob scheme """
        ts = iob2io(sent['ts'])
        return [ts[wi].lower() for wi,w in enumerate(sent['ws']) for c in w]

class Repspec(object):

    def get_cseq(self, sent):
        return [c for w in sent['ws'] for c in [utils.WSTART]+list(w)+[utils.WEND]]

    def get_wiseq(self, sent):
        return [wi for wi,w in enumerate(sent['ws']) for c in [utils.WSTART]+list(w)+[utils.WEND]]

    def get_tseq(self, sent):
        """ assumes sent['ts'] is in iob scheme """
        ts = iob2io(sent['ts'])
        return [ts[wi].lower() for wi,w in enumerate(sent['ws']) for c in [utils.WSTART]+list(w)+[utils.WEND]]

def get_ts(wiseq, tseq):
    tgroup = [[e[0] for e in g] for k, g in groupby(enumerate(wiseq),lambda x: x[1]) if k >= 0]
    tseqgrp = [[tseq[ti] for ti in ts] for ts in tgroup]
    return [Counter(tseq).most_common(1)[0][0].upper() for tseq in tseqgrp]


def sent_word_indx(sent):
    import numpy as np
    space_indx = [a for a,b in enumerate(sent['wiseq']) if b==-1]
    indx = [0] + space_indx + [len(sent['wiseq'])]
    arr = np.array([[a,b] for a,b in zip(indx,indx[1:])])
    arr[1:,0] += 1
    return arr


if __name__ == '__main__':
    from utils import get_sents, sample_sents
    from encoding import io2iob
    trn, dev, tst = get_sents('eng')

    trn = sample_sents(trn, 3, 5,6)
    r = Repstd()

    for sent in trn:
        sent.update({
            'cseq': r.get_cseq(sent), 
            'wiseq': r.get_wiseq(sent), 
            'tseq': r.get_tseq(sent)})
        r.pprint(sent)
        print 
