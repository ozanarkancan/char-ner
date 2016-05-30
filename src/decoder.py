import numpy as np
import logging
from itertools import imap, groupby
from tabulate import tabulate

class WDecoder(object):

    def __init__(self, trn, feat):
        self.feat = feat
        # self.states = dd(set)
        self.transition_tensor = np.zeros((1, feat.NC, feat.NC)) + np.log(np.finfo(float).eps)
        for sent in trn:
            tseq = feat.yenc.transform([t for t in sent['y']])
            for t,tprev in zip(tseq[1:],tseq):
                self.transition_tensor[0,tprev,t] = 1
        logging.info('transition tensor:')
        logging.info(tabulate(self.transition_tensor[0]))

    def decode(self, sent, logprobs, debug=False):
        from viterbi import viterbi_log_multi

        tstates = [0] * len(sent['x'])
        emissions = map(lambda x:x[0], enumerate(sent['x']))
        tseq_ints = viterbi_log_multi(logprobs.T, self.transition_tensor, emissions, tstates)

        """
        if not self.sanity_check(sent, tseq_ints):
            logging.critical(' '.join(sent['ws']))
            logging.critical(' '.join(sent['ts']))
            logging.critical('gold tseq: {}'.format(sent['tseq']))
            logging.critical('decoded tseq: {}'.format(tseq_ints))
            logging.critical(logprobs)
            raise Exception('decoder sanity check failed')
        """

        return tseq_ints

class ViterbiDecoder(object):

    def __init__(self, trn, feat):
        from collections import defaultdict as dd

        self.feat = feat

        self.states = dd(set)
        for sent in trn:
            wistates =  map(lambda x:int(x<0), sent['wiseq'])
            tseq = feat.yenc.transform([t for t in sent['tseq']])
            # for (tprev,t), (wstate_prev, wstate) in zip(zip(tseq[1:],tseq), zip(wistates[1:], wistates)):
            for (t,tprev), (wstate, wstate_prev) in zip(zip(tseq[1:],tseq), zip(wistates[1:], wistates)):
                indx = int(''.join(map(str,(wstate_prev,wstate))), 2)
                # states[(wstate_prev,wstate)].add((tprev,t))
                self.states[indx].add((tprev,t))
        self.transition_tensor = np.zeros((len(self.states.keys()), feat.NC, feat.NC)) + np.log(np.finfo(float).eps)
        for i, valids in self.states.iteritems():
            for k,l in valids:
                self.transition_tensor[i,k,l] = 1
        for i in range(self.transition_tensor.shape[0]):
            logging.debug(self.transition_tensor[i])

    def decode(self, sent, logprobs, debug=False):
        from viterbi import viterbi_log_multi
        to_indx = lambda wstate_prev, wstate: int(''.join(map(str,(wstate_prev,wstate))), 2)

        wistates =  map(lambda x:int(x<0), sent['wiseq'])
        tstates = list(imap(to_indx, [0]+wistates, wistates))
        emissions = map(lambda x:x[0], enumerate(sent['cseq']))
        tseq_ints = viterbi_log_multi(logprobs.T, self.transition_tensor, emissions, tstates)

        if not self.sanity_check(sent, tseq_ints):
            logging.critical(' '.join(sent['ws']))
            logging.critical(' '.join(sent['ts']))
            logging.critical('gold tseq: {}'.format(sent['tseq']))
            logging.critical('decoded tseq: {}'.format(tseq_ints))
            logging.critical(logprobs)
            raise Exception('decoder sanity check failed')

        return tseq_ints

    def sanity_check(self, sent, tseq_ints):
        tseq = self.feat.yenc.inverse_transform(tseq_ints)
        if any(len(set(group)) != 1 for k, group in groupby(filter(lambda x: x[0]>-1, zip(sent['wiseq'], tseq)))):
            return False
        sp_indxs = [i for i,wi in enumerate(sent['wiseq']) if wi == -1]
        if any(not (tseq[i-1] == tseq[i] == tseq[i+1]) for i in sp_indxs if tseq[i] != 'o'):
            return False
        return True

    def pprint(self):
        for i in xrange(self.transition_tensor.shape[0]):
            for r in (self.transition_tensor[i] > 0).astype(np.int):
                print ' '.join(map(str, r.tolist()))
            print 
        clist = self.feat.yenc.classes_.tolist()
        print clist 
        """
        for i in xrange(self.transition_tensor.shape[0]):
            table = [[''] + clist]
            for r, tag in zip((self.transition_tensor[i] > 0).astype(np.int), clist):
                table.append([tag] + r.tolist())
            print tabulate(table)
        """

def randlogprob(sent, nc):
    sent_len = len(sent['cseq'])
    randvals = np.random.rand(sent_len, nc)
    randlogprobs = np.log(randvals / np.sum(randvals,axis=0))
    return randlogprobs

class MaxDecoder(object):

    def __init__(self, trn, feat):
        pass

    def decode(self, sent, logprobs, debug=False):
        return np.argmax(logprobs, axis=-1).flatten()

    def sanity_check(self, sent, tseq_ints):
        return True

def main():
    from utils import get_sents
    import featchar, rep

    trn, dev, tst = get_sents('toy')

    r = rep.Repstd()

    for sent in trn:
        sent.update({
            'cseq': r.get_cseq(sent), 
            'wiseq': r.get_wiseq(sent), 
            'tseq': r.get_tseq(sent)})
    r.pprint(trn[0])
    print
    r.pprint(trn[1])

    print rep.get_ts_bio(trn[0]['wiseq'], trn[0]['tseq'])

    feat = featchar.Feat('basic')
    feat.fit(trn,dev,tst)

    vdecoder = ViterbiDecoder(trn, feat)
    vdecoder.pprint()
    sent = trn[0]
    vdecoder.decode(sent, randlogprob(sent, feat.NC), debug=True)

    """
    sent = random.choice(filter(lambda x: len(x['cseq']) < 10 and len(x['cseq']) > 6, trn))
    randvals = np.random.rand(len(sent['cseq']),feat.NC)
    randlogprobs = np.log(randvals / np.sum(randvals,axis=0))
    tseq = vdecoder.decode(sent, randlogprobs, debug=True)
    print tseq
    """

if __name__ == '__main__':
    main()
