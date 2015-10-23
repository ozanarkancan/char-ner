import numpy as np
import argparse
import random

from lazrnn import RDNN as RNN
import rep, featchar
from exper import Batcher, Reporter
from utils import get_sents

class Sampler(object):

    def __init__(self, model_file):
        dat = np.load(model_file)
        args = dat['args'].tolist()
        rnn_param_values = dat['rnn_param_values'].tolist()


        trn, dev, tst = get_sents(args['lang'])

        repclass = getattr(rep, 'Rep'+args['rep'])
        repobj = repclass()
        for d in (trn,dev,tst):
            for sent in d:
                sent.update({
                    'cseq': repobj.get_cseq(sent), 
                    'wiseq': repobj.get_wiseq(sent), 
                    'tseq': repobj.get_tseq(sent)})

        trn = sorted(trn, key=lambda sent: len(sent['cseq']))
        dev = sorted(dev, key=lambda sent: len(sent['cseq']))
        tst = sorted(tst, key=lambda sent: len(sent['cseq']))

        self.feat = featchar.Feat(args['feat'])
        self.feat.fit(trn,dev,tst)

        batcher = Batcher(1, self.feat) # batch size 1
        devdat = batcher.get_batches(dev) 
        tstdat = batcher.get_batches(tst) 

        rdnn = RNN(self.feat.NC, self.feat.NF, args)
        rdnn.set_param_values(rnn_param_values)
        cost, dev_predictions = rdnn.predict(devdat)
        cost, tst_predictions = rdnn.predict(tstdat)

        self.predictions = {}
        self.predictions['dev'] = dev_predictions
        self.predictions['tst'] = tst_predictions

        self.dset = {}
        self.dset['dev'] = dev
        self.dset['tst'] = tst
        self.repobj = repobj

        # reporter = Reporter(feat, rep.get_ts)
        # print reporter.report(dev, predictions)
    def sample(self, dsetname, sample_size=10, correct_perc=0.9):
        sents = []
        for sent, ipred in zip(self.dset[dsetname],self.predictions[dsetname]):
            tseq_pred = self.feat.tseqenc.inverse_transform(ipred)
            num_correct = sum(1 for t1,t2 in zip(sent['tseq'], tseq_pred) if t1==t2)
            num_t = len(sent['tseq'])
            correct_perc = num_correct/float(num_t)
            if correct_perc < 0.9:
                sents.append((sent, tseq_pred))

        for sent, tseq_pred in random.sample(sents,sample_size):
            self.repobj.pprint(sent, tseq_pred)
            print 

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('model_file')
    pargs = parser.parse_args()
    sampler = Sampler(pargs.model_file)
    sampler.sample('dev')
