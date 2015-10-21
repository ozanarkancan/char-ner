import numpy as np
import argparse

from lazrnn import RDNN as RNN
import rep, featchar
from exper import Batcher, Reporter
from utils import get_sents

def main(pargs):
    dat = np.load(pargs.model_file)
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

    feat = featchar.Feat(args['feat'])
    feat.fit(trn,dev,tst)

    batcher = Batcher(1, feat) # batch size 1
    devdat = batcher.get_batches(dev) 

    rdnn = RNN(feat.NC, feat.NF, args)
    rdnn.set_param_values(rnn_param_values)
    cost, predictions = rdnn.predict(devdat)
    print cost
    print predictions[0]
    reporter = Reporter(feat, rep.get_ts)
    print reporter.report(dev, predictions)
    for sent, ipred in zip(dev,predictions):
        pass

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('model_file')
    pargs = parser.parse_args()
    main(pargs)
