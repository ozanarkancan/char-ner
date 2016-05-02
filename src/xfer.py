import numpy as np
import argparse
import random
import lasagne
import logging

from lazrnn import RDNN as RNN
import rep, featchar, exper
from exper import Batcher, Reporter, Validator
from utils import get_sents
import decoder

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('model_file')
    pargs = parser.parse_args()

    dat = np.load(pargs.model_file)
    args = dat['argsd'].tolist()
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

    vdecoder = decoder.ViterbiDecoder(trn, feat)

    batcher = Batcher(args['n_batch'], feat) # batch size 1
    devdat = batcher.get_batches(dev) 
    tstdat = batcher.get_batches(tst) 

    rdnn = RNN(feat.NC, feat.NF, args)
    cost, dev_predictions = rdnn.predict(devdat)
    cost, tst_predictions = rdnn.predict(tstdat)

    predictions = {}
    predictions['dev'] = dev_predictions
    predictions['tst'] = tst_predictions

    dset = {}
    dset['dev'] = dev
    dset['tst'] = tst
    repobj = repobj

    reporter = exper.Reporter(feat, rep.get_ts_bio)

    print rdnn.l_soft_out.get_params()
    print rdnn.blayers[0][0].get_params()
    params = lasagne.layers.get_all_param_values(rdnn.layers[-1])
    print map(np.shape, params)
    lasagne.layers.set_all_param_values(rdnn.layers[-1], rnn_param_values[:len(params)])

    logger = logging.getLogger()
    logger.setLevel(logging.INFO)
    shandler = logging.StreamHandler()
    shandler.setLevel(logging.INFO)
    logger.addHandler(shandler)

    validator = Validator(trn, dev, tst, batcher, reporter)
    validator.validate(rdnn, args, vdecoder)

    dset = dset['dev']
    pred = [p for b in predictions[dsetname] for p in b]
    pred2 = [vdecoder.decode(s,p) for s,p in zip(dset,pred)]
    cerr, werr, wacc, pre, recall, f1, conll_print, char_conmat_str, word_conmat_str = reporter.report(dset, pred2) 
    print pre, recall, f1

if __name__ == '__main__':
    main()
