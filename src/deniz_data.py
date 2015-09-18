import logging
from featchar import *
import featchar
from utils import get_sents, get_sent_indx, sample_sents
from exper_lasagne import Feat
import numpy as np

if __name__ == '__main__':
    logger = logging.getLogger()
    logger.setLevel(logging.INFO)
    shandler = logging.StreamHandler()
    shandler.setLevel(logging.INFO)
    logger.addHandler(shandler);

    trn, dev, tst = get_sents()

    ctag2wtag_func = get_ts2 
    wtag2ctag_func = get_tseq2

    for d in (trn,dev,tst):
        for sent in d:
            sent.update({
                'cseq': get_cseq(sent), 
                'wiseq': get_wiseq(sent), 
                'tseq': wtag2ctag_func(sent)})


    trn = sorted(trn, key=lambda sent: len(sent['cseq']))
    dev = sorted(dev, key=lambda sent: len(sent['cseq']))

    ntrnsent, ndevsent, ntstsent = list(map(len, (trn,dev,tst)))
    logger.info('# of sents trn, dev, tst: {} {} {}'.format(ntrnsent, ndevsent, ntstsent))


    MAX_LENGTH = max(len(sent['cseq']) for sent in chain(trn,dev))
    MIN_LENGTH = min(len(sent['cseq']) for sent in chain(trn,dev))
    logger.info('maxlen: {} minlen: {}'.format(MAX_LENGTH, MIN_LENGTH))

    feat = Feat(featchar.get_cfeatures_basic_seg)
    feat.fit(trn)

    trndat = [feat.transform(sent) for sent in trn]
    devdat = [feat.transform(sent) for sent in dev]
    np.savez('/ai/home/okuru13/tmp/for_deniz.npz',trn=trndat,dev=devdat)
