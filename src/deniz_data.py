import logging
import numpy as np
from itertools import *
from scipy.sparse import coo_matrix

import rep
import featchar
import encoding
from utils import get_sents, sample_sents

def write_sparse(fname, mat):
    with open(fname,'w') as out:
        for i,j,v in zip(mat.row, mat.col, mat.data):
            out.write('{} {} {}\n'.format(i,j,int(v)))

if __name__ == '__main__':
    logger = logging.getLogger()
    logger.setLevel(logging.INFO)
    shandler = logging.StreamHandler()
    shandler.setLevel(logging.INFO)
    logger.addHandler(shandler);

    trn, dev, tst = get_sents('eng')

    """
    trn = sample_sents(trn,5,2,4)
    dev = sample_sents(dev,5,2,4)
    tst = sample_sents(tst,5,2,4)
    """

    repobj = rep.Repstd()
    for d in (trn,dev,tst):
        for sent in d:
            sent['ts'] = encoding.any2io(sent['ts'])
            sent.update({
                'cseq': repobj.get_cseq(sent), 
                'wiseq': repobj.get_wiseq(sent), 
                'tseq': repobj.get_tseq(sent)})


    trn = sorted(trn, key=lambda sent: len(sent['cseq']))
    dev = sorted(dev, key=lambda sent: len(sent['cseq']))
    tst = sorted(tst, key=lambda sent: len(sent['cseq']))

    ntrnsent, ndevsent, ntstsent = list(map(len, (trn,dev,tst)))
    logger.info('# of sents trn, dev, tst: {} {} {}'.format(ntrnsent, ndevsent, ntstsent))


    MAX_LENGTH = max(len(sent['cseq']) for sent in chain(trn,dev,tst))
    MIN_LENGTH = min(len(sent['cseq']) for sent in chain(trn,dev,tst))
    logger.info('maxlen: {} minlen: {}'.format(MAX_LENGTH, MIN_LENGTH))

    feat = featchar.Feat('basic')
    feat.fit(trn,dev,tst)
    sent = dev[999]
    Xsent, Ysent = feat.transform(sent)
    print sent['cseq']
    print [c+1 for r,c in zip(*np.where(Xsent))]

    """
    # for dset,dname in zip((trn,),('trn',)):
    for dset,dname in zip((trn,dev,tst),('trn','dev','tst')):
        Xsents, Ysents = [],[]
        for sent in dset:
            Xsent,Ysent = feat.transform(sent)
            # print 'Xsent:{} Ysent:{}'.format(Xsent.shape, Ysent.shape)
            Xpad = np.zeros((MAX_LENGTH-Xsent.shape[0],Xsent.shape[1]),dtype=np.bool)
            Ypad = np.zeros((MAX_LENGTH-Ysent.shape[0],Ysent.shape[1]),dtype=np.bool)
            Xsent = np.vstack((Xsent,Xpad))
            Ysent = np.vstack((Ysent,Ypad))
            Xsents.append(Xsent)
            Ysents.append(Ysent)
        Xsents = coo_matrix(np.vstack(Xsents))
        Ysents = coo_matrix(np.vstack(Ysents))
        write_sparse('/ai/home/okuru13/research/char-ner.jl/X{}.txt'.format(dname),Xsents)
        write_sparse('/ai/home/okuru13/research/char-ner.jl/Y{}.txt'.format(dname),Ysents)
    """
