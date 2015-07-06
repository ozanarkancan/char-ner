import copy
from itertools import *
import random, numpy as np
from sklearn.feature_extraction import DictVectorizer
from sklearn.preprocessing import LabelEncoder

def read_sents(file):
    a,b,c,d = [],[],[],[]
    sentences = []
    with open(file) as src:
        for l in src:
            if len(l.strip()):
                w, pt, ct, t = l.strip().split('\t')
                a.append(w);b.append(pt);
                c.append(ct);d.append(t);
            else: # emtpy line
                if len(a):
                    sentences.append({'ws':a,'ts':d,'tsg':copy.deepcopy(d),\
                            'pts':b,'cts':c})
                a,b,c,d = [],[],[],[]
    return sentences

def get_sents():
    return read_sents('data/train.bilou'), read_sents('data/testa.bilou'), read_sents('data/testb.bilou')

def get_cfeatures(wi, ci, sent):
    return {'c':sent['cseq'][ci]}

def sent2mat(sents, dvec, le):
    XL, yL = [],[]
    for sent in sents:
        # Xsent = dvec.transform(get_features(i,sent,c) for i in range(len(sent['ws'])))
        Xsent = dvec.transform(get_features(i, sent, c) for (i,w) in enumerate(sent['ws']) for c in w)
        ysent = le.transform([sent['ts'][i] for (i,w) in enumerate(sent['ws']) for c in w])
        XL.append(Xsent)
        yL.append(ysent)
    return XL, yL

def dset2mat(sents, dvec, le):
    XL, yL = [],[]
    for sent in sents:
        # Xsent = dvec.transform(get_features(i,sent,c) for i in range(len(sent['ws'])))
        Xsent = dvec.transform(get_features(i, sent, c) for (i,w) in enumerate(sent['ws']) for c in w)
        ysent = le.transform([sent['ts'][i] for (i,w) in enumerate(sent['ws']) for c in w])
        XL.append(Xsent)
        yL.append(ysent)
    return XL, yL
    pass

def extend_sent(sent):
    cseq, tseq, wiseq = [], [], []
    wi = 0
    for w, t in zip(sent['ws'], sent['ts']):
        if t == 'O': tp, ttype = 'O', 'O'
        else: tp, ttype = t.split('-')
        ttype = ttype[0]

        cseq.extend([c for c in w])
        wiseq.extend([wi for c in w])
        cseq.append(' ') # for space
        wiseq.append(wi)
        wi+=1
        if tp == 'U':
            tseq.append('b-'+ttype)
            for c in w[1:-1]:
                tseq.append('i-'+ttype)
            tseq.append('l-'+ttype)
            tseq.append('o') # for space
        elif tp == 'B':
            tseq.append('b-'+ttype)
            for c in w[1:]:
                tseq.append('i-'+ttype)
            tseq.append('i-'+ttype) # for space
        elif tp == 'L':
            for c in w[:-1]:
                tseq.append('i-'+ttype)
            tseq.append('l-'+ttype)
            tseq.append('o') # for space
        elif tp == 'I':
            for c in w:
                tseq.append('i-'+ttype)
            tseq.append('i-'+ttype) # for space
        else: # O
            for c in w:
                tseq.append('o')
            tseq.append('o') # for space
    sent['cseq'] = cseq[:-1]
    sent['tseq'] = tseq[:-1]
    sent['wiseq'] = wiseq

def get_sent_indx(dset):
    start = 0
    indexes = []
    for sent in dset:
        indexes.append((start,start+len(sent['cseq'])))
        start += len(sent['cseq'])
    return indexes

if __name__ == '__main__':
    trn, dev, tst = get_sents()
    trn = random.sample(trn,1000)
    dev = random.sample(trn,100)

    for d in (trn,dev,tst):
        for sent in d:
            extend_sent(sent)

    dvec = DictVectorizer(dtype=np.float32, sparse=False)
    lblenc = LabelEncoder()
    dvec.fit(get_cfeatures(wi, ci, sent)  for sent in trn for c,wi,ci in zip(sent['cseq'],sent['wiseq'],count(0)))
    lblenc.fit([t for sent in trn for t in sent['tseq']])
    print dvec.get_feature_names()
    print lblenc.classes_

    nf = len(dvec.get_feature_names())
    nc = len(lblenc.classes_)
    print '# of sents: ', map(len, (trn,dev,tst))
    print '# of feats: ', nf 
    print '# of lbls: ', nc

    Xtrn = dvec.transform(get_cfeatures(wi, ci, sent)  for sent in trn for c,wi,ci in zip(sent['cseq'],sent['wiseq'],count(0)))
    Xdev = dvec.transform(get_cfeatures(wi, ci, sent)  for sent in dev for c,wi,ci in zip(sent['cseq'],sent['wiseq'],count(0)))
    Xtst = dvec.transform(get_cfeatures(wi, ci, sent)  for sent in tst for c,wi,ci in zip(sent['cseq'],sent['wiseq'],count(0)))

    print Xtrn.shape, Xdev.shape

    ytrn = lblenc.transform([t for sent in trn for t in sent['tseq']])
    ydev = lblenc.transform([t for sent in dev for t in sent['tseq']])
    ytst = lblenc.transform([t for sent in tst for t in sent['tseq']])

    print ytrn.shape, ydev.shape

    trnIndx = get_sent_indx(trn)
    devIndx = get_sent_indx(dev)
    tstIndx = get_sent_indx(tst)

    print len(trnIndx), len(devIndx)
