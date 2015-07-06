import copy, itertools as ite
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

def get_features(i, sent, c):
    return {'c':c}

def sent2mat(sents, dvec, le):
    XL, yL = [],[]
    for sent in sents:
        # Xsent = dvec.transform(get_features(i,sent,c) for i in range(len(sent['ws'])))
        Xsent = dvec.transform(get_features(i, sent, c) for (i,w) in enumerate(sent['ws']) for c in w)
        ysent = le.transform([sent['ts'][i] for (i,w) in enumerate(sent['ws']) for c in w])
        XL.append(Xsent)
        yL.append(ysent)
    return XL, yL

if __name__ == '__main__':
    sents = read_sents('data/train.bilou')
    dev_sents = read_sents('data/testa.bilou')
    tst_sents = read_sents('data/testb.bilou')

    dvec = DictVectorizer(dtype=np.float32, sparse=False)
    lblenc = LabelEncoder()
    dvec.fit(get_features(i, sent, c)  for sent in sents for (i,w) in enumerate(sent['ws']) for c in w)
    lblenc.fit([t for sent in sents for t in sent['ts']])

    # sents = random.sample(sents,3)
    # dev_sents = random.sample(dev_sents,2)

    nf = len(dvec.get_feature_names())
    nc = len(lblenc.classes_)
    ns = len(sents)
    print '# of sents: ', ns
    print '# of feats: ', nf 
    print '# of lbls: ', nc

    # sent2mat(sents, dvec, lblenc)
    from rnn import MetaRNN
    model = MetaRNN(n_in=nf, n_hidden=50, n_out=nc,
                    learning_rate=0.001, learning_rate_decay=0.999,
                    n_epochs=1, activation='tanh',
                    output_type='softmax', use_symbolic_softmax=False)

    for e in xrange(100): # epochs
        for sent in sents:
            Xsent = dvec.transform(get_features(i, sent, c) for (i,w) in enumerate(sent['ws']) for c in w)
            ysent = lblenc.transform([sent['ts'][i] for (i,w) in enumerate(sent['ws']) for c in w])
            Xsent = np.reshape(Xsent,(1,)+Xsent.shape)
            ysent = np.reshape(ysent, (1,)+ysent.shape)
            print Xsent.shape, ysent.shape
            model.fit(Xsent, ysent, validation_frequency=1000)
        ncorrect, ntotal = 0, 0
        for sent in dev_sents:
            Xsent = dvec.transform(get_features(i, sent, c) for (i,w) in enumerate(sent['ws']) for c in w)
            ysent = lblenc.transform([sent['ts'][i] for (i,w) in enumerate(sent['ws']) for c in w])
            ypred = model.predict(Xsent)
            res = ypred == ysent
            ncorrect = np.sum(res)
            ntotal = res.size
        print 'acc: ', ncorrect/float(ntotal)

    """
    np.savez('topsecret.npz',
            trn = sent2mat(sents, dvec, lblenc),
            dev = sent2mat(dev_sents, dvec, lblenc),
            tst = sent2mat(tst_sents, dvec, lblenc))
    """

    """
    for w,t in zip(sent['ws'],sent['ts']):
        for a,b in zip([c for c in w],ite.repeat(t)):
            print a,b
        print 
    """
