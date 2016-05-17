import string, numpy as np, logging
from itertools import chain

from sklearn.feature_extraction import DictVectorizer
from sklearn.preprocessing import LabelEncoder

import utils

class Feat(object):

    def __init__(self, featstr):
        self.dvec = DictVectorizer(dtype=np.float32, sparse=False)
        self.tseqenc = LabelEncoder()
        self.tsenc = LabelEncoder()
        self.featfuncs = featstr.split('_')
        self.randemb = {}

    def getcfeat(self, ci, sent):
        d = {}
        for f in self.featfuncs:
            d.update(getattr(self, 'feat_'+f)(ci,sent))
        return d

    def fit(self, dset, xdsets=[]):
        trn, dev, tst = dset.trn, dset.dev, dset.tst
        xtrn = (s for d in xdsets for s in d.trn)
        self.dvec.fit(self.getcfeat(ci, sent)  for sent in chain(trn,xtrn) for ci,c in enumerate(sent['cseq']))
        self.tseqenc.fit([t for sent in trn for t in sent['tseq']])
        self.tsenc.fit([t for sent in chain(trn,dev,tst) for t in sent['ts']])
        self.feature_names = self.dvec.get_feature_names()
        self.ctag_classes = self.tseqenc.classes_
        self.wtag_classes = self.tsenc.classes_
        logging.info(self.feature_names)
        logging.debug(' '.join([fn for fn in self.feature_names]))
        logging.info(self.ctag_classes)
        logging.info(self.wtag_classes)
        self.NF = len(self.feature_names)
        self.NC = len(self.ctag_classes)
        logging.info('NF: {} NC: {}'.format(self.NF, self.NC))

    def transform(self, sent):
        Xsent = self.dvec.transform([self.getcfeat(ci, sent) for ci,c in enumerate(sent['cseq'])]) # nchar x nf
        ysent = self.one_hot(self.tseqenc.transform([t for t in sent['tseq']]), self.NC) # nchar x nc
        return Xsent, ysent

    def one_hot(self, labels, n_classes):
        one_hot = np.zeros((labels.shape[0], n_classes)).astype(bool)
        one_hot[range(labels.shape[0]), labels] = True
        return one_hot

    def feat_basic(self, ci, sent):
        return {'c': sent['cseq'][ci]}

    def feat_dgen(self, ci, sent):
        c = sent['cseq'][ci]
        if c in string.digits:
            return {'c': 'digit'}
        else:
            return {'c': c}

    def feat_gen(self, ci, sent):
        c = sent['cseq'][ci]
        if c in string.ascii_letters:
            return {'c': c}
        elif c in (utils.WSTART, utils.WEND):
            return {'c': c}
        elif c == ' ':
            return {'c': 'space'}
        elif c in string.digits:
            return {'c': 'digit'}
        elif c in string.punctuation:
            return {'c': 'punc'}
        else:
            return {'c': 'other'}

    def feat_rand(self, ci, sent):
        c = sent['cseq'][ci]
        if not (c in self.randemb):
            self.randemb[c] = np.random.rand(10)
        emb = self.randemb[c]
        return dict(('e'+str(i),v) for i,v in enumerate(emb))
        
    def feat_simple(ci, sent):
        d = {}
        c = sent['cseq'][ci]
        if c in 'abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ': # TODO does not work for all lang
            d['c'] = c.lower()
        else:
            d['c'] = 'not_letter'
        d['isupper'] = c.isupper()
        d['isdigit'] = c.isdigit()
        d['ispunc'] = c in string.punctuation

        return d

    def feat_seg(self, ci, sent):
        d = {}
        # wstart
        if ci==0:
            d['wstart'] = 1
        elif ci>0:
            d['wstart'] = sent['wiseq'][ci-1] != sent['wiseq'][ci]

        # wend
        if ci==(len(sent['cseq'])-1):
            d['wend'] = 1
        elif ci<(len(sent['cseq'])-1):
            d['wend'] = sent['wiseq'][ci+1] != sent['wiseq'][ci]
        return d

    def feat_ptag(self, ci, sent):
        c = sent['cseq'][ci]
        wi = sent['wiseq'][ci]
        return {'pt' : 'space_pt'} if c == ' ' else {'pt':sent['pts'][wi]}

    def feat_ctag(self, ci, sent):
        c = sent['cseq'][ci]
        wi = sent['wiseq'][ci]
        return {'ct' : 'space_ct'} if c == ' ' else {'ct':sent['cts'][wi]}

    def feat_cap(self, ci, sent):
        return {'is_capital' : sent['cseq'][ci].isupper()}
        

if __name__ == '__main__':
    import rep
    trn, dev, tst = utils.get_sents('eng')
    
    repstd = rep.Repstd()
    for sent in trn:
        sent['cseq'] = repstd.get_cseq(sent)
        sent['wiseq'] = repstd.get_wiseq(sent)
        sent['tseq'] = repstd.get_tseq(sent)

    feat = Feat('rand')
    feat.fit(trn)

    sents = utils.sample_sents(trn, 5, 6, 8)

    for sent in sents:
        Xsent, ysent = feat.transform(sent)
        print 'X:'
        print Xsent
        print 'y:'
        print ysent
        

