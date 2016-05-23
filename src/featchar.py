import numpy as np, logging, string
from itertools import chain
from utils import DROPSYM

from sklearn.feature_extraction import DictVectorizer
from sklearn.preprocessing import LabelEncoder


class Feat(object):

    def __init__(self, featstr='basic', dtype=np.float32):
        self.dvec = DictVectorizer(dtype=dtype, sparse=False)
        self.yenc = LabelEncoder()
        self.featfuncs = featstr.split('_')
        self.randemb = {}

    def getcfeat(self, ci, sent):
        d = {}
        for f in self.featfuncs:
            d.update(getattr(self, 'feat_'+f)(ci,sent))
        return d

    def fit(self, dset, xdsets=[]):
        trn = dset.trn
        xtrn = (s for d in xdsets for s in d.trn)
        self.dvec.fit(self.getcfeat(ci, sent)  for sent in chain(trn,xtrn) for ci,c in enumerate(sent['x']))
        self.yenc.fit([t for sent in trn for t in sent['y']])
        # self.tsenc.fit([t for sent in chain(trn,dev,tst) for t in sent['ts']])
        self.feature_names = self.dvec.get_feature_names()
        self.tag_classes = self.yenc.classes_
        logging.info(self.feature_names)
        logging.debug(' '.join([fn for fn in self.feature_names]))
        logging.info(self.tag_classes)
        self.NF = len(self.feature_names)
        self.NC = len(self.tag_classes)
        logging.info('NF: {} NC: {}'.format(self.NF, self.NC))

    def transform(self, sent):
        Xsent = self.dvec.transform([self.getcfeat(ci, sent) for ci,c in enumerate(sent['x'])]) # nchar x nf
        ysent = self.one_hot(self.yenc.transform([t for t in sent['y']]), self.NC) # nchar x nc
        return Xsent, ysent

    def one_hot(self, labels, n_classes):
        one_hot = np.zeros((labels.shape[0], n_classes)).astype(bool)
        one_hot[range(labels.shape[0]), labels] = True
        return one_hot

    def feat_basic(self, ci, sent):
        return {'c': sent['x'][ci]}

    def feat_cdrop(self, ci, sent):
        return {'c': sent['x'][ci] if np.random.rand() > .2 else DROPSYM}

    def feat_dgen(self, ci, sent):
        c = sent['x'][ci]
        if c in string.digits:
            return {'c': 'digit'}
        else:
            return {'c': c}

    def feat_gen(self, ci, sent):
        c = sent['x'][ci]
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
        c = sent['x'][ci]
        if not (c in self.randemb):
            self.randemb[c] = np.random.rand(10)
        emb = self.randemb[c]
        return dict(('e'+str(i),v) for i,v in enumerate(emb))

    def feat_cap(self, ci, sent):
        return {'is_capital' : sent['x'][ci].isupper()}
        

if __name__ == '__main__':
    from dataset import Dset
    import utils
    utils.logger()
    dset = Dset(level='word')
    feat = Feat()
    feat.fit(dset)
    x,y = feat.transform(dset.trn[0])
    print x
    print y
