import logging
import numpy as np

import rep
import utils

class Dset(object):

    def __init__(self, lang='eng', level='char', tagging='bio', breaktrn=False, captrn=500, sample=0, charrep='std', sort=True, **kwargs):

        trn, dev, tst = utils.get_sents(lang)

        repclass = getattr(rep, 'Rep'+charrep)
        repobj = repclass()

        for d in (trn,dev,tst):
            for sent in d:
                sent.update({
                    'cseq': repobj.get_cseq(sent), 
                    'wiseq': repobj.get_wiseq(sent), 
                    'tseq': repobj.get_tseq(sent)})
                sent['x'] = sent['cseq'] if level == 'char' else sent['ws']
                sent['y'] = sent['tseq'] if level == 'char' else sent['ts']


        if captrn:
            trn = filter(lambda sent: len(' '.join(sent['x']))<captrn, trn)

        if sample>0:
            trn_size = sample*1000
            trn = utils.sample_sents(trn,trn_size)

        if sort:
            trn = sorted(trn, key=lambda sent: len(sent['x']))
            dev = sorted(dev, key=lambda sent: len(sent['x']))
            tst = sorted(tst, key=lambda sent: len(sent['x']))

        ntrnsent, ndevsent, ntstsent = list(map(len, (trn,dev,tst)))
        logging.info('# of sents trn, dev, tst: {} {} {}'.format(ntrnsent, ndevsent, ntstsent))

        for dset, dname in zip((trn,dev,tst),('trn','dev','tst')):
            slens = [len(sent['x']) for sent in dset]
            MAX_LENGTH, MIN_LENGTH, AVG_LENGTH, STD_LENGTH = max(slens), min(slens), np.mean(slens), np.std(slens)
            logging.info('input: {}\tmaxlen: {} minlen: {} avglen: {:.2f} stdlen: {:.2f}'.format(dname,MAX_LENGTH, MIN_LENGTH, AVG_LENGTH, STD_LENGTH))
        self.trn, self.dev, self.tst = trn, dev, tst

if __name__ == '__main__':
    utils.logger()
    dset = Dset(level='word')
