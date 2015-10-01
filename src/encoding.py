import random
from itertools import *
from utils import get_sents, sample_sents
from score import conlleval


def bio2bilou(ts):
    from bio2bilou import c4
    return c4(' '.join(ts))


def bilou2io(ts):
    return [e.split('-')[1] if len(e.split('-')) == 2 else e for e in ts]

def bio2io(ts):
    return bilou2io(ts)

def io2bio(ts):
    io_phrases = [(key,list(subite)) for key, subite in groupby(ts)]
    # print 'io_phrases:', io_phrases
    phrases = []
    for (kn,tn), (kpre,tpre) in izip(io_phrases, [(None,None)]+io_phrases):
        if kn == 'O':
            phrases.extend(tn)
        else:
            # tn[0]
            if not kn is None and kn == kpre: phrases.append('B-'+tn[0])
            else: phrases.append('I-'+tn[0])
            # end tn[0]

            # tn[1:]
            phrases.extend(map(lambda x: 'I-'+x, tn[1:]))
            # end tn[1:]
    return phrases

def test():
    trn, dev, tst = get_sents(enc='bio')
    sent = random.choice(trn)
    print ' '.join(sent['ws'])
    print ' '.join(sent['ts'])
    print ' '.join(bio2io(sent['ts']))
    print ' '.join(bio2bilou(sent['ts']))

    ts_gold = [sent['ts'] for sent in dev]
    ts_pred = [io2bio(bio2io(sent['ts'])) for sent in dev]
    r1,r2 = conlleval(ts_gold, ts_pred)
    print r2
    print '\t'.join(map(str,r1))

if __name__ == '__main__':
    test()
