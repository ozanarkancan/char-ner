import random
from itertools import *
from utils import get_sents, sample_sents
from score import conlleval


def bio2bilou(ts):
    from bio2bilou import c4
    return c4(' '.join(ts))


def bilou2uni(ts):
    return [e.split('-')[1] if len(e.split('-')) == 2 else e for e in ts]

def bio2uni(ts):
    return bilou2uni(ts)

def uni2bio(ts):
    uni_phrases = [(key,list(subite)) for key, subite in groupby(ts)]
    # print 'uni_phrases:', uni_phrases
    phrases = []
    for (kn,tn), (kpre,tpre) in izip(uni_phrases, [(None,None)]+uni_phrases):
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
    print ' '.join(bio2uni(sent['ts']))
    print ' '.join(bio2bilou(sent['ts']))

    ts_gold = [sent['ts'] for sent in dev]
    ts_pred = [uni2bio(bio2uni(sent['ts'])) for sent in dev]
    r1,r2 = conlleval(ts_gold, ts_pred)
    print r2
    print '\t'.join(map(str,r1))

if __name__ == '__main__':
    from utils import sample_sents
    from stats import num_of_phrases_sent

    trn, dev, tst = get_sents()
    sents = sample_sents([sent for sent in trn if num_of_phrases_sent(sent) > 2],10)
    for sent in sents:
        print sent['ts']
        pass
    """
    ts = ['B-PER','L-PER','O','U-PER']
    print ts
    print bilou_post_correct(ts)
    print sum(1 if t.startswith('B') or t.startswith('U') else 0 for t in ts)
    """

