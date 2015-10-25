from itertools import *

def iob2bilou(ts):
    from bio2bilou import c4
    return c4(' '.join(ts))

def bio2iob(ts):
    phrases, curphrase = [], []
    for t in ts:
        if t.startswith('B'):
            len(curphrase) and phrases.append(curphrase)
            curphrase = [t]
        elif t.startswith('I'):
            curphrase.append(t)
        elif t.startswith('O'):
            len(curphrase) and phrases.append(curphrase)
            curphrase = [t]
        else:
            raise Exception()
    len(curphrase) and phrases.append(curphrase)
    ts_iob = []
    for phrase_prev, phrase_now in zip([None]+phrases, phrases):
        if phrase_now == ['O']:
            ts_iob.append('O')
        else:
            now_type = phrase_now[0].split('-')[1]
            if phrase_prev and phrase_prev != ['O']:
                prev_type = phrase_prev[0].split('-')[1]
                if prev_type == now_type:
                    ts_iob.append('B-'+now_type)
                    ts_iob.extend('I-'+now_type for t in phrase_now[1:])
                else:
                    ts_iob.extend('I-'+now_type for t in phrase_now)
            else:
                ts_iob.extend('I-'+now_type for t in phrase_now)
    return ts_iob

def any2io(ts):
    return ['I-'+e.split('-')[1] if len(e.split('-')) == 2 else e for e in ts]

def io2iob(ts):
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
    import random
    trn, dev, tst = get_sents()
    sent = random.choice(trn)
    print ' '.join(sent['ws'])
    print ' '.join(sent['ts'])
    print ' '.join(any2io(sent['ts']))

if __name__ == '__main__':
    test()
