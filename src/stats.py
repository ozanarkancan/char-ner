import numpy as np
from collections import Counter
from itertools import product

import encoding
from utils import get_sents

""" assumes sents are in iob format """



### SENT ###
def contains_consecutive_same_type(sent):
    encoding.iob2
    for t1,t2 in zip(sent['ts'], sent['ts'][1:]):
        if t1 != 'O' and t2 != 'O':
            t1_pos, t1_type = t1.split('-')
            t2_pos, t2_type = t2.split('-')
            if t1_type == t2_type:
                if (t1_pos,t2_pos) in [e for e in product(['U','L'],['B','U'])]:
                    return True
    return False

def num_of_phrases_sent(sent):
    ts = encoding.iob2bilou(sent['ts'])
    return sum(1 for t in ts if t.startswith('B') or t.startswith('U'))

### end SENT ###

### DSET ###

def stat_num_of_chars(dset):
    a = np.mean([len(''.join(sent['ws'])) for sent in dset])
    return np.mean(a), np.std(a)

def stat_num_of_words(dset):
    a = [len(sent['ws']) for sent in dset]
    return np.mean(a), np.std(a)

def num_of_consecutive_same_type(dset):
    return sum(1 for sent in dset for t in sent['ts'] if t.startswith('B'))

def entity_tagged_vocab(dset):
    return set(w for sent in dset for w,t in zip(sent['ws'],sent['ts']) if t != 'O')

def unk_perc(trn, dset):
    trn_vocab = entity_tagged_vocab(trn)
    cntr = Counter(w for sent in dset for w,t in zip(sent['ws'],sent['ts']) if t != 'O')
    z = sum(v for k,v in cntr.iteritems())
    nom = sum(v for k,v in cntr.iteritems() if not k in trn_vocab)
    return nom/float(z)

def num_of_phrases(dset):
    return sum(num_of_phrases_sent(sent) for sent in dset)

def io_ideal(dev,tst):
    from score import conlleval
    print 'io tagging ideal scores'
    for dset, dset_str in zip((dev,tst),('dev','tst')):
        ts_gold = [sent['ts'] for sent in dset]
        ts_pred = [encoding.any2io(sent['ts']) for sent in dset]
        r1,r2 = conlleval(ts_gold, ts_pred)
        print '\t'.join([dset_str]+map(str,r1))

def get_vocab(dset):
    return set(w for sent in dset for w in sent['ws'])

def main():
    langs = ['eng', 'deu', 'spa', 'ned', 'tr', 'cze', 'ger', 'arb0', 'ita']
    # langs = ['eng', 'deu']
    dsetnames = ['trn','dev','tst']

    data = dict((lang,dict((dname,dset) for dname,dset in zip(dsetnames, get_sents(lang)))) for lang in langs)

    for l in langs:
        print l, sorted(set(t for sent in data[l]['trn'] for t in sent['ts']))
    print 

    table = []
    for l in langs:
        table.append([l,sum(1 for sent in data[l]['trn'] if len(' '.join(sent['ws'])) > 500)])
    print tabulate(table)

    table = []
    for dname in dsetnames:
        table.append([dname]+map(len,[data[l][dname] for l in langs]))
    print tabulate(table,headers=['#sent']+langs, tablefmt='latex')
    print

    table = []
    for dname in dsetnames:
        table.append([dname]+[sum(len(sent['ws']) for sent in data[l][dname]) for l in langs])
    print tabulate(table,headers=['#token']+langs)
    print

    table = []
    for dname in dsetnames:
        table.append([dname]+[float(sum(len([c for w in sent['ws'] for c in w]) for sent in data[l][dname])) for l in langs])
    print tabulate(table,headers=['#char']+langs,floatfmt='.1e')
    print

    table = []
    for l in langs:
        # nchar_sents = [sum(1 for w in sent['ws']) for sent in chain(*data[l].values())]
        for dname in dsetnames:
            nchar_sents = [sum(1 for w in sent['ws']) for sent in data[l][dname]]
            table.append(['{}-{}'.format(l,dname)]+[int(f(nchar_sents)) if len(nchar_sents) else 0 for f in (np.min,np.max,np.mean,np.std)])
        table.append(['...']*5)
    print tabulate(table,headers=['#word per sent']+['min','max','mean','std'])
    print

    table = []
    for l in langs:
        # nchar_sents = [sum(1 for c in ' '.join(sent['ws'])) for sent in chain(*data[l].values())]
        for dname in dsetnames:
            nchar_sents = [sum(1 for c in ' '.join(sent['ws'])) for sent in data[l][dname]]
            table.append(['{}-{}'.format(l,dname)]+[int(f(nchar_sents)) if len(nchar_sents) else 0 for f in (np.min,np.max,np.mean,np.std)])
        table.append(['...']*5)
    print tabulate(table,headers=['#char per sent']+['min','max','mean','std'])
    print

    table = []
    for dname in dsetnames:
        table.append([dname]+[len(get_vocab(data[l][dname])) for l in langs])
    print tabulate(table,headers=['size(vocab)']+langs)
    print

    table = []
    for l, dname in product(langs,('dev','tst')):
        vdst = get_vocab(data[l][dname])
        vsrc = get_vocab(data[l]['trn'])
        vdiff = vdst.difference(vsrc)
        uperc = len(vdiff) / float(len(vdst)) * 100

        cnt = Counter(w for sent in data[l][dname] for w,t in zip(sent['ws'],sent['ts']) if t!='O')
        pperc = sum(cnt[w] for w in vdiff) / float(sum(cnt.values())) * 100

        cnt = Counter(w for sent in data[l][dname] for w in sent['ws'])
        cperc = sum(cnt[w] for w in vdiff) / float(sum(cnt.values())) * 100


        table.append([l+'-'+dname]+[uperc, pperc, cperc])
    print tabulate(table, headers=['unk', 'unique', 'phrase', 'corpus'], floatfmt='.2f')

    table = []
    for l, dname in product(langs,('dev','tst')):
        dset = data[l][dname]
        ts_gold = [sent['ts'] for sent in dset]
        ts_pred = [encoding.any2io(sent['ts']) for sent in dset]
        r1,r2 = conlleval(ts_gold, ts_pred)
        table.append([l+'-'+dname]+map(str,r1))
    print tabulate(table, headers=['io-ideal', 'wacc','pre','rec','f1'])
    print


def paper():
    # langs = ['eng', 'deu', 'spa', 'ned', 'tr', 'cze', 'ger', 'arb', 'ita']
    # langs = ['arb0', 'cze', 'ned', 'eng', 'deu', 'spa', 'tr']
    langs = ['cze-pos', 'eng-pos', 'deu-pos', 'spa-pos', 'pos', 'chu']
    dsetnames = ['trn','dev','tst']

    data = dict((lang,dict((dname,dset) for dname,dset in zip(dsetnames, get_sents(lang)))) for lang in langs)

    table = []
    for l in langs:
        table.append([l]+map(len,[data[l][dname] for dname in dsetnames]))
    print tabulate(np.array(table).T,headers=['#sent']+dsetnames, tablefmt='latex')
    print


    table = []
    for l in langs:
        # nchar_sents = [sum(1 for c in ' '.join(sent['ws'])) for sent in chain(*data[l].values())]
        # for dname in dsetnames:
        nchar_sents = [sum(1 for c in ' '.join(sent['ws'])) for dname in dsetnames for sent in data[l][dname]]
        # table.append(['%s'%l]+[int(f(nchar_sents)) for f in (np.min,np.max,np.mean,np.std)])
        table.append(['%s'%l]+[int(f(nchar_sents)) for f in (np.mean,np.std)])
    print tabulate(table,headers=['#char per sent']+['mean','std'], tablefmt='latex')
    print

    table = []
    for l in langs:
        # char_set = set(c for dname in dsetnames for sent in data[l][dname] for c in ''.join(sent['ws']))
        char_set = set(c for dname in ('trn','dev') for sent in data[l][dname] for w in sent['ws'] for c in w)
        # char_set = set(c for sent in data[l]['trn'] for w in sent['ws'] for c in w)
        tag_set = set(t for dname in dsetnames for sent in data[l][dname] for t in encoding.any2io(sent['ts']))
        # table.append(['%s'%l]+[int(f(nchar_sents)) for f in (np.min,np.max,np.mean,np.std)])
        table.append(['%s'%l, len(char_set)+1, len(tag_set)])
    print tabulate(table,headers=['i/o']+['input','output'], tablefmt='latex')
    print

    table = []
    for l in langs:
        # char_set = set(c for dname in dsetnames for sent in data[l][dname] for c in ''.join(sent['ws']))
        # char_set = set(c for dname in dsetnames for sent in data[l][dname] for w in sent['ws'] for c in w)
        char_set = set(c for sent in data[l]['trn'] for w in sent['ws'] for c in w)
        tag_set = set(t for dname in dsetnames for sent in data[l][dname] for t in encoding.any2io(sent['ts']))
        # table.append(['%s'%l]+[int(f(nchar_sents)) for f in (np.min,np.max,np.mean,np.std)])
        table.append(['%s'%l, len(char_set), len(tag_set)])
    print tabulate(table,headers=['i/o']+['input','output'], tablefmt='latex')
    print


    table = []
    # for l, dname in product(langs,('dev','tst')):
    for l in langs:
        dname = 'tst'
        vdst = get_vocab(data[l][dname])
        vsrc = get_vocab(data[l]['trn'])
        vdiff = vdst.difference(vsrc)
        uperc = len(vdiff) / float(len(vdst)) * 100

        cnt = Counter(w for sent in data[l][dname] for w,t in zip(sent['ws'],sent['ts']) if t!='O')
        pperc = sum(cnt[w] for w in vdiff) / float(sum(cnt.values())) * 100

        cnt = Counter(w for sent in data[l][dname] for w in sent['ws'])
        cperc = sum(cnt[w] for w in vdiff) / float(sum(cnt.values())) * 100


        table.append([l+'-'+dname]+[uperc, pperc, cperc])
    print tabulate(np.array(table).T, headers=['unk', 'unique', 'phrase', 'corpus'], tablefmt='latex', floatfmt='.2f')

### end DSET ###

def quick():
    """
    langs = ['eng', 'deu', 'spa', 'ned', 'tr', 'cze', 'ger', 'arb0', 'ita']
    # langs = ['eng', 'deu']
    dsetnames = ['trn','dev','tst']
    data = dict((lang,dict((dname,dset) for dname,dset in zip(dsetnames, get_sents(lang)))) for lang in langs)
    """
    from exper import Dset
    from functools import partial

    Dset2 = partial(Dset, captrn=0)
    # dsets = map(Dset, ('eng','deu','spa','ned','tr','cze','arb0'))
    # langs = ('arb0', 'cze', 'ned', 'eng', 'deu', 'spa', 'tr')
    langs = ('arb0', 'cze', 'arb-pos', 'cze-pos')
    dsets = map(Dset2, langs)
    # cvocabs = map(lambda dset: set(c for sent in dset.trn for c in sent['cseq']), dsets)
    table = []
    table.append(map(lambda dset: sum(len(sent['ws']) for sent in dset.trn ), dsets))
    """
    table.append(map(lambda dset: len(set(c for sent in dset.trn for c in sent['cseq'])), dsets))
    table.append(map(lambda dset: len(set(c for sent in dset.trn for c in sent['ws'])), dsets))
    table.append(map(lambda dset: int(np.mean([len(sent['cseq']) for sent in dset.trn])), dsets))
    table.append(map(lambda dset: int(np.mean([len(sent['ws']) for sent in dset.trn])), dsets))
    """
    print tabulate(table, tablefmt='latex', floatfmt='.2f')



if __name__ == '__main__':
    from tabulate import tabulate
    from score import conlleval

    paper()

