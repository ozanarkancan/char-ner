import re, random, sys
from itertools import *
from tabulate import tabulate
import argparse

def process_sent(l):
    tokens = re.split(r' +',l.strip())
    ws = []; ts = []
    cur_tag = 'O'
    tphrase = []
    for w in tokens:
        if w in ('[ORG','[LOC','[PER'):
            if len(tphrase): ts.append(tphrase)
            tphrase = []
            cur_tag = w[1:]
        elif w == ']':
            if len(tphrase): ts.append(tphrase)
            tphrase = []
            cur_tag = 'O'
        else:
            ws.append(w)
            tphrase.append(cur_tag)
    if len(tphrase): ts.append(tphrase)
    assert sum(len(p) for p in ts)==len(ws)
    return ws, ts

def get_subsents(sent):
    subsents = []
    cursubsent = {'ws':[],'ts':[]}
    for w,t in zip(sent['ws'],sent['ts']):
        cursubsent['ws'].append(w)
        cursubsent['ts'].append(t)
        if w == '.' and t == 'O': # not breaking phrase boundary
            subsents.append(cursubsent)
            cursubsent = {'ws':[],'ts':[]}
    if len(cursubsent['ws']) > 0:
        subsents.append(cursubsent)
    return subsents

def get_sents(fname):
    # lines = [l.strip() for l in sys.stdin]
    with open(fname) as src:
        lines = [l.strip() for l in src]
    sents = []
    for l in lines:
        ws, tphrases = process_sent(l)
        ts = []
        for tphrase in tphrases:
            if tphrase[0] == 'O':
                ts.extend(tphrase)
            else:
                ts.append('B-'+tphrase[0])
                ts.extend('I-'+t for t in tphrase[1:])

        sents.append({'ws':ws,'ts':ts})
        """
        for w,t in zip(ws,ts):
            print '{}\t{}'.format(w,t)
        print
        """
    return sents

def write_to_file(dset, fname):
    with open(fname,'w') as out:
        for sent in dset:
            for w,t in zip(sent['ws'],sent['ts']):
                out.write('{}\t{}\n'.format(w,t))
            out.write('\n')

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--trn_size', type=int, default=30000)
    args = parser.parse_args()
    trn = [ssent for sent in get_sents('reyyan.train.txt') for ssent in get_subsents(sent)]
    tst = [ssent for sent in get_sents('reyyan.test.txt') for ssent in get_subsents(sent)]
    print map(len, (trn,tst))

    random.seed(7)
    random.shuffle(trn)
    trn, dev = trn[:args.trn_size], trn[args.trn_size:]
    print map(len, (trn,dev,tst))


    write_to_file(trn, 'train.bio')
    write_to_file(dev, 'testa.bio')
    write_to_file(tst, 'testb.bio')
