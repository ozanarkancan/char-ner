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

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('mode', choices=['train','test'])
    args = parser.parse_args()
    dset = get_sents('reyyan.{}.txt'.format(args.mode))
    if args.mode == 'train':
        sents = []
        for sent in dset:
            sents.extend(get_subsents(sent))
        for sent in sents:
            for w,t in zip(sent['ws'],sent['ts']):
                print '{}\t{}'.format(w,t)
            print
    elif args.mode == 'test':
        for sent in dset:
            for w,t in zip(sent['ws'],sent['ts']):
                print '{}\t{}'.format(w,t)
            print
