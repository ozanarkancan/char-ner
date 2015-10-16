import re, random, sys
from itertools import *
from tabulate import tabulate

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

def main():
    lines = [l.strip() for l in sys.stdin]
    for l in lines:
        ws, tphrases = process_sent(l)
        ts = []
        for tphrase in tphrases:
            if tphrase[0] == 'O':
                ts.extend(tphrase)
            else:
                ts.append('B-'+tphrase[0])
                ts.extend('I-'+t for t in tphrase[1:])
        for w,t in zip(ws,ts):
            print '{}\t{}'.format(w,t)
        print

if __name__ == '__main__':
    lines = [l.strip() for l in sys.stdin]
    print sum(sum(map(l.count, ('[ORG','[LOC','[PER'))) for l in lines)
    # main()
