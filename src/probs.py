import numpy as np
import argparse
from tabulate import tabulate

def is_consec(sent):
    return any(t1.startswith('I-') and t2.startswith('B-') and t1.split('-')[1] == t2.split('-')[1]
            for t1,t2 in zip(sent['ts'],sent['ts'][1:]))

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('fname')
    args = parser.parse_args()
    dat = np.load(args.fname)
    for sent, prob, pred in zip(dat['dev'],dat['probs'],dat['preds']):
        if is_consec(sent):
            print tabulate([sent['ws'],sent['ts']])
            print tabulate([sent['cseq'],sent['tseq'],pred]) 
