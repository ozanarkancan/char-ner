from utils import get_sents
import numpy as np

class Stats:
    def __init__(self):
        trn,dev,tst = get_sents()
        print np.mean([len(' '.join(sent['ws'])) for sent in trn])
        print np.std([len(' '.join(sent['ws'])) for sent in trn])

if __name__ == '__main__':
    stt = Stats()
