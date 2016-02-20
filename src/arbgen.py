from collections import Counter
import sys, random, argparse, os, shutil
from utils import DATA_DIR


def write_to_file(dset, fname):
    with open(fname,'w') as out:
        for sent in dset:
            for w,t in zip(sent['ws'],sent['ts']):
                out.write('{}\t{}\n'.format(w.encode('utf-8'),t))
            out.write('\n')

if __name__ == '__main__':
    parser = argparse.ArgumentParser(prog="create arb dset with random seed", formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('--seed', required=True, type=int)
    args = parser.parse_args()

    dset = []
    with open('{}/ANERCorp'.format(DATA_DIR)) as src:
        sent = {'ws':[],'ts':[]}
        for l in src:
            l = l.strip()
            w,t = l.split(' ')
            sent['ws'].append(w.decode('utf-8'))
            sent['ts'].append(t.replace('PERS','PER'))
            if l in  ['. O', '? O', '! O']:
                dset.append(sent)
                sent = {'ws':[],'ts':[]}

    print 'num of sents in ANERCorp:', len(dset)
    random.seed(args.seed)
    dset = filter(lambda sent: len(' '.join(sent['ws']))<=500, dset)
    random.shuffle(dset)
    trn_size = len(dset) - len(dset)/6
    trn, dev = dset[:trn_size], dset[trn_size:]
    print 'trn:{} dev:{}'.format(*map(len,(trn,dev)))

    dirname = 'arb%d'%args.seed
    dirpath = '{}/{}'.format(DATA_DIR,dirname)
    print dirpath
    files = map(lambda x: x.format(DATA_DIR, dirname),('{}/{}/train.bio','{}/{}/testa.bio','{}/{}/testb.bio'))
    print files
    if raw_input('continue?[y/n]:') == 'y':
        os.mkdir(dirpath)
        for d,f in zip((trn,dev,dev),files): write_to_file(d,f)
        print 'done'
    else:
        print 'bye'

