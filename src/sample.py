import argparse, random, os
from utils import get_sents

def get_args():
    parser = argparse.ArgumentParser(prog="sample")

    parser.add_argument("dset", help="dataset")
    parser.add_argument("nums", nargs='+', type=int, help="num of sents to sample from dataset for train, dev, tst (in the order of K)")
    args = vars(parser.parse_args())

    return args

def write_to_file(fpath, dset_part):
    with open(fpath,'w') as src:
        for sent in dset_part:
            for w,t in zip(sent['ws'], sent['ts']):
                src.write(('%s\t%s\n'%(w,t)).encode('utf-8'))
            src.write('\n')


def get_sample(l, k):
    rand_indices = random.sample(xrange(len(l)),k)
    return [l[i] for i in rand_indices]

if __name__ == '__main__':
    random.seed(7)
    args = get_args()
    print args
    trn, dev, tst = get_sents(args['dset'])
    dset_parts = (trn,dev,tst)
    print map(len, (trn,dev,tst))
    strn, sdev, stst = map(get_sample, dset_parts, map(lambda x: x*1000, args['nums']))
    print map(len, (strn,sdev,stst))

    # filter out sents in sdev & stst if they contain a tag that is not in strn
    trn_tags = set(t for sent in strn for t in sent['ts'])
    sdev= filter(lambda sent: all(t in trn_tags for t in sent['ts']), sdev)
    stst= filter(lambda sent: all(t in trn_tags for t in sent['ts']), stst)
    print map(len, (strn,sdev,stst))

    dir_name = 'data/%s-sample'%args['dset']
    os.mkdir(dir_name)

    for fname, dpart in zip(('train.bio','testa.bio', 'testb.bio'), (strn,sdev,stst)):
        fpath = '%s/%s'%(dir_name,fname)
        print fpath
        write_to_file(fpath, dpart)

