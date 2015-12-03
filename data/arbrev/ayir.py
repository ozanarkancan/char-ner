from collections import Counter
import sys, random


if __name__ == '__main__':
    random.seed(7)
    with open('ANERCorp') as src:
        dset = []
        sent = {'ws':[],'ts':[]}
        for l in src:
            l = l.strip()
            w,t = l.split(' ')
            w_uni = w.decode('utf-8')
            sent['ws'].append(''.join(reversed(w_uni)).encode('utf-8'))
            sent['ts'].append(t.replace('PERS','PER'))
            if l in  ['. O', '? O', '! O']:
                dset.append(sent)
                sent = {'ws':[],'ts':[]}
    print 'num of sents in ANERCorp:',len(dset)
    random.shuffle(dset)
    trn = dset[:4000]
    dev = dset[4000:]

    with open('train.bio','w') as out:
        for sent in trn:
            for w,t in zip(sent['ws'],sent['ts']):
                out.write('%s\n'%'\t'.join((w,t)))
            out.write('\n')

    with open('testa.bio','w') as out:
        for sent in dev:
            for w,t in zip(sent['ws'],sent['ts']):
                out.write('%s\n'%'\t'.join((w,t)))
            out.write('\n')

    with open('testb.bio','w') as out:
        for sent in dev:
            for w,t in zip(sent['ws'],sent['ts']):
                out.write('%s\n'%'\t'.join((w,t)))
            out.write('\n')
