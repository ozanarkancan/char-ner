from collections import Counter
import sys


if __name__ == '__main__':
    with open('ANERCorp') as src:
        dset = []
        sent = {'ws':[],'ts':[]}
        for l in src:
            l = l.strip()
            w,t = l.split(' ')
            sent['ws'].append(w)
            sent['ts'].append(t.replace('PERS','PER'))
            if l in  ['. O', '? O', '! O']:
                dset.append(sent)
                sent = {'ws':[],'ts':[]}
    print 'num of sents in ANERCorp:',len(dset)

    with open('train.bio','w') as out:
        for sent in dset:
            for w,t in zip(sent['ws'],sent['ts']):
                out.write('%s\n'%'\t'.join((w,t)))
            out.write('\n')
