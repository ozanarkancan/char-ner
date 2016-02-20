import random


def read_sents(file, enc, delim='\t'):
    a = []
    sentences = []
    with open(file) as src:
        for l in src:
            if len(l.strip()):
                a.append(l.strip().split(delim))
            else: # emtpy line
                if len(a):
                    ws = [el[0].decode(enc) for el in a] # TODO
                    ts = [el[-1].upper() for el in a]
                    sentences.append({'ws':ws,'ts':ts})
                a = []
    return sentences

def write_to_file(dset, fname, enc='latin1'):
    with open(fname,'w') as out:
        for sent in dset:
            for w,t in zip(sent['ws'],sent['ts']):
                out.write('{}\t{}\n'.format(w.encode(enc),t))
            out.write('\n')

if __name__ == '__main__':
    random.seed(0)

    TRN_FILE = 'I-CAB-evalita09-NER-training.iob2'
    TST_FILE = 'I-CAB-evalita09-NER-test.iob2'

    trn = read_sents(TRN_FILE, 'latin1')
    tst = read_sents(TST_FILE, 'latin1')
    print len(trn), len(tst)

    random.shuffle(trn)
    trn_size = 10000
    trn, dev = trn[:trn_size], trn[trn_size:]

    write_to_file(trn, 'train.bio')
    write_to_file(dev, 'testa.bio')

