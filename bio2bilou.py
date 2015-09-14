"""
converts from bio to bilou
"""
import sys, itertools as ite
test_str= """O
O
O
O
I-MISC
O
O
I-PER
I-PER
O
O
O
I-LOC
I-LOC
B-LOC
I-LOC
I-LOC
O
O
I-LOC
O
O
O
I-LOC
I-LOC
O
O"""


def c1(ss):
    ll=[]
    l=[]
    for s in ss.split(' '):
        if s == 'O':
            if len(l): ll.append(l)
            l=[]
            ll.append([s])
        else:
            l.append(s)
    if len(l):
        ll.append(l)
    return ll

#ttags = ['I-LOC', 'I-LOC', 'B-LOC', 'I-LOC', 'B-LOC']
def c2(taglist):
    ll,l=[],[]
    for t in taglist:
        if t.startswith('B-'):
            ll.append(l)
            l=[t]
        else:
            l.append(t)
    if len(l):
        ll.append(l)
    return ll

def c3(ss):
    tlist=[]
    for tags in c1(ss):
        if len(tags)>1:
            for tt in c2(tags):
                tlist.append(tt)
        else:
            tlist.append(tags)
    return tlist

def c4(ss):
    flist=[]
    for tags in c3(ss):
        if len(tags)==1:
            if tags[0]=='O':
                flist.append('O')
            else:
                ttype = tags[0][2:]
                flist.append('U-'+ttype)
        elif len(tags)==2:
            ttype = tags[0][2:]
            flist.extend(['B-'+ttype,'L-'+ttype])
        else:
            try:
                ttype = tags[0][2:]
            except:
                print >> sys.stderr, ss
                print >> sys.stderr, tags
                sys.exit(1)
            icount = len(tags)-2
            flist.append('B-'+ttype)
            for i in range(icount): flist.append('I-'+ttype)
            flist.append('L-'+ttype)
    return flist

def main():
    lines = [l.strip() for l in sys.stdin]
    tags = ite.imap(lambda x: x.split(' ')[3] if len(x) else x, lines)
    btags = []
    for ss in mygen(tags):
        #print ss+'$'
        #print c4(ss)
        btags.extend(c4(ss))
        btags.append('')
    for old, new in ite.izip(lines, btags):
        if len(new):
            print '%s %s'%(old,new)
        else:
            print

def mygen(tags):
    l=[]
    for t in tags:
        if len(t):
            l.append(t)
        else:
            yield ' '.join(l)
            l=[]

if __name__ == '__main__':
    print c4(' '.join(['O','O','I-PER','B-PER']))
    # main()

