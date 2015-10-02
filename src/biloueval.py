def bilou_post_correct(ts):
    ts_corrected = []
    for t, tnext in zip(ts, ts[1:]+[None]):
        if t.startswith('B'):
            pos,type = t.split('-')
            if tnext == None:
                ts_corrected.append('O') # B cannot be at the end of sent
            else:
                pos_next, type_next =  ('O', '') if tnext == 'O' else tnext.split('-')
                if pos_next in ('I','L') and type == type_next:
                    ts_corrected.append(t)
                else: # invalid tagging
                    ts_corrected.append('O')
        elif t.startswith('I'):
            pos,type = t.split('-')
            if tnext == None:
                ts_corrected.append('O') # I cannot be at the end of sent
            else:
                pos_next, type_next =  'O', '' if tnext == 'O' else tnext.split('-')
                if pos_next == 'L' and type == type_next:
                    ts_corrected.append(t)
                else: # invalid tagging
                    ts_corrected.append('O')
        elif t.startswith('L'): # it can be followed by anything
            ts_corrected.append(t)
        elif t.startswith('O'): # it can be followed by anything
            ts_corrected.append(t)
        elif t.startswith('U'): # it can be followed by anything
            ts_corrected.append(t)
        else:
            raise Exception()
    return ts_corrected

def bilouEval2(ytrue, ypred):
    gl, gll = [], []
    plist=[]
    tokenc, pphrase = 0,0
    correctp=0

    ln = 0
    for tsg, ts in zip(ytrue,ypred):
        for g,p in zip(tsg,ts):
            tokenc+=1
            plist.append(p)
            if g==p: correctp+=1

            if g.startswith('B'):
                gll = [(g,ln)]
            if g.startswith('I'):
                gll.append((g,ln))
            if g.startswith('L'):
                gll.append((g,ln))
                gl.append(gll)
                gll = []
            if g.startswith('U'):
                gl.append([(g,ln)])

            if p.startswith('B') or p.startswith('U'):
                pphrase+=1
            ln += 1
        else:
            plist.append(' ')
            ln += 1
    gphrase = len(gl)
    cphrase = correct(gl,plist)
    precision, recall, F1 = 0, 0, 0
    if pphrase:
        precision = float(cphrase)/pphrase
    if gphrase:
        recall = float(cphrase)/gphrase
    if precision or recall:
        F1 = (precision*recall)/(precision+recall)*200
    return float(correctp)/tokenc*100, precision*100, recall*100, F1

def bilouEval(sents):
    gl, gll = [], []
    plist=[]
    tokenc, pphrase = 0,0
    correctp=0

    ln = 0
    for sent in sents:
        for g,p in zip(sent['tsg'],sent['ts']):
            tokenc+=1
            plist.append(p)
            if g==p: correctp+=1

            if g.startswith('B'):
                gll = [(g,ln)]
            if g.startswith('I'):
                gll.append((g,ln))
            if g.startswith('L'):
                gll.append((g,ln))
                gl.append(gll)
                gll = []
            if g.startswith('U'):
                gl.append([(g,ln)])

            if p.startswith('B') or p.startswith('U'):
                pphrase+=1
            ln += 1
        else:
            plist.append(' ')
            ln += 1
    gphrase = len(gl)
    cphrase = correct(gl,plist)
    precision, recall, F1 = 0, 0, 0
    if pphrase:
        precision = float(cphrase)/pphrase
    if gphrase:
        recall = float(cphrase)/gphrase
    if precision or recall:
        F1 = (precision*recall)/(precision+recall)*200
    return float(correctp)/tokenc*100, precision*100, recall*100, F1

def correct(gl, plist):
    cn = 0
    for l in gl:
        cn+= all([plist[t[1]]==t[0] for t in l])
    return cn

if __name__ == '__main__':
    #main()
    import json
    with open('sents.toy.json') as src:
        sents = json.load(src)
    llen = len(sents['devSents'][1]['ts'])
    sents['devSents'][1]['ts'] = ['O']*llen
    print bilouEval(sents['devSents'])
