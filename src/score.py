import re, subprocess
from utils import SRC_DIR


def conlleval(ts_gold, ts_pred):
    text = ''
    for ts1, ts2 in zip(ts_gold, ts_pred):
        for t1, t2 in zip(ts1, ts2):
            text += 'x x x %s %s\n'%(t1,t2)
        text += '\n'

    proc = subprocess.Popen(
        '%s/conlleval'%SRC_DIR,stdout=subprocess.PIPE,
        stdin=subprocess.PIPE)
    proc.stdin.write(text)
    proc.stdin.close()
    result = proc.stdout.read()
    res = re.findall('\d+.\d+', result.splitlines()[1])
    proc.wait()
    return map(float,res), result

