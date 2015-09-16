#!/usr/bin/env python
import subprocess
text = 'x x x I-PER I-LOC\nx x x I-PER I-PER\n'
proc = subprocess.Popen(
    './conlleval',stdout=subprocess.PIPE,
    stdin=subprocess.PIPE)
proc.stdin.write(text)
proc.stdin.close()
result = proc.stdout.read()
print result
proc.wait()
