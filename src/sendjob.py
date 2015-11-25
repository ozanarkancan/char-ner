import argparse, subprocess
from utils import ROOT_DIR
import getpass

STR = """#!/bin/bash
#$ -N exper-{}
#$ -q {}.q@{}
#$ -S /bin/bash
##$ -l h_rt=00:59:00 #how many mins run
#$ -pe smp {}
#$ -cwd
#$ -o /dev/null
#$ -e /mnt/kufs/scratch/{}/char-ner/job.err
#$ -M {}@ku.edu.tr
#$ -m bea
 
source ~/setenv.sh
cd /mnt/kufs/scratch/{}/char-ner
MKL_NUM_THREADS={} THEANO_FLAGS=mode=FAST_RUN,device={},floatX=float32 python src/{}.py {}
"""

if __name__ == '__main__':
    parser = argparse.ArgumentParser(prog="creatjob")
    parser.add_argument("--p", type=bool, default=False, help='just print, dont submit') 
    parser.add_argument("--script", default='exper')
    parser.add_argument("--script_args", required=True) 
    parser.add_argument("--m", required=True, choices=['biyofiz-4-0','biyofiz-4-1','biyofiz-4-2','biyofiz-4-3','parcore-6-0','iui-5-0'])
    parser.add_argument("--d", required=True, choices=['gpu','cpu','gpu0','gpu1'])
    parser.add_argument("--smp", default=18, type=int)
    args = parser.parse_args()

    username = getpass.getuser()
    args.smp = 1 if args.d.startswith('gpu') else args.smp
    queue = args.m.split('-')[0]
    queue = queue if queue == 'biyofiz' else 'all'

    job_text = STR.format(args.d, queue, args.m, args.smp, username, username, username, args.smp, args.d, args.script, args.script_args)
    print job_text

    if not args.p:
        proc = subprocess.Popen(
            'qsub',stdout=subprocess.PIPE,
            stdin=subprocess.PIPE,shell=True)
        proc.stdin.write(job_text)
        proc.stdin.close()
        result = proc.stdout.read()
        proc.wait()
        print 'result:', result
