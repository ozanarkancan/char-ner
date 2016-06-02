import argparse, subprocess
import getpass

STR = """#!/bin/bash
#$ -N {}-{}
#$ -q {}.q{}
#$ -S /bin/bash
#$ -l gpu={}
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
    parser = argparse.ArgumentParser(prog="sendjob", formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument("--p", default=False, action='store_true', help='just print, dont submit') 
    parser.add_argument("--script", default='exper', help='python script to run')
    parser.add_argument("--script_args", required=True, help='arguments to python script') 
    parser.add_argument("--m", default='', choices=['', 'ahtapot-5-1', 'biyofiz-4-0','biyofiz-4-1','biyofiz-4-2','biyofiz-4-3','parcore-6-0','iui-5-0'], help='machine name')
    parser.add_argument("--d", default='gpu', choices=['gpu','cpu','gpu0','gpu1','gpu2','gpu3','gpu4','gpu5','gpu6','gpu7'], help='device name')
    parser.add_argument("--smp", default=18, type=int, help='num of cpu threads')
    args = parser.parse_args()

    username = getpass.getuser()
    is_gpu, machine = 0, ''
    if args.d.startswith('gpu'):
        args.smp, is_gpu = 1, 1

    if len(args.m):
        queue = args.m.split('-')[0]
        if queue == 'ahtapot':
            queue = 'ai'
        elif queue == 'parcore' or queue == 'iui':
            queue = 'all'
        else:
            queue = 'biyofiz'
        machine = '@%s'%args.m
    else:
        queue = 'biyofiz'

    job_text = STR.format(args.script, args.d, queue, machine, is_gpu, args.smp, username, username, username, args.smp, args.d, args.script, args.script_args)
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
