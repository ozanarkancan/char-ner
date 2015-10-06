import argparse
from utils import ROOT_DIR

STR = """#!/bin/bash
#$ -N gpu
#$ -q biyofiz.q@biyofiz-4-{}
#$ -S /bin/bash
##$ -l h_rt=05:00:00 #how many mins run
#$ -pe smp 1
#$ -cwd
#$ -o /tmp/job.out
#$ -e /tmp/job.err
#$ -M okuru13@ku.edu.tr
#$ -m bea
 
source ~/setenv.sh
cd /mnt/kufs/scratch/okuru13/char-ner
python src/{}.py {}
"""

def valid_file_name(s):
    return "".join(i for i in s if i not in "\"\/ &*?<>|[]()'-")

if __name__ == '__main__':
    parser = argparse.ArgumentParser(prog="creatjob")
    parser.add_argument("--script", required=True) 
    parser.add_argument("--script_args", required=True) 
    parser.add_argument("--biyofiz", required=True, type=int, choices=range(4)) 
    args = parser.parse_args()
    job_text = STR.format(args.biyofiz, args.script, args.script_args)
    file_name_param = valid_file_name(args.script_args.replace(' ','.')) if args.script_args else 'defaults'
    job_file_name = 'biyofiz{}.'.format(args.biyofiz)+file_name_param+'.job'
    job_file_path = '{}/jobs/{}'.format(ROOT_DIR,job_file_name)
    with open(job_file_path,'w') as out:
        out.write(job_text)
    print job_file_path
