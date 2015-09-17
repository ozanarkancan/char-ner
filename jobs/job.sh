#!/bin/bash
#$ -N sleep_for_gpu
#$ -q biyofiz.q@biyofiz-4-0
#$ -S /bin/bash
##$ -l h_rt=05:00:00 #how many mins run
#$ -pe smp 1
#$ -cwd
#$ -o tmp/job.out
#$ -e tmp/job.err
#$ -M okuru13@ku.edu.tr
#$ -m bea
 
source ~/setenv.sh
cd /mnt/kufs/scratch/okuru13/char-ner
python src/exper_lasagne.py
