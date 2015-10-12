import argparse
from utils import ROOT_DIR
from os import walk
import re
import pandas as pd

LOG_DIR = '{}/logs'.format(ROOT_DIR)

def get_arg_parser():
    parser = argparse.ArgumentParser(prog="results")

    parser.add_argument("--job", default='collect', choices=['collect'], help= "job choice")
    
    return parser

def get_lines(all_lines, pattern):
    lines = []
    for line in all_lines:
        stripped = line.strip()
        if stripped.startswith(pattern):
            lines += [stripped]

    return lines

def get_experiment_frame(lines):
    df = pd.DataFrame(columns=['epoch', 'mcost', 'mtime', 'cerr', 'werr', 'wacc',
        'precision', 'recall', 'f1', 'best', 'best_epoch'])
    for i in range(len(lines)):
        df.loc[i] = map(float, lines[i].split()[1:])

    return df

def get_lang(lines):
    lang = "eng"
    for l in lines:
        if "lang" in l:
            lang = l.strip().split()[-1]
            break
    return lang

def collect():
    all_results = pd.DataFrame(columns=['file_name', 'lang', 'trn_best', 'dev_best', 'trn_logs', 'dev_logs'])

    files = []
    for (_, _, fnames) in walk(LOG_DIR):
        files += fnames

    files = filter(lambda s: s.endswith("info"), files)

    i = 0
    
    for fname in files:
        with open(LOG_DIR + "/" + files[0], 'r') as f:
            
            lines = f.readlines()
            trn_lines = get_lines(lines, "trn")
            dev_lines = get_lines(lines, "dev")
            
            trn_df = get_experiment_frame(trn_lines)
            dev_df = get_experiment_frame(dev_lines)
            
            all_results.loc[i] = [fname, get_lang(lines), trn_df['best'].max(),
                dev_df['best'].max(), trn_df, dev_df]

            i += 1
    print all_results

if __name__ == "__main__":
    parser = get_arg_parser()
    args = vars(parser.parse_args())

    if args['job'] == 'collect':
        collect()
    

