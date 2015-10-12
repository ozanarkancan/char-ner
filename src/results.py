import argparse
from utils import ROOT_DIR
from os import walk
import re
import pandas as pd
import sys
import seaborn as sns
import matplotlib.pyplot as plt

LOG_DIR = '{}/logs'.format(ROOT_DIR)

def get_arg_parser():
    parser = argparse.ArgumentParser(prog="results")

    parser.add_argument("--job", default='lang_results',
        choices=['lang_results', 'lang_best_plot'], help = "job choice")
    parser.add_argument("--lang", default='eng',
        choices=['eng', 'deu', 'spa', 'ned'], help ='language choice')
    
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
    try:
        all_results = pd.load('all_results.pkl')
    except:
        all_results = pd.DataFrame(columns=['file_name', 'lang', 'trn_best', 'dev_best', 'trn_logs', 'dev_logs'])

    files = []
    for (_, _, fnames) in walk(LOG_DIR):
        files += fnames

    files = filter(lambda s: s.endswith("info"), files)

    i = all_results.shape[0]
    indx = 0
    for fname in files:
        print "Processing: {} / {}\r".format(indx, len(files)),
        sys.stdout.flush()
        indx += 1
        if fname in all_results['file_name'].values.tolist():
            break
        
        with open(LOG_DIR + "/" + fname, 'r') as f:
            try:
                lines = f.readlines()
                trn_lines = get_lines(lines, "trn")
                dev_lines = get_lines(lines, "dev")
            
                trn_df = get_experiment_frame(trn_lines)
                dev_df = get_experiment_frame(dev_lines)
            
                all_results.loc[i] = [fname, get_lang(lines), trn_df['best'].max(),
                    dev_df['best'].max(), trn_df, dev_df]

                i += 1
            except:
                print "Error during processing file: " + fname

    all_results.save('all_results.pkl')
    return all_results

def show_best_results(df):
    g = df.groupby('lang')
    print g['dev_best'].max()

def plot_lang_best(df, lang):
    idx = df.groupby('lang')['dev_best'].transform(max) == df['dev_best']
    trn = df[idx][df['lang'] == lang]['trn_logs'].values[0]
    dev = df[idx][df['lang'] == lang]['dev_logs'].values[0]

    plt.title('{} best experiment'.format(lang))
    plt.xlabel('epoch')
    plt.ylabel('f1')
    plt.plot(trn['epoch'], trn['f1'], label ='trn')
    plt.plot(trn['epoch'], dev['f1'], label = 'dev')
    plt.legend(bbox_to_anchor=(1.002, 1), loc=2, borderaxespad=0.)
    plt.show()    

if __name__ == "__main__":
    parser = get_arg_parser()
    args = vars(parser.parse_args())
    df = collect()

    if args['job'] == "lang_results":
        show_best_results(df)
    elif args['job'] == "lang_best_plot":
        plot_lang_best(df, args['lang'])
        

