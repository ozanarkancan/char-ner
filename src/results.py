import argparse
from utils import ROOT_DIR
from os import walk
import re
import pandas as pd
import sys
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np
import traceback
from tabulate import tabulate

LOG_DIR = '{}/logs'.format(ROOT_DIR)

def get_arg_parser():
    parser = argparse.ArgumentParser(prog="results")

    parser.add_argument("--job", default='lang_results',
        choices=['lang_results', 'lang_best_plot'], help = "job choice")
    parser.add_argument("--lang", default='eng',
        choices=['eng', 'deu', 'spa', 'ned', 'tr', 'cze'], help ='language choice')
    
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

def get_params(lines):
    param_dict = {}
    for line in lines:
        if line.startswith("base_log_name"):
            break
        ps = line.strip().split()
        param_dict[ps[0][:-1]] = " ".join(ps[1:])

    #print param_dict
    return param_dict

def get_an_entry(params, trn_df, dev_df, tst_df, fname):
    entry = []

    if "feat" in params:
        entry.append(params["feat"])
    else:
        entry.append("basic_seg")
    
    if "rep" in params:
        entry.append(params["rep"])
    else:
        entry.append("std")
    
    if params["activation"].startswith("["):
        entry.append(eval(params["activation"]))
    else:
        entry.append(params["activation"])
    entry.append(eval(params["n_hidden"]))

    if "fbmerge" in params:
        entry.append(params["fbmerge"])
    else:
        entry.append("concat")

    entry.append(eval(params["drates"]))
    entry.append(eval(params["recout"]))
    entry.append(params["opt"])
    entry.append(float(params["lr"]))
    entry.append(float(params["norm"]))
    entry.append(int(params["n_batch"]))
    entry.append(int(params["fepoch"]))

    if "in2out" in params:
        entry.append(int(params["in2out"]))
    else:
        entry.append(0)

    if "emb" in params:
        entry.append(int(params["emb"]))
    else:
        entry.append(0)

    if "lang" in params:
        entry.append(params["lang"])
    else:
        entry.append("eng")

    if "reverse" in params:
        entry.append(params["reverse"] == "True")
    else:
        entry.append(False)

    if "tagging" in params:
        entry.append(params["tagging"])
    else:
        entry.append("io")

    trn_best = trn_df.ix[trn_df["best"].idxmax()]
    dev_best = dev_df.ix[dev_df["best"].idxmax()]

    entry.append(trn_best['cerr'])
    entry.append(trn_best['werr'])
    entry.append(trn_best['f1'])
    entry.append(dev_best['cerr'])
    entry.append(dev_best['werr'])
    entry.append(dev_best['f1'])

    if tst_df is None:
        entry.append(100)
        entry.append(100)
        entry.append(0)
    else:
        tst_at_dev_best = len(np.unique(dev_df["best"]))
        
        tst_best = tst_df.ix[tst_at_dev_best - 1]
    
        entry.append(tst_best['cerr'])
        entry.append(tst_best['werr'])
        entry.append(tst_best['f1'])

    entry.append(trn_best['mcost'])
    entry.append(dev_best['best_epoch'])
    entry.append(trn_best['mtime'])
    entry.append(trn_df["epoch"].max())
    entry.append(fname)

    if "shuf" in params:
        entry.append(int(params["shuf"]))
    else:
        entry.append(0)

    entry.append(trn_df)
    entry.append(dev_df)
    entry.append(tst_df)

    return entry

def collect():
    cols = ["feat", "rep", "activation", "n_hidden", "fbmerge", "drates",
        "recout", "opt", "lr", "norm", "n_batch", "fepoch", "in2out", "emb",
        "lang", "reverse", "tagging", "trn-cerr", "trn-werr", "trn-f1",
        "dev-cerr", "dev-werr", "dev-f1", "tst-cerr", "tst-werr", "tst-f1",
        "trn-cost", "best-epoch", "trn-time", "max-epoch", "log_fname", "shuf",
        "trn_log", "dev_log", "tst_log"]
    
    try:
        all_results = pd.read_json('all_results.json')
    except:
        all_results = pd.DataFrame(columns=cols)

    files = []
    for (_, _, fnames) in walk(LOG_DIR):
        files += fnames

    files = filter(lambda s: s.endswith("info"), files)

    i = all_results.shape[0]
    indx = 0
    for fname in files:
        indx += 1
        if fname in all_results['log_fname'].values.tolist():
            break
        
        print "Processing: {} / {}\r".format(indx, len(files)),
        sys.stdout.flush()
        with open(LOG_DIR + "/" + fname, 'r') as f:
            try:
                lines = f.readlines()

                params_dict = get_params(lines)
                #print params_dict

                trn_lines = get_lines(lines, "trn ")
                dev_lines = get_lines(lines, "dev")
                tst_lines = get_lines(lines, "tst")
            
                trn_df = get_experiment_frame(trn_lines)
                dev_df = get_experiment_frame(dev_lines)
                tst_df = None if len(tst_lines) == 0 else get_experiment_frame(tst_lines)
                
                all_results.loc[i] = get_an_entry(params_dict, trn_df, dev_df, tst_df, fname)

                i += 1
                
            except Exception as e:
                print "Error during processing file: " + fname
                print e
                traceback.print_exc()

    all_results.to_json('all_results.json')
    return all_results

def show_best_results(df):
    g = df.groupby('lang')

    print tabulate(df[g['dev-f1'].transform(max) == df['dev-f1']][['lang',
        'dev-f1', 'tst-f1']].values, headers=["lang", "dev", "tst"])
    
    print "\n**** Files ****"
    print tabulate(df[g['dev-f1'].transform(max) == df['dev-f1']][['lang',
        'log_fname']].values, headers=["lang", "fname"])

def plot_lang_best(df, lang):
    idx = df.groupby('lang')['dev-f1'].transform(max) == df['dev-f1']
    trn = df[idx][df['lang'] == lang]['trn_log'].values[0]
    dev = df[idx][df['lang'] == lang]['dev_log'].values[0]

    plt.title('{} best experiment'.format(lang))
    plt.xlabel('epoch')
    #plt.ylabel('f1')
    plt.plot(trn['epoch'], trn['f1'], label ='trn f1')
    plt.plot(trn['epoch'], dev['f1'], label = 'dev f1')
    plt.plot(trn['epoch'], (1 - trn['cerr']) * 100, label='trn char acc')
    plt.plot(trn['epoch'], (1 - dev['cerr']) * 100, label='dev char acc')
    plt.legend(bbox_to_anchor=(1.0001, 1), loc=2, borderaxespad=0.)
    plt.show()    

if __name__ == "__main__":
    parser = get_arg_parser()
    args = vars(parser.parse_args())
    df = collect()

    if args['job'] == "lang_results":
        show_best_results(df)
    elif args['job'] == "lang_best_plot":
        plot_lang_best(df, args['lang'])
        
