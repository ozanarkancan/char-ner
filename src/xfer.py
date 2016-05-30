import numpy as np
import argparse
import lasagne
import theano
import logging

from lazrnn import RDNN
import featchar
from exper import Batcher, Reporter, Validator, Dset
import utils

"""
[W, # for emb layer
W_in_to_ingate, # forward lstm start
W_hid_to_ingate,
b_ingate,
W_in_to_forgetgate,
W_hid_to_forgetgate,
b_forgetgate,
W_in_to_cell,
W_hid_to_cell,
b_cell,
W_in_to_outgate,
W_hid_to_outgate,
b_outgate,
W_cell_to_ingate,
W_cell_to_forgetgate,
W_cell_to_outgate,
cell_init,
hid_init, # forward lstm end
W_in_to_ingate, # backward lstm start
W_hid_to_ingate,
b_ingate,
W_in_to_forgetgate,
W_hid_to_forgetgate,
b_forgetgate,
W_in_to_cell,
W_hid_to_cell,
b_cell,
W_in_to_outgate,
W_hid_to_outgate,
b_outgate,
W_cell_to_ingate,
W_cell_to_forgetgate,
W_cell_to_outgate,
cell_init,
hid_init, # backward lstm end
W, # softmax
b] # softmax
"""

def get_args():
    parser = argparse.ArgumentParser(prog="xfer")
    parser.add_argument('model_file')
    parser.add_argument('lang')
    parser.add_argument('log')
    parser.add_argument("--breaktrn", default=0, type=int, help="break trn sents to subsents")
    parser.add_argument("--captrn", default=500, type=int, help="consider sents lt this as trn")
    parser.add_argument("--sorted", default=1, type=int, help="sort datasets before training and prediction")
    parser.add_argument("--shuf", default=1, type=int, help="shuffle the batches.")
    parser.add_argument("--tagging", default='bio', choices=['io','bio'], help="tag scheme to use")
    parser.add_argument("--sample", default=0, type=int, help="num of sents to sample from trn in the order of K")
    parser.add_argument("--rep", default='std', choices=['std','nospace','spec'], help="which representation to use")
    parser.add_argument("--fepoch", default=600, type=int, help="number of epochs")
    parser.add_argument("--save", default='', help="save param values to file")
    args = vars(parser.parse_args())
    return args

def setup_logger(args):
    import socket
    host = socket.gethostname().split('.')[0]
    logger = logging.getLogger()
    logger.setLevel(logging.DEBUG)
    shandler = logging.StreamHandler()
    shandler.setLevel(logging.INFO)
    # param_log_name = ','.join(['{}:{}'.format(p,args[p]) for p in LPARAMS])
    # param_log_name = valid_file_name(param_log_name)
    base_log_name = '{}:{},{}'.format(host, theano.config.device, args['log'])
    ihandler = logging.FileHandler('{}/{}.info'.format(utils.LOG_DIR,base_log_name), mode='w')
    ihandler.setLevel(logging.INFO)
    dhandler = logging.FileHandler('{}/{}.debug'.format(utils.LOG_DIR,base_log_name), mode='w')
    dhandler.setLevel(logging.DEBUG)
    logger.addHandler(shandler);logger.addHandler(ihandler);logger.addHandler(dhandler);

def main():
    args = get_args()
    setup_logger(args)

    logging.info('loading params')
    dat = np.load(args['model_file'])
    dat_args = dat['argsd'].tolist()
    rnn_param_values = dat['rnn_param_values'].tolist()
    logging.info('params loaded')
    dat_args['fepoch'] = args['fepoch'] # TODO
    dat_args['save'] = args['save'] # TODO

    for k,v in sorted(dat_args.iteritems()):
        logging.info('{}:\t{}'.format(k,v))


    dset = Dset(**args)
    feat = featchar.Feat(dat_args['feat'])
    feat.fit(dset, xdsets=[Dset(dname) for dname in dat_args['charset']])

    batcher = Batcher(dat_args['n_batch'], feat)
    reporter = Reporter(dset, feat)

    validator = Validator(dset, batcher, reporter)

    rdnn = RDNN(feat.NC, feat.NF, dat_args)

    params = lasagne.layers.get_all_params(rdnn.layers[-1])
    lasagne.layers.set_all_param_values(rdnn.layers[-1], rnn_param_values[:len(params)])

    validator.validate(rdnn, dat_args)

if __name__ == '__main__':
    main()
