import lasagne, theano, numpy as np, logging
from theano import tensor as T

class Identity(lasagne.init.Initializer):

    def sample(self, shape):
        return lasagne.utils.floatX(np.eye(*shape))

class RDNN_Dummy:
    def __init__(self, nc, nf, kwargs):
        self.nc = nc

    def train(self, dsetdat):
        return self.predict(dsetdat)

    def predict(self, dsetdat):
        ecost, rnn_last_predictions = 0, []
        for Xdset, Xdsetmsk, ydset, ydsetmsk in zip(*dsetdat):
            ecost += 0
            sentLens = Xdsetmsk.sum(axis=-1)
            for i, slen in enumerate(sentLens):
                rnn_last_predictions.append(np.random.random_integers(0,self.nc-1,slen))
        return ecost, rnn_last_predictions

class RDNN:
    def __init__(self, nc, nf, kwargs):
        assert nf; assert nc
        LayerType = lasagne.layers.RecurrentLayer # default layer type
        if kwargs['ltype'] == 'recurrent':
            LayerType = lasagne.layers.RecurrentLayer
        elif kwargs['ltype'] == 'lstm':
            LayerType = lasagne.layers.LSTMLayer
        else:
            raise Exception()
        nonlin = getattr(lasagne.nonlinearities, kwargs['activation'][0]) # change when deep
        optim = getattr(lasagne.updates, kwargs['opt'])
        n_hidden = kwargs['n_hidden']
        grad_clip =  kwargs['grad_clip'] if kwargs['grad_clip'] > 0 else False
        lr = kwargs['lr']
        norm = kwargs['norm']
        ldepth = len(kwargs['n_hidden'])
        recout = kwargs['recout']

        # network
        l_in = lasagne.layers.InputLayer(shape=(None, None, nf))
        logging.debug('l_in: {}'.format(lasagne.layers.get_output_shape(l_in)))
        N_BATCH_VAR, MAX_SEQ_LEN_VAR, _ = l_in.input_var.shape # symbolic ref to input_var shape
        l_mask = lasagne.layers.InputLayer(shape=(N_BATCH_VAR, MAX_SEQ_LEN_VAR))
        logging.debug('l_mask: {}'.format(lasagne.layers.get_output_shape(l_mask)))

        l_forward = LayerType(l_in, n_hidden[0], mask_input=l_mask, grad_clipping=grad_clip, W_hid_to_hid=Identity(),
                W_in_to_hid=lasagne.init.GlorotUniform(gain='relu'), nonlinearity=nonlin)
        logging.debug('l_forward: {}'.format(lasagne.layers.get_output_shape(l_forward)))
        l_backward = LayerType(l_in, n_hidden[0], mask_input=l_mask, grad_clipping=grad_clip, W_hid_to_hid=Identity(),
                W_in_to_hid=lasagne.init.GlorotUniform(gain='relu'), nonlinearity=nonlin, backwards=True)
        logging.debug('l_backward: {}'.format(lasagne.layers.get_output_shape(l_backward)))
        l_concat = lasagne.layers.ConcatLayer([l_forward, l_backward], axis=2)
        logging.debug('l_concat: {}'.format(lasagne.layers.get_output_shape(l_concat)))

        if recout:
            logging.info('using recout.')
            l_out = lasagne.layers.RecurrentLayer(l_concat, num_units=nc, mask_input=l_mask, W_hid_to_hid=Identity(),
                    W_in_to_hid=lasagne.init.GlorotUniform(), nonlinearity=lasagne.nonlinearities.softmax)
            logging.debug('l_out: {}'.format(lasagne.layers.get_output_shape(l_out)))
        else:
            l_reshape = lasagne.layers.ReshapeLayer(l_concat, (-1, n_hidden[-1]*2))
            logging.debug('l_reshape: {}'.format(lasagne.layers.get_output_shape(l_reshape)))
            l_rec_out = lasagne.layers.DenseLayer(l_reshape, num_units=nc, nonlinearity=lasagne.nonlinearities.softmax)

            logging.debug('l_rec_out: {}'.format(lasagne.layers.get_output_shape(l_rec_out)))
            l_out = lasagne.layers.ReshapeLayer(l_rec_out, (N_BATCH_VAR, MAX_SEQ_LEN_VAR, nc))
            logging.debug('l_out: {}'.format(lasagne.layers.get_output_shape(l_out)))

        self.output_layer = l_out

        target_output = T.tensor3('target_output')
        out_mask = T.tensor3('mask')

        def cost(output):
            return -T.sum(out_mask*target_output*T.log(output))/T.sum(out_mask)

        cost_train = cost(lasagne.layers.get_output(l_out, deterministic=False))
        cost_eval = cost(lasagne.layers.get_output(l_out, deterministic=True))

        # cost_train = T.switch(T.or_(T.isnan(cost_train), T.isinf(cost_train)), 1000, cost_train)

        all_params = lasagne.layers.get_all_params(l_out, trainable=True)
        logging.debug(all_params)

        f_hid2hid = l_forward.get_params()[-1]
        b_hid2hid = l_backward.get_params()[-1]

        all_grads = T.grad(cost_train, all_params)

        all_grads, total_norm = lasagne.updates.total_norm_constraint(all_grads, norm, return_norm=True)
        all_grads = [T.switch(T.or_(T.isnan(total_norm), T.isinf(total_norm)), p*0.1 , g) for g,p in zip(all_grads, all_params)]

        updates = optim(all_grads, all_params, lr)

        logging.info("Compiling functions...")
        self.train_model = theano.function(inputs=[l_in.input_var, target_output, l_mask.input_var, out_mask], outputs=cost_train, updates=updates)
        self.predict_model = theano.function(
                inputs=[l_in.input_var, target_output, l_mask.input_var, out_mask],
                outputs=[cost_eval, lasagne.layers.get_output(l_out, deterministic=True)])

        # aux
        self.train_model_debug = theano.function(
                inputs=[l_in.input_var, target_output, l_mask.input_var, out_mask],
                outputs=[cost_train]+lasagne.layers.get_output([l_out, l_concat], deterministic=True)+[f_hid2hid, b_hid2hid, total_norm],
                updates=updates)
        self.compute_cost = theano.function([l_in.input_var, target_output, l_mask.input_var, out_mask], cost_eval)
        self.compute_cost_train = theano.function([l_in.input_var, target_output, l_mask.input_var, out_mask], cost_train)
        logging.info("Compiling done.")

    def train(self, dsetdat):
        tcost = sum(self.train_model(Xdset, ydset, Xdsetmsk, ydsetmsk) for Xdset, Xdsetmsk, ydset, ydsetmsk in zip(*dsetdat))
        pcost, pred = self.predict(dsetdat)
        return tcost, pred

    def predict(self, dsetdat):
        ecost, rnn_last_predictions = 0, []
        for Xdset, Xdsetmsk, ydset, ydsetmsk in zip(*dsetdat):
            bcost, pred = self.predict_model(Xdset, ydset, Xdsetmsk, ydsetmsk)
            ecost += bcost
            predictions = np.argmax(pred*ydsetmsk, axis=-1).flatten()
            sentLens, mlen = Xdsetmsk.sum(axis=-1), Xdset.shape[1]
            for i, slen in enumerate(sentLens):
                rnn_last_predictions.append(predictions[i*mlen:i*mlen+slen])
        return ecost, rnn_last_predictions

    def sing(self, dsetdat, mode):
        ecost, rnn_last_predictions = 0, []
        for Xdset, Xdsetmsk, ydset, ydsetmsk in zip(*dsetdat):
            if mode == 'train':
                bcost, pred, l_sum_out, f_hid2hid, b_hid2hid, total_norm = self.train_model(Xdset, ydset, Xdsetmsk, ydsetmsk)
                logging.debug('cost: {}'.format(bcost))
                logging.debug('lconcat mean {} max {} min {}'.format(np.mean(l_sum_out), np.max(l_sum_out), np.min(l_sum_out)))
                logging.debug('forward hid2hid mean {} max {} min {}'.format(np.mean(f_hid2hid), np.max(f_hid2hid), np.min(f_hid2hid)))
                logging.debug('backwar hid2hid mean {} max {} min {}'.format(np.mean(b_hid2hid), np.max(b_hid2hid), np.min(b_hid2hid)))
                logging.debug('total_norm: {}'.format(total_norm))
                # print 'mean {} max {} min {}'.format(np.mean(grads), np.max(grads), np.min(grads))
            else:
                bcost, pred = getattr(self, mode+'_model')(Xdset, ydset, Xdsetmsk, ydsetmsk)
            ecost += bcost
            predictions = np.argmax(pred*ydsetmsk, axis=-1).flatten()
            sentLens, mlen = Xdsetmsk.sum(axis=-1), Xdset.shape[1]
            for i, slen in enumerate(sentLens):
                rnn_last_predictions.append(predictions[i*mlen:i*mlen+slen])
        return ecost, rnn_last_predictions

"""
self.layers = [l_in]
for level in range(1,ldepth+1):
    prev_layer = self.layers[level-1]
    l_forward = LayerType(prev_layer, n_hidden[level-1], mask_input=l_mask, grad_clipping=grad_clip, W_hid_to_hid=Identity(), b=lasagne.init.Constant(0.001),  nonlinearity=nonlin)
    logging.info('l_forward: {}'.format(lasagne.layers.get_output_shape(l_forward)))
    l_backward = LayerType(prev_layer, n_hidden[level-1], mask_input=l_mask, grad_clipping=grad_clip, W_hid_to_hid=Identity(), b=lasagne.init.Constant(0.001), nonlinearity=nonlin, backwards=True)
    logging.info('l_backward: {}'.format(lasagne.layers.get_output_shape(l_backward)))
    l_concat = lasagne.layers.ConcatLayer([l_forward, l_backward], axis=2)
    logging.info('l_concat: {}'.format(lasagne.layers.get_output_shape(l_concat)))
    self.layers.append(l_concat)
l_reshape = lasagne.layers.ReshapeLayer(self.layers[-1], (-1, n_hidden[-1]*2))
logging.info('l_reshape: {}'.format(lasagne.layers.get_output_shape(l_reshape)))
"""
