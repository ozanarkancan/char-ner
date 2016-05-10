import lasagne, theano, numpy as np, logging
from theano import tensor as T
from batch_norm import BatchNormLayer

class Identity(lasagne.init.Initializer):

    def sample(self, shape):
        return lasagne.utils.floatX(np.eye(*shape))

def log_softmax(x):
    xdev = x - x.max(1, keepdims=True)
    return xdev - T.log(T.sum(T.exp(xdev), axis=1, keepdims=True))

class LogSoftMerge(lasagne.layers.MergeLayer):

    def __init__(self, incomings):
        super(LogSoftMerge, self).__init__(incomings)

    def get_output_for(self, inputs, **kwargs):
        """
        inputs : list of Theano expressions
        Returns
        -------
        Theano expressions
            The output of this layer given the inputs to this layer.
        """

        return T.log((T.exp(inputs[0]) + T.exp(inputs[1])) * 0.5)

    def get_output_shape_for(self, input_shapes):
        return input_shapes[0]


class RDNN_Dummy:
    def __init__(self, nc, nf, kwargs):
        self.nc = nc

    def train(self, dsetdat):
        import time
        time.sleep(1)
        return 19

    def get_param_values(self):
        return []

    def set_param_values(self, values):
        pass

    def predict(self, dsetdat):
        ecost, rnn_last_predictions = 0, []
        for Xdset, Xdsetmsk, ydset, ydsetmsk in dsetdat:
            ecost += 0
            sentLens = Xdsetmsk.sum(axis=-1)

            rnn_last_predictions.append(\
                    [self.randlogprob(slen, self.nc) for i, slen in enumerate(sentLens)])
        return ecost, rnn_last_predictions

    def randlogprob(self, sent_len, nc):
        randvals = np.random.rand(sent_len, nc)
        randlogprobs = np.log(randvals / np.sum(randvals,axis=0))
        return randlogprobs

    def viterbi(self, dsetdat):
        from viterbi import viterbi_log
        ecost, rnn_last_predictions = 0, []
        for Xdset, Xdsetmsk, ydset, ydsetmsk in dsetdat:
            ecost += 0
            pred = np.log(np.random.rand(len(sentLens),mlen,self.nc))

            sentLens, mlen = Xdsetmsk.sum(axis=-1), Xdset.shape[1]
            rnn_last_predictions.append([viterbi_log(pred[i,:slen,:].T, self.tprobs, range(slen)) for i, slen in enumerate(sentLens)])
        return ecost, rnn_last_predictions

def extract_rnn_params(kwargs):
    return dict((pname,kwargs[pname]) for pname in RDNN.param_names)

class RDNN:
    param_names=['activation','n_hidden','fbmerge','drates','opt','lr','norm','gclip','truncate','recout','batch_norm','in2out','emb','fbias','gnoise','eps']

    def __init__(self, nc, nf, kwargs):
        assert nf; assert nc
        self.kwargs = extract_rnn_params(kwargs)
        for pname in RDNN.param_names:
            setattr(self, pname, kwargs[pname])
        
        self.lr = theano.shared(np.array(self.lr, dtype='float32'), allow_downcast=True)
        self.gclip = False if self.gclip == 0 else self.gclip # mysteriously, we need this line

        self.activation = [self.activation] * len(self.n_hidden)
        self.deep_ltypes = [act_str.split('-')[1] for act_str in self.activation]

        self.opt = getattr(lasagne.updates, self.opt)
        ldepth = len(self.n_hidden)

        # network
        default_gate = lambda : lasagne.layers.Gate(W_in=lasagne.init.GlorotUniform(), 
            W_hid=lasagne.init.GlorotUniform())
        
        forget_gate = lambda : lasagne.layers.Gate(W_in=lasagne.init.GlorotUniform(), 
            W_hid=lasagne.init.GlorotUniform(),
            b=lasagne.init.Constant(self.fbias))
        
        """default_gate = lambda : lasagne.layers.Gate(W_in=lasagne.init.Orthogonal(), 
            W_hid=lasagne.init.Orthogonal())
        
        forget_gate = lambda : lasagne.layers.Gate(W_in=lasagne.init.Orthogonal(), W_hid=lasagne.init.Orthogonal(),
            b=lasagne.init.Constant(self.fbias))"""

        l_in = lasagne.layers.InputLayer(shape=(None, None, nf))
        logging.debug('l_in: {}'.format(lasagne.layers.get_output_shape(l_in)))
        N_BATCH_VAR, MAX_SEQ_LEN_VAR, _ = l_in.input_var.shape # symbolic ref to input_var shape
        # l_mask = lasagne.layers.InputLayer(shape=(N_BATCH_VAR, MAX_SEQ_LEN_VAR))
        l_mask = lasagne.layers.InputLayer(shape=(None, None))
        logging.debug('l_mask: {}'.format(lasagne.layers.get_output_shape(l_mask)))

        curlayer = l_in
        if self.emb:
            l_reshape = lasagne.layers.ReshapeLayer(l_in, (-1, nf))
            logging.debug('l_reshape: {}'.format(lasagne.layers.get_output_shape(l_reshape)))
            l_emb = lasagne.layers.DenseLayer(l_reshape, num_units=self.emb, nonlinearity=None, b=None)
            logging.debug('l_emb: {}'.format(lasagne.layers.get_output_shape(l_emb)))
            l_emb = lasagne.layers.ReshapeLayer(l_emb, (N_BATCH_VAR, MAX_SEQ_LEN_VAR, self.emb))
            logging.debug('l_emb: {}'.format(lasagne.layers.get_output_shape(l_emb)))
            curlayer = l_emb

        if self.drates[0] > 0:
            l_in_drop = lasagne.layers.DropoutLayer(curlayer, p=self.drates[0])
            logging.debug('l_drop: {}'.format(lasagne.layers.get_output_shape(l_in_drop)))
            curlayer = l_in_drop
        self.layers = [curlayer]
        self.blayers = []
        for level, ltype, n_hidden in zip(range(1,ldepth+1), self.deep_ltypes, self.n_hidden):
            prev_layer = self.layers[level-1]
            if ltype in ['relu','lrelu', 'relu6', 'elu']:
                LayerType = lasagne.layers.RecurrentLayer
                if ltype == 'relu': nonlin = lasagne.nonlinearities.rectify
                elif ltype == 'lrelu': nonlin = lasagne.nonlinearities.leaky_rectify
                elif ltype == 'relu6': nonlin = lambda x: T.min(lasagne.nonlinearities.rectify(x), 6)
                elif ltype == 'elu': nonlin = lambda x: T.switch(x >= 0, x, T.exp(x) - 1)
                l_forward = LayerType(prev_layer, n_hidden, mask_input=l_mask, grad_clipping=self.gclip, gradient_steps=self.truncate,
                        W_hid_to_hid=Identity(), W_in_to_hid=lasagne.init.GlorotUniform(gain='relu'), nonlinearity=nonlin)
                l_backward = LayerType(prev_layer, n_hidden, mask_input=l_mask, grad_clipping=self.gclip, gradient_steps=self.truncate,
                        W_hid_to_hid=Identity(), W_in_to_hid=lasagne.init.GlorotUniform(gain='relu'), nonlinearity=nonlin, backwards=True)
            elif ltype == 'lstm':
                LayerType = lasagne.layers.LSTMLayer
                l_forward = LayerType(prev_layer, n_hidden, ingate=default_gate(),
                    forgetgate=forget_gate(), outgate=default_gate(), mask_input=l_mask, grad_clipping=self.gclip, gradient_steps=self.truncate)
                l_backward = LayerType(prev_layer, n_hidden, ingate=default_gate(),
                    forgetgate=forget_gate(), outgate=default_gate(), mask_input=l_mask, grad_clipping=self.gclip, gradient_steps=self.truncate, backwards=True)

            elif ltype == 'gru':
                LayerType = lasagne.layers.GRULayer
                l_forward = LayerType(prev_layer, n_hidden, mask_input=l_mask, grad_clipping=self.gclip, gradient_steps=self.truncate)
                l_backward = LayerType(prev_layer, n_hidden, mask_input=l_mask, grad_clipping=self.gclip, gradient_steps=self.truncate, backwards=True)

            logging.debug('l_forward: {}'.format(lasagne.layers.get_output_shape(l_forward)))
            logging.debug('l_backward: {}'.format(lasagne.layers.get_output_shape(l_backward)))

            if self.fbmerge == 'concat':
                l_fbmerge = lasagne.layers.ConcatLayer([l_forward, l_backward], axis=2)
            elif self.fbmerge == 'sum':
                l_fbmerge = lasagne.layers.ElemwiseSumLayer([l_forward, l_backward])
            logging.debug('l_fbmerge: {}'.format(lasagne.layers.get_output_shape(l_fbmerge)))

            if self.batch_norm:
                logging.info('using batch norm')
                l_fbmerge = BatchNormLayer(l_fbmerge, axes=(0,1))

            if self.drates[level] > 0:
                l_fbmerge = lasagne.layers.DropoutLayer(l_fbmerge, p=self.drates[level])

            self.blayers.append((l_forward, l_backward))
            self.layers.append(l_fbmerge)
        
        l_fbmerge = lasagne.layers.ConcatLayer([l_fbmerge, curlayer], axis=2) if self.in2out else l_fbmerge

        if self.recout == 1:
            logging.info('using recout:%d.'%self.recout)
            l_out = lasagne.layers.RecurrentLayer(l_fbmerge, num_units=nc, mask_input=l_mask, W_hid_to_hid=Identity(),
                    W_in_to_hid=lasagne.init.GlorotUniform(), nonlinearity=log_softmax)
                    # W_in_to_hid=lasagne.init.GlorotUniform(), nonlinearity=lasagne.nonlinearities.softmax) CHANGED
            logging.debug('l_out: {}'.format(lasagne.layers.get_output_shape(l_out)))
        elif self.recout == 2:
            logging.info('using recout:%d.'%self.recout)
            l_fout = lasagne.layers.RecurrentLayer(l_fbmerge, num_units=nc, mask_input=l_mask, W_hid_to_hid=Identity(),
                    W_in_to_hid=lasagne.init.GlorotUniform(), nonlinearity=log_softmax)
            l_bout = lasagne.layers.RecurrentLayer(l_fbmerge, num_units=nc, mask_input=l_mask, W_hid_to_hid=Identity(),
                    W_in_to_hid=lasagne.init.GlorotUniform(), nonlinearity=log_softmax, backwards=True)
            l_out = lasagne.layers.ElemwiseSumLayer([l_fout, l_bout], coeffs=0.5)
            # l_out = LogSoftMerge([l_fout, l_bout])
            logging.debug('l_out: {}'.format(lasagne.layers.get_output_shape(l_out)))
        else:
            l_reshape = lasagne.layers.ReshapeLayer(l_fbmerge, (-1, self.n_hidden[-1]*(2 if self.fbmerge=='concat' else 1)))
            logging.debug('l_reshape: {}'.format(lasagne.layers.get_output_shape(l_reshape)))
            l_rec_out = lasagne.layers.DenseLayer(l_reshape, num_units=nc, nonlinearity=log_softmax)

            logging.debug('l_rec_out: {}'.format(lasagne.layers.get_output_shape(l_rec_out)))
            l_out = lasagne.layers.ReshapeLayer(l_rec_out, (N_BATCH_VAR, MAX_SEQ_LEN_VAR, nc))
            logging.debug('l_out: {}'.format(lasagne.layers.get_output_shape(l_out)))

        self.l_soft_out = l_rec_out
        self.output_layer = l_out

        target_output = T.tensor3('target_output')
        out_mask = T.tensor3('mask')

        """
        def cost(output):
            return -T.sum(out_mask*target_output*T.log(output))/T.sum(out_mask)
        """
        def cost(output): # expects log softmax output
            return -T.sum(out_mask*target_output*output)/T.sum(out_mask)

        cost_train = cost(lasagne.layers.get_output(l_out, deterministic=False))
        cost_eval = cost(lasagne.layers.get_output(l_out, deterministic=True))


        all_params = lasagne.layers.get_all_params(l_out, trainable=True)
        logging.debug(all_params)

        f_hid2hid = l_forward.get_params()[-1]
        b_hid2hid = l_backward.get_params()[-1]
        self.recout_hid2hid = lambda : l_out.get_params() if self.recout == 0 else lambda : l_out.get_params()[-1].get_value()

        grads = T.grad(cost_train, all_params)

        all_grads, total_norm = lasagne.updates.total_norm_constraint(grads, self.norm, return_norm=True)
        #all_grads.append(grads[-2])
        #all_grads.append(grads[-1])
        all_grads = [T.switch(T.or_(T.isnan(total_norm), T.isinf(total_norm)), p*0.01 , g) for g,p in zip(all_grads, all_params)]
        
        if self.gnoise:
            from theano.tensor.shared_randomstreams import RandomStreams
            srng = RandomStreams(seed=1234)
            e_prev = theano.shared(lasagne.utils.floatX(0.))
            nu = 0.01
            gamma = 0.55
            gs = [g + srng.normal(T.shape(g), std=(nu / ((1 + e_prev)**gamma))) for g in all_grads]
            updates = self.opt(gs, all_params, self.lr, self.eps)
            updates[e_prev] = e_prev + 1
        else:
            updates = self.opt(all_grads, all_params, self.lr, self.eps)
        

        logging.info("Compiling functions...")
        self.train_model = theano.function(inputs=[l_in.input_var, target_output, l_mask.input_var, out_mask], outputs=cost_train, updates=updates, allow_input_downcast=True)
        self.predict_model = theano.function(
                inputs=[l_in.input_var, target_output, l_mask.input_var, out_mask],
                outputs=[cost_eval, lasagne.layers.get_output(l_out, deterministic=True)])

        # aux
        self.train_model_debug = theano.function(
                inputs=[l_in.input_var, target_output, l_mask.input_var, out_mask],
                outputs=[cost_train]+lasagne.layers.get_output([l_out, l_fbmerge], deterministic=True)+[total_norm],
                updates=updates)
        self.compute_cost = theano.function([l_in.input_var, target_output, l_mask.input_var, out_mask], cost_eval)
        self.compute_cost_train = theano.function([l_in.input_var, target_output, l_mask.input_var, out_mask], cost_train)
        # self.info_model = theano.function([],recout_hid2hid)
        logging.info("Compiling done.")

    def train(self, dsetdat):
        tcost = np.mean([self.train_model(Xdset, ydset, Xdsetmsk, ydsetmsk) for Xdset, Xdsetmsk, ydset, ydsetmsk in dsetdat])
        # pcost, pred = self.predict(dsetdat)
        return tcost


    def predict(self, dsetdat):
        bcosts, rnn_last_predictions = [], []
        for Xdset, Xdsetmsk, ydset, ydsetmsk in dsetdat:
            bcost, pred = self.predict_model(Xdset, ydset, Xdsetmsk, ydsetmsk)
            bcosts.append(bcost)
            # predictions = np.argmax(pred*ydsetmsk, axis=-1).flatten()
            sentLens, mlen = Xdsetmsk.sum(axis=-1), Xdset.shape[1]
            rnn_last_predictions.append([pred[i,0:slen,:] for i, slen in enumerate(sentLens)])
        return np.mean(bcosts), rnn_last_predictions

    def get_param_values(self):
        return lasagne.layers.get_all_param_values(self.output_layer)

    def get_all_layers(self):
        return lasagne.layers.get_all_layers(self.output_layer)

    def set_param_values(self, values):
        lasagne.layers.set_all_param_values(self.output_layer, values)

if __name__ == '__main__':
    import exper
    parser = exper.get_arg_parser()
    args = vars(parser.parse_args())
    rdnn = RDNN(5, 90, args)
    print rdnn.blayers
    print rdnn.get_all_layers()

