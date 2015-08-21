import lasagne, theano, numpy as np
from theano import tensor as T

class RDNN:
    def __init__(self, nc, nf, max_seq_length, **kwargs):
        # batch_size=None, n_hidden=100, grad_clip=7, lr=.001):
        assert nf; assert max_seq_length
        LayerType = lasagne.layers.RecurrentLayer
        if kwargs['ltype'] == 'recurrent':
            LayerType = lasagne.layers.RecurrentLayer
        elif kwargs['ltype'] == 'lstm':
            LayerType = lasagne.layers.LSTMLayer
        else:
            raise Exception()
        nonlin = getattr(lasagne.nonlinearities, kwargs['activation'])
        optim = getattr(lasagne.updates, kwargs['opt'])
        n_hidden = kwargs['n_hidden']
        grad_clip = kwargs['grad_clip']
        lr = kwargs['lr']
        ldepth = kwargs['deep']

        # network
        l_in = lasagne.layers.InputLayer(shape=(kwargs['n_batch'], max_seq_length, nf))
        print 'l_in:', lasagne.layers.get_output_shape(l_in)
        N_BATCH_VAR, _, _ = l_in.input_var.shape # symbolic ref to input_var shape
        l_mask = lasagne.layers.InputLayer(shape=(N_BATCH_VAR, max_seq_length))
        print 'l_mask:', lasagne.layers.get_output_shape(l_mask)

        layers = [l_in]
        for level in range(1,ldepth+1):
            prev_layer = layers[level-1]
            l_forward = LayerType(prev_layer, n_hidden[level-1], mask_input=l_mask, grad_clipping=grad_clip, nonlinearity=nonlin)
            print 'l_forward:', lasagne.layers.get_output_shape(l_forward)
            l_backward = LayerType(prev_layer, n_hidden[level-1], mask_input=l_mask, grad_clipping=grad_clip, nonlinearity=nonlin, backwards=True)
            print 'l_backward:', lasagne.layers.get_output_shape(l_backward)
            l_sum = lasagne.layers.ElemwiseSumLayer([l_forward, l_backward])
            print 'l_sum:', lasagne.layers.get_output_shape(l_sum)
            layers.append(l_sum)

        l_reshape = lasagne.layers.ReshapeLayer(layers[-1], (-1, n_hidden[-1]))
        print 'l_reshape:', lasagne.layers.get_output_shape(l_reshape)
        l_rec_out = lasagne.layers.DenseLayer(l_reshape, num_units=nc, nonlinearity=lasagne.nonlinearities.softmax)
        print 'l_rec_out:', lasagne.layers.get_output_shape(l_rec_out)
        l_out = lasagne.layers.ReshapeLayer(l_rec_out, (N_BATCH_VAR, max_seq_length, nc))
        print 'l_out:', lasagne.layers.get_output_shape(l_out)
        print lasagne.layers.get_output_shape(l_out)

        target_output = T.tensor3('target_output')
        out_mask = T.tensor3('mask')

        def cost(output):
             return -T.sum(out_mask*target_output*T.log(output))/T.sum(out_mask)

        cost_train = cost(lasagne.layers.get_output(l_out, deterministic=False))
        cost_eval = cost(lasagne.layers.get_output(l_out, deterministic=True))

        all_params = lasagne.layers.get_all_params(l_out, trainable=True)
        # Compute SGD updates for training
        print("Computing updates ...")
        # updates = lasagne.updates.adagrad(cost_train, all_params, LEARNING_RATE)
        updates = optim(cost_train, all_params, lr)
        # Theano functions for training and computing cost
        print("Compiling functions ...")
        self.train_model = theano.function(
                inputs=[l_in.input_var, target_output, l_mask.input_var, out_mask],
                outputs=[cost_train, lasagne.layers.get_output(l_out, deterministic=True)], updates=updates)
        self.compute_cost = theano.function(
            [l_in.input_var, target_output, l_mask.input_var, out_mask], cost_eval)
        self.predict_model = theano.function(
                # inputs=[l_in.input_var, l_mask.input_var],
                inputs=[l_in.input_var, target_output, l_mask.input_var, out_mask],
                outputs=[cost_eval, lasagne.layers.get_output(l_out, deterministic=True)])

    def sing(self, dsetdat, mode):
        ecost, rnn_last_predictions = 0, []
        for Xdset, Xdsetmsk, ydset, ydsetmsk in zip(*dsetdat):
            bcost, pred = getattr(self, mode+'_model')(Xdset, ydset, Xdsetmsk, ydsetmsk)
            ecost += bcost
            predictions = np.argmax(pred*ydsetmsk, axis=-1).flatten()
            sentLens, mlen = Xdsetmsk.sum(axis=-1), Xdset.shape[1]
            for i, slen in enumerate(sentLens):
                rnn_last_predictions.append(predictions[i*mlen:i*mlen+slen])
        return ecost, rnn_last_predictions
