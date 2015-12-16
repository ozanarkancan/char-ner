import numpy as np
from tabulate import tabulate

def viterbi(initial_dist, emission_dist, transition_dist,  emissions):
    num_states = transition_dist.shape[0]
    probs = emission_dist[:,emissions[0]] * initial_dist
    stack = []

    for emission in emissions[1:]:
        trans_probs = transition_dist * np.row_stack(probs)
        max_col_ixs = np.argmax(trans_probs, axis=0)
        probs = emission_dist[:,emission] * trans_probs[max_col_ixs, np.arange(num_states)]

        stack.append(max_col_ixs)

    state_seq = [np.argmax(probs)]

    while stack:
        max_col_ixs = stack.pop()
        state_seq.append(max_col_ixs[state_seq[-1]])

    state_seq.reverse()

    return state_seq

def viterbi_log(emission_dist, transition_dist, emissions):
    num_states = transition_dist.shape[0]
    initial_dist = np.log(np.array([1.0/num_states for i in range(num_states)]))
    probs = emission_dist[:,emissions[0]] + initial_dist
    stack = []

    for emission in emissions[1:]:
        trans_probs = transition_dist + np.row_stack(probs)
        max_col_ixs = np.argmax(trans_probs, axis=0)
        probs = emission_dist[:,emission] + trans_probs[max_col_ixs, np.arange(num_states)]

        stack.append(max_col_ixs)

    state_seq = [np.argmax(probs)]

    while stack:
        max_col_ixs = stack.pop()
        state_seq.append(max_col_ixs[state_seq[-1]])

    state_seq.reverse()

    return state_seq

def viterbi2(emission_dist, transition_dist, emissions, wmat):
    num_states = transition_dist[0].shape[0]
    initial_dist = np.array([1.0/num_states for i in range(num_states)])
    probs = emission_dist[:,emissions[0]] * initial_dist
    stack = []

    for emission,matnumber in zip(emissions[1:],wmat[1:]):
        print emission, matnumber
        trans_probs = transition_dist[matnumber] * np.row_stack(probs)
        max_col_ixs = np.argmax(trans_probs, axis=0)
        probs = emission_dist[:,emission] * trans_probs[max_col_ixs, np.arange(num_states)]

        stack.append(max_col_ixs)

    state_seq = [np.argmax(probs)]

    while stack:
        max_col_ixs = stack.pop()
        state_seq.append(max_col_ixs[state_seq[-1]])

    state_seq.reverse()

    return state_seq

def viterbi_log_multi(emission_dist, transition_dist, emissions, wmat):
    num_states = transition_dist[0].shape[0]
    initial_dist = np.log(np.array([1.0/num_states for i in range(num_states)]))
    probs = emission_dist[:,emissions[0]] + initial_dist
    stack = []

    for emission,matnumber in zip(emissions[1:],wmat[1:]):
        # print emission, matnumber
        trans_probs = transition_dist[matnumber] + np.row_stack(probs)
        max_col_ixs = np.argmax(trans_probs, axis=0)
        probs = emission_dist[:,emission] + trans_probs[max_col_ixs, np.arange(num_states)]

        stack.append(max_col_ixs)

    state_seq = [np.argmax(probs)]

    while stack:
        max_col_ixs = stack.pop()
        state_seq.append(max_col_ixs[state_seq[-1]])

    state_seq.reverse()

    return state_seq

if __name__ == '__main__':
    transtypes, ntags = 3,3
    transition_probs_tensor = np.zeros((transtypes,ntags,ntags))
    transition_probs_tensor[0] = np.eye(ntags) # 0=per, 1=loc, 2=o in_word
    transition_probs_tensor[1] = np.array([[1,0,1],[0,1,1],[0,0,1]]) # from_word
    transition_probs_tensor[2] = np.array([[1,0,0],[0,1,0],[1,1,1]])  # to_word
    """
    transition_probs_tensor[0] = np.array([[1,1e-10,1e-10],[1e-10,1,1e-10],[1e-10,1e-10,1]]) # np.eye(3) 0=per, 1=loc, 2=o
    transition_probs_tensor[1] = np.array([[1,1e-10,1],[1e-10,1,1],[1e-10,1e-10,1]]).T
    """
    print transition_probs_tensor[0]
    print transition_probs_tensor[1]
    sent_str = 'abc de'
    wiseq = [0,0,0,-1,0,0]
    wmat = [0,  0,  0,  1,   2,  0]
    # wmat = map(lambda x:int(x<0),wiseq)
    # wmat.pop(0); wmat += [0]
    emissions = range(len(sent_str))
    print 'emissions:',emissions
    print 'wmat:',wmat
    print tabulate([sent_str,wiseq,emissions,wmat])
    # emissions = [2, 1, 0]
    emission_probs = np.array([
        [0.5, 0.1, 0.4],
        [0.6, 0.3, 0.1],
        [0.7, 0.1, 0.2],
        [0.1, 0.1, 0.8],
        [0.2, 0.7, 0.1],
        [0.2, 0.7, 0.1],
        ]) # 0=per, 1=loc, 2=o
    print tabulate(emission_probs.T)

    # print viterbi(initial_dist, emission_probs, transition_probs, emissions)
    # print viterbi_log(np.log(emission_probs.T), np.log(transition_probs_tensor[0]), emissions)
    print viterbi_log_multi(np.log(emission_probs.T), np.log(transition_probs_tensor), emissions, wmat)
    print viterbi2(emission_probs.T, transition_probs_tensor, emissions, wmat)

