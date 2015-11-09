import numpy as np

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

if __name__ == '__main__':
    transition_probs = np.array([[0.7, 0.4], [0.3, 0.6]]) #0=Healthy, 1=Fever
    emissions = [2, 1, 0]
    emission_probs = np.array([[0.1, 0.4, 0.5], [0.6, 0.3, 0.1]]) #0=Dizzy, 1=Cold, 2=Normal
    initial_dist = np.array([[0.6, 0.4]])

    print viterbi(initial_dist, emission_probs, transition_probs, emissions)
    print viterbi_log(np.log(emission_probs), np.log(transition_probs), emissions)

