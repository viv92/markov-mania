# Example implementing Viterbi (max-sum) algorithm for finding the most likely sequence of latent states, given the sequence of observations

import numpy as np

# example hmm
states = ['Healthy', 'Fever']
observations = ['normal', 'cold', 'dizzy']
start_probabilities = {'Healthy':0.6, 'Fever':0.4}
transition_probabilities = {'Healthy' : {'Healthy':0.7, 'Fever':0.3},
                            'Fever': {'Healthy':0.4, 'Fever':0.6}}
emission_probabilities = {'Healthy' : {'normal':0.5, 'cold':0.4, 'dizzy':0.1},
                            'Fever': {'normal':0.1, 'cold':0.3, 'dizzy':0.6}}


def viterbi(states, observations, start_probabilities, transition_probabilities, emission_probabilities):
    N = len(observations)
    S = len(states)
    most_likely_state_seq = []
    prev = np.zeros((N-1, S)).astype('int') # back pointers
    alpha = np.zeros((N, S)) # dp estimate of max log likelihood

    # alpha[0]
    curr_obs = observations[0]
    for j, state in enumerate(states):
        alpha[0][j] = np.log(start_probabilities[state]) + np.log(emission_probabilities[state][curr_obs])

    # iterate
    for i in range(1, N):
        curr_obs = observations[i]
        for j, curr_state in enumerate(states):
            candidates = np.zeros(S)
            for k, prev_state in enumerate(states):
                candidates[k] = alpha[i-1][k] + np.log(transition_probabilities[prev_state][curr_state]) + np.log(emission_probabilities[curr_state][curr_obs])
            index = np.argmax(candidates)
            prev[i-1][j] = index
            alpha[i][j] = candidates[index]


    # final max
    z_n_index = np.argmax(alpha[N-1])
    max_log_likelihood = alpha[N-1][z_n_index]

    # backtrack to get the most likely sequence of latent states
    most_likely_state_seq = [states[z_n_index]] + most_likely_state_seq
    for i in range(N-2, -1, -1):
        z_n_index = prev[i][z_n_index]
        most_likely_state_seq = [states[z_n_index]] + most_likely_state_seq

    return most_likely_state_seq, max_log_likelihood


# main
most_likely_state_seq, max_log_likelihood = viterbi(states, observations, start_probabilities, transition_probabilities, emission_probabilities)
max_likelihood = np.exp(max_log_likelihood)
print('max_likelihood: {:.5f}'.format(max_likelihood))
print('most_likely_state_seq: ', most_likely_state_seq)
