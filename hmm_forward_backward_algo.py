# Example implementing forward - backward (alpha - beta) algorithm for calculating latent posteriors in HMM (smoothing)
# NOTE: this is an unscaled version - prone to underflow of probabilities (particularly for long sequences) - check scaled version

import numpy as np

# example hmm
states = ['Healthy', 'Fever']
end_state = 'E' # note that an explicit end state can be hadled like a state with a ghost observation (end token) with emission prob = 1
observations = ['normal', 'cold', 'dizzy']
start_probabilities = {'Healthy':0.6, 'Fever':0.4}
transition_probabilities = {'Healthy' : {'Healthy':0.69, 'Fever':0.3, 'E':0.01},
                            'Fever': {'Healthy':0.4, 'Fever':0.59, 'E':0.01}}
emission_probabilities = {'Healthy' : {'normal':0.5, 'cold':0.4, 'dizzy':0.1},
                            'Fever': {'normal':0.1, 'cold':0.3, 'dizzy':0.6}}

# forward-backward algo (scaled)
def fwd_bkwd(observations, states, start_probabilities, transition_probabilities, emission_probabilities, end_state):
    N = len(observations)
    S = len(states)
    alpha = np.zeros((N, S))
    beta = np.zeros((N, S))

    ## forward pass

    # alpha for first observation
    curr_obs = observations[0]
    for j, curr_state in enumerate(states):
        alpha[0][j] = start_probabilities[curr_state] * emission_probabilities[curr_state][curr_obs]

    # iterate for subsequent observations
    for i in range(1, N):
        curr_obs = observations[i]
        for j, curr_state in enumerate(states):
            for k, prev_state in enumerate(states):
                alpha[i][j] += alpha[i-1][k] * transition_probabilities[prev_state][curr_state] * emission_probabilities[curr_state][curr_obs]

    # alpha_end_state - accounting for end state
    alpha_end_state = 0
    for k, prev_state in enumerate(states):
        alpha_end_state += alpha[N-1][k] * transition_probabilities[prev_state][end_state] * 1 # emission prob for end token (ghost observvation) = 1


    ## backward pass

    # init beta (accounting for the explicit end state)
    beta_end_state = 1
    for j, curr_state in enumerate(states):
        beta[N-1][j] = beta_end_state * transition_probabilities[curr_state][end_state]

    # iterate to calculate rest of the beta values
    for i in range(N-2, -1, -1):
        next_obs = observations[i+1]
        for j, curr_state in enumerate(states):
            for k, next_state in enumerate(states):
                beta[i][j] += beta[i+1][k] * transition_probabilities[curr_state][next_state] * emission_probabilities[next_state][next_obs]


    ## calculate posterior probabilities

    p_z_given_X = np.zeros((N,S)) # marginal posterior, i.e., p_z_given_X[n][j] = p(zn = state[j] | X)
    p_zn_znplus1_given_X = np.zeros((N-1, S, S)) # joint posterior, i.e., p_zn_znplus1_given_X[n][j][k] = p(zn = state[j],zn+1 = state[k] | X)

    # p_X
    p_X = sum(alpha[N-1] * beta[N-1])
    # also p_X
    p_X = alpha_end_state

    # iteratively calculate marginal postriors
    for i in range(N):
        # p_z_given_X when using un-scalled values of alpha and beta
        p_z_given_X[i] = (alpha[i] * beta[i]) / p_X

    # iteratively calculate joint posteriors
    for i in range(0, N-1):
        next_obs = observations[i+1]
        for j, curr_state in enumerate(states):
            for k, next_state in enumerate(states):
                p_zn_znplus1_given_X[i][j][k] = alpha[i][j] * beta[i+1][k] * transition_probabilities[curr_state][next_state] * emission_probabilities[next_state][next_obs]
        # normalization factor when using un-scaled values
        p_zn_znplus1_given_X[i] /= p_X

    return p_X, alpha, beta, p_z_given_X, p_zn_znplus1_given_X


# main

p_X, alpha, beta, p_z_given_X, p_zn_znplus1_given_X = fwd_bkwd(observations, states, start_probabilities, transition_probabilities, emission_probabilities, end_state)

print('------- p_X -------')
print(p_X)
print('------ alphas -----')
print(alpha.T)
print('------ betas ------')
print(beta.T)
print('------ p_z_given_X -----')
print(p_z_given_X.T)
print('------ p_zn_znplus1_given_X -----')
print(p_zn_znplus1_given_X)
