# Program implementing policy evaluation from given finite horizon mdp using modified backward pass - equivalent to bellmann equation

# In this example, we use the given finite horizon mdp (true) parameters instead of learning them from data
# p(s_1) is a categorical distribution where s_1 can take one of k states (thus parameterized by k X 1 parameters )
# p(a_n|s_n) is a categorical distribution where a_n can take one of A values (thus parameterized by k X A-1 parameters)
# p(s_n|s_n-1, a_n-1) is a categorical distribution where s_n can take one of k states conditioned upon the state taken by s_n-1 and the action value a_n-1 (thus parameterized by k X A X k-1 parameters)
# p(r_n|s_n, a_n) is a gaussian distribution where r_n can take a real number value conditioned upon the state taken by s_n and the action a_n-1 (thus parameterized by k X A X 2 params for mus and sigmas)

# TODO: optimize the implementation using vectorized implementation

# NOTE 1: this implementation assumes that all seqeunces (trajectories) from the mdp are of same length.
# Check the finite horizon version with an explicit end state, where we remove this assumption by adding an explicit terminal_state in the mdp

# NOTE 2: so in the version with no explicit terminal state (this version), the expected return has no upper bound (increases with increase in the seq_len). While in the version with an explicit terninal state, the expected_return gets naturtally upper bounded (no matter how big the value of seq_len) due to the nature of transition probabilities.

import numpy as np
import torch
import torch.distributions as tdist
import torch.nn.functional as F
from tqdm import tqdm


# function implementing policy evaluation as a modified backward message passing over the finite horizon mdp that takes expectation of the reward at each factor node
# this is equivalent to bellmann equation for policy evaluation
def policy_eval(n_states, n_actions, seq_len, df, mdp):
    N = seq_len

    starting_state_probs, action_probs, transition_probs, reward_mus, reward_sigmas = mdp

    # dp container for state values
    state_values = torch.zeros((N, n_states))

    # value of end state s_N-1
    for k in range(n_states):
        for a in range(n_actions):
            state_values[N-1][k] += action_probs[k][a] * reward_mus[k][a]

    # calculate state values for states s_N-2 to s_0
    # backward message passing - or bellman backup equation
    for n in range(N-2, -1, -1):
        for k in range(n_states):
            for a in range(n_actions):
                for k_next in range(n_states):
                    state_values[n][k] += action_probs[k][a] * ( reward_mus[k][a] + df * transition_probs[k][a][k_next] * state_values[n+1][k_next] )

    # expected return of policy over given mdp
    expected_return = 0
    for k in range(n_states):
        expected_return += starting_state_probs[k] * state_values[0][k]

    # print(state_values.int())

    return expected_return




# main
def main(seed):
    # hyperparams
    n_states = 2 # K
    n_actions = 2 # A
    seq_len = 100000 # N - equal to time horizon
    df = 1
    random_seed = seed

    # set random seed
    np.random.seed(random_seed)
    torch.manual_seed(random_seed)

    # true parameters to be estimated
    starting_state_probs_true = [0.6, 0.4] # shape = (1,K)
    action_probs_true = [[0.8, 0.2], [0.6, 0.4]] # aka policy; shape = (K, A)
    transition_probs_true = np.zeros((n_states, n_actions, n_states)) # shape = (K, A, K)
    transition_probs_true[0][0] = [.1, .9]
    transition_probs_true[0][1] = [.2, .8]
    transition_probs_true[1][0] = [.7, .3]
    transition_probs_true[1][1] = [.6, .4]
    reward_mus = [[8., 4.], [2., 10.]] # shape = (K, A)
    reward_sigmas = [[.5, .5], [1., 1.]] # shape = (K, A)

    mdp = [starting_state_probs_true, action_probs_true, transition_probs_true, reward_mus, reward_sigmas]

    # evaluate given policy over given finite horizon mdp
    expected_return = policy_eval(n_states, n_actions, seq_len, df, mdp)

    print('expected_return: ', expected_return.data.numpy())


if __name__ == '__main__':
    # random_seeds = [0,1,13,42,69,420,2048]
    random_seeds = [13]
    for s in random_seeds:
        main(s)
