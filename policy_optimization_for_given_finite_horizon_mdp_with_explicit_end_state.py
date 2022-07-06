# Program implementing policy optimization from given finite horizon mdp (with explicit end state) using modified backward pass - equivalent to bellmann equation

# In this example, we use the given finite horizon mdp (true) parameters instead of learning them from data
# p(s_1) is a categorical distribution where s_1 can take one of k states (thus parameterized by k X 1 parameters )
# p(a_n|s_n) is a categorical distribution where a_n can take one of A values (thus parameterized by k X A-1 parameters)
# p(s_n|s_n-1, a_n-1) is a categorical distribution where s_n can take one of k states conditioned upon the state taken by s_n-1 and the action value a_n-1 (thus parameterized by k X A X k-1 parameters)
# p(r_n|s_n, a_n) is a gaussian distribution where r_n can take a real number value conditioned upon the state taken by s_n and the action a_n-1 (thus parameterized by k X A X 2 params for mus and sigmas)

# TODO: optimize the implementation using vectorized implementation

# NOTE 1: the way we handle a terminal state is that:
# 1.We add the explicit terminal state, modifying the transition probabilities accordingly. We must ensure that the reward for taking actions from the terminal state is zero (since we cant take actions from the terminal state / we always loop into the terminal state irrespective of the action taken)
# 2. we decide a cap for the episode length (seq_len). Though this assumes that there are trajectories that go on for that long, the policy evaluation and optimization step will discard unnecessary long sequences, since in the bellmann equation, the repeated multiplication of transition probabilities will diminish the likelihood of very long episodes. Also, in policy optimization, the learnt policy will prefer a shorter horizon than the cap decided by us, if that is the optimal thing to do.

# NOTE 2: so in the version with no explicit terminal state, the expected return has no upper bound (increases with increase in the seq_len). While in the version with an explicit terninal state (this version), the expected_return gets naturtally upper bounded (no matter how big the value of seq_len) due to the nature of transition probabilities.

import numpy as np
import torch
import torch.distributions as tdist
import torch.nn.functional as F
from tqdm import tqdm


# function implementing policy evaluation as a modified backward message passing over the finite horizon mdp that takes expectation of the reward at each factor node
# this is equivalent to bellmann equation for policy evaluation
def policy_eval(n_states, n_actions, seq_len, df, mdp, policy):
    N = seq_len
    action_probs = policy

    starting_state_probs, transition_probs, reward_mus, reward_sigmas = mdp

    # dp container for state values
    state_values = torch.zeros((N, n_states))

    # value of state s_N-1 - is always bootstrapped from the terminal state (value of terminal state is zero).
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


# function performing policy optimization
def policy_optimization(n_states, n_actions, seq_len, df, mdp, n_iters):
    # policy parameter
    policy_scores = np.random.rand(n_states, n_actions)
    policy_scores = torch.from_numpy(policy_scores).requires_grad_()
    policy = F.softmax(policy_scores, dim=1)

    # optimizer
    params = [policy_scores]
    optimizer = torch.optim.Adam(lr=1e-1, params=params)

    for i in tqdm(range(n_iters)):
        expected_return = policy_eval(n_states, n_actions, seq_len, df, mdp, policy)
        loss = -expected_return
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        # recalculate policy from updated params
        policy = F.softmax(policy_scores, dim=1)

        # if i % 1 == 0:
        #     print('expected_return: ', expected_return.item())

    return policy, expected_return


# main
def main(seed):
    # hyperparams
    n_states = 3 # K
    n_actions = 2 # A
    seq_len = 50 # N - this is an upper bound on the horizon. Mostly the horizon will be determined by the terminal state
    df = 1
    n_iters = 50
    random_seed = seed

    # set random seed
    np.random.seed(random_seed)
    torch.manual_seed(random_seed)

    # true parameters to be estimated
    starting_state_probs_true = [0.6, 0.4, 0] # shape = (1,K)

    transition_probs_true = np.zeros((n_states, n_actions, n_states)) # shape = (K, A, K)
    transition_probs_true[0][0] = [.09, .89, 0.02]
    transition_probs_true[0][1] = [.19, .79, 0.02]
    transition_probs_true[1][0] = [.69, .29, 0.02]
    transition_probs_true[1][1] = [.59, .39, 0.02]
    transition_probs_true[2][0] = [0, 0, 1] # terminal state
    transition_probs_true[2][1] = [0, 0, 1] # terminal state
    reward_mus = [[8., 4.], [2., 10.], [0, 0]] # shape = (K, A) - zero rewards for taking actions from terminal state
    reward_sigmas = [[.5, .5], [1., 1.], [1e-5, 1e-5]] # shape = (K, A)

    mdp = [starting_state_probs_true, transition_probs_true, reward_mus, reward_sigmas]

    # find optimal policy
    optimal_policy, optimal_expected_return = policy_optimization(n_states, n_actions, seq_len, df, mdp, n_iters)

    print('-------- optimal policy : --------')
    print(optimal_policy)
    print('optimal_expected_return: ', optimal_expected_return.data.numpy())


if __name__ == '__main__':
    # random_seeds = [0,1,13,42,69,420,2048]
    random_seeds = [13]
    for s in random_seeds:
        main(s)
