# Program implementing model based policy optimization from a dataset of trajectories generated from an unknown finite horizon mdp - using modified backward pass - equivalent to bellmann equation

# In this example, we use a finite horizon mdp (true) parameters to generate a dataset of trajectories to learn from
# p(s_1) is a categorical distribution where s_1 can take one of k states (thus parameterized by k X 1 parameters )
# p(a_n|s_n) is a categorical distribution where a_n can take one of A values (thus parameterized by k X A-1 parameters)
# p(s_n|s_n-1, a_n-1) is a categorical distribution where s_n can take one of k states conditioned upon the state taken by s_n-1 and the action value a_n-1 (thus parameterized by k X A X k-1 parameters)
# p(r_n|s_n, a_n) is a gaussian distribution where r_n can take a real number value conditioned upon the state taken by s_n and the action a_n-1 (thus parameterized by k X A X 2 params for mus and sigmas)

# TODO: optimize the implementation using vectorized implementation

import numpy as np
import torch
import torch.distributions as tdist
import torch.nn.functional as F
from tqdm import tqdm

# function to learn model using mle objective of mdp
def learn_model_using_mle(dataset, n_states, n_actions, seq_len, n_iters, lr):

    # init params
    start_scores = np.random.rand(n_states)
    actions_scores = np.random.rand(n_states, n_actions)
    transition_scores = np.random.rand(n_states, n_actions, n_states)
    reward_mus = np.random.rand(n_states, n_actions)
    reward_sigmas_raw = np.random.rand(n_states, n_actions)

    # warm starting emission probs
    # emission_scores = np.array([[.5, .4, .1], [.1, .3, .6]])

    start_scores = torch.from_numpy(start_scores).requires_grad_()
    actions_scores = torch.from_numpy(actions_scores).requires_grad_()
    transition_scores = torch.from_numpy(transition_scores).requires_grad_()
    reward_mus = torch.from_numpy(reward_mus).requires_grad_()
    reward_sigmas_raw = torch.from_numpy(reward_sigmas_raw).requires_grad_()

    # optimizer
    params = list([start_scores, actions_scores, transition_scores, reward_mus, reward_sigmas_raw])
    optimizer = torch.optim.Adam(params, lr=lr)

    # start iterations
    for iter in tqdm(range(n_iters)):

        # calculate probabilities from scores
        start_probs = F.softmax(start_scores, dim=0)
        action_probs = F.softmax(actions_scores, dim=1)
        transition_probs = F.softmax(transition_scores, dim=2)
        # ensure reward_sigmas are positive
        reward_sigmas = F.relu(reward_sigmas_raw) + 1e-8

        # number of gradient steps to be made (aternatively use a stopping criteria for checking convergence)
        num_samples = 50

        for _ in range(num_samples):

            # sample a sequence (index) from dataset
            m = np.random.randint(num_samples)

            s_0, a_0, r_0 = dataset[m][0]
            objective = torch.log(start_probs[s_0]) + \
                        torch.log(action_probs[s_0][a_0]) + \
                        tdist.Normal(loc=reward_mus[s_0][a_0], scale=reward_sigmas[s_0][a_0]).log_prob(r_0)

            s_n_minus_1, a_n_minus_1 = s_0, a_0
            for n in range(1, seq_len):
                s_n, a_n, r_n = dataset[m][n]
                objective += torch.log(transition_probs[s_n_minus_1][a_n_minus_1][s_n]) + \
                             torch.log(action_probs[s_n][a_n]) + \
                             tdist.Normal(loc=reward_mus[s_n][a_n], scale=reward_sigmas[s_n][a_n]).log_prob(r_n)
                # for next datapoint
                s_n_minus_1, a_n_minus_1 = s_n, a_n

            loss = -objective
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            # recalculate probabilities from scores after gradient update
            start_probs = F.softmax(start_scores, dim=0)
            action_probs = F.softmax(actions_scores, dim=1)
            transition_probs = F.softmax(transition_scores, dim=2)
            # ensure reward_sigmas are positive
            reward_sigmas = F.relu(reward_sigmas_raw) + 1e-8

        # if iter % 10 == 0:
        #     print('iter:{} \t loss:{:3f}'.format(iter, loss.item()))

    return start_probs, action_probs, transition_probs, reward_mus, reward_sigmas


# function implementing policy evaluation as a modified backward message passing over the learnt model of the finite horizon mdp that takes expectation of the reward at each factor node
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


# function performing policy optimization over the learnt model of the finite horizon mdp
def policy_optimization(n_states, n_actions, seq_len, df, mdp, n_iters, lr):
    # policy parameter
    policy_scores = np.random.rand(n_states, n_actions)
    policy_scores = torch.from_numpy(policy_scores).requires_grad_()
    policy = F.softmax(policy_scores, dim=1)

    # optimizer
    params = [policy_scores]
    optimizer = torch.optim.Adam(lr=lr, params=params)

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
    n_seq = 200 # M - number of sequences (trajectories) in the dataset
    df = 1
    n_iters_model_learning = 20
    lr_model_learning = 1e-1
    n_iters_policy_optimization = 50
    lr_policy_optimization = 1e-1
    random_seed = seed

    # set random seed
    np.random.seed(random_seed)
    torch.manual_seed(random_seed)

    # true parameters to be estimated
    starting_state_probs_true = [0.6, 0.4, 0] # shape = (1,K)
    action_probs_true = [[0.8, 0.2], [0.6, 0.4], [0.4, 0.6]] # aka policy; shape = (K, A) - this is just an acting policy to generate trajectories
    transition_probs_true = np.zeros((n_states, n_actions, n_states)) # shape = (K, A, K)
    transition_probs_true[0][0] = [.09, .89, 0.02]
    transition_probs_true[0][1] = [.19, .79, 0.02]
    transition_probs_true[1][0] = [.69, .29, 0.02]
    transition_probs_true[1][1] = [.59, .39, 0.02]
    transition_probs_true[2][0] = [0, 0, 1] # terminal state
    transition_probs_true[2][1] = [0, 0, 1] # terminal state
    reward_mus = [[8., 4.], [2., 10.], [0, 0]] # shape = (K, A) - zero rewards for taking actions from terminal state
    reward_sigmas = [[.5, .5], [1., 1.], [1e-5, 1e-5]] # shape = (K, A)

    # generate dataset
    dataset = np.zeros((n_seq, seq_len, 3)) # each entry is a tuple (s_n, a_n, r_n)

    for m in range(n_seq):

        # starting sample
        s_1 = np.random.choice(n_states, p=starting_state_probs_true)
        a_1 = np.random.choice(n_actions, p=action_probs_true[s_1])
        r_1 = np.random.normal(loc=reward_mus[s_1][a_1], scale=reward_sigmas[s_1][a_1])
        dataset[m][0] = [s_1, a_1, r_1]

        # iterate for subsequent sampels
        s_n_minus_1 = s_1
        a_n_minus_1 = a_1
        for n in range(1, seq_len):
            s_n = np.random.choice(n_states, p=transition_probs_true[s_n_minus_1][a_n_minus_1])
            a_n = np.random.choice(n_actions, p=action_probs_true[s_n])
            r_n = np.random.normal(loc=reward_mus[s_n][a_n], scale=reward_sigmas[s_n][a_n])
            dataset[m][n] = [s_n, a_n, r_n]
            # for next data point
            s_n_minus_1 = s_n
            a_n_minus_1 = a_n

    dataset = torch.from_numpy(dataset).int()

    # learn model of the mdp from dataset of trajectories
    print('LEARNING MODEL...')
    starting_state_probs_est, action_probs_est, transition_probs_est, reward_mus_est, reward_sigmas_est = learn_model_using_mle(dataset, n_states, n_actions, seq_len, n_iters_model_learning, lr_model_learning)

    # print('----starting_state_probs_est----\n', starting_state_probs_est.data.numpy())
    # print('----action_probs_est----\n', action_probs_est.data.numpy())
    # print('----transition_probs_est----\n', transition_probs_est.data.numpy())
    # print('----reward_mus_est----\n', reward_mus_est.data.numpy())
    # print('----reward_sigmas_est----\n', reward_sigmas_est.data.numpy())

    # freeze learnt model
    starting_state_probs_est = starting_state_probs_est.detach()
    transition_probs_est = transition_probs_est.detach()
    reward_mus_est = reward_mus_est.detach()
    reward_sigmas_est = reward_sigmas_est.detach()
    mdp_model = [starting_state_probs_est, transition_probs_est, reward_mus_est, reward_sigmas_est]

    # find optimal policy
    print('CALCULATING OPTIMAL POLICY...')
    optimal_policy, optimal_expected_return = policy_optimization(n_states, n_actions, seq_len, df, mdp_model, n_iters_policy_optimization, lr_policy_optimization)

    print('-------- optimal policy : --------')
    print(optimal_policy)
    print('optimal_expected_return: ', optimal_expected_return.data.numpy())


if __name__ == '__main__':
    # random_seeds = [0,1,13,42,69,420,2048]
    random_seeds = [13]
    for s in random_seeds:
        main(s)
