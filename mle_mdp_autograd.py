# Program implementing maximum likelihood estimation on Markov Decision Process using autograd on MLE objective

# In this example,
# p(s_1) is a categorical distribution where s_1 can take one of k states (thus parameterized by k X 1 parameters )
# p(a_n|s_n) is a categorical distribution where a_n can take one of A values (thus parameterized by k X A-1 parameters)
# p(s_n|s_n-1, a_n-1) is a categorical distribution where s_n can take one of k states conditioned upon the state taken by s_n-1 and the action value a_n-1 (thus parameterized by k X A X k-1 parameters)
# p(r_n|s_n, a_n) is a gaussian distribution where r_n can take a real number value conditioned upon the state taken by s_n and the action a_n-1 (thus parameterized by k X A X 2 params for mus and sigmas)

# TODO: optimize the implementation using vectorized implementation (alias transition probs into 1D categorical vector and modify dataset accordingly - then use batching with tdist.Categorical)

import numpy as np
import torch
import torch.distributions as tdist
import torch.nn.functional as F
from tqdm import tqdm


# function implementing MLE
def mle(dataset, n_states, n_actions, seq_len, n_iters, n_seq):

    # init params
    start_scores = np.random.rand(n_states)
    actions_scores = np.random.rand(n_states, n_actions)
    transition_scores = np.random.rand(n_states, n_actions, n_states)
    reward_mus = np.random.rand(n_states, n_actions)
    reward_sigmas = np.random.rand(n_states, n_actions)

    start_scores = torch.from_numpy(start_scores).requires_grad_()
    actions_scores = torch.from_numpy(actions_scores).requires_grad_()
    transition_scores = torch.from_numpy(transition_scores).requires_grad_()
    reward_mus = torch.from_numpy(reward_mus).requires_grad_()
    reward_sigmas = torch.from_numpy(reward_sigmas).requires_grad_()

    # optimizer
    params = list([start_scores, actions_scores, transition_scores, reward_mus, reward_sigmas])
    optimizer = torch.optim.Adam(params, lr=1e-2)

    # start iterations
    for iter in tqdm(range(n_iters)):

        # calculate probabilities from scores
        start_probs = F.softmax(start_scores, dim=0)
        action_probs = F.softmax(actions_scores, dim=1)
        transition_probs = F.softmax(transition_scores, dim=2)

        # number of gradient steps to be made (aternatively use a stopping criteria for checking convergence)
        num_samples = 50

        for _ in range(num_samples):

            # sample a sequence (index) from dataset
            m = np.random.randint(n_seq)

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

        # if iter % 10 == 0:
        #     print('iter:{} \t loss:{:3f}'.format(iter, loss.item()))

    return start_probs, action_probs, transition_probs, reward_mus, reward_sigmas




# main
def main(seed):
    # hyperparams
    n_states = 2 # K
    n_actions = 2 # A
    n_seq = 200 # M
    seq_len = 50 # N
    n_iters = 100
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

    ## generate dataset

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

    starting_state_probs_est, action_probs_est, transition_probs_est, reward_mus_est, reward_sigmas_est = mle(dataset, n_states, n_actions, seq_len, n_iters, n_seq)

    print('----starting_state_probs_est----\n', starting_state_probs_est.data.numpy())
    print('----action_probs_est----\n', action_probs_est.data.numpy())
    print('----transition_probs_est----\n', transition_probs_est.data.numpy())
    print('----reward_mus_est----\n', reward_mus_est.data.numpy())
    print('----reward_sigmas_est----\n', reward_sigmas_est.data.numpy())


if __name__ == '__main__':
    random_seeds = [0,1,13,42,69,420,2048]
    # random_seeds = [13]
    for s in random_seeds:
        main(s)
