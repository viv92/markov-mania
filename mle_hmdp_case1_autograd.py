## Program estimating the parameters of an example hidden markov decision process (HMDP) by formulating MLE as a two step algorithm:
# 1. forward-backward algorithm for calculating responsibilities (E step)
# 2. formulating the likelihood in terms of the responsibilities calculated in step 1 and maximizing the likelihood objective using autograd (M step)

## Case - 1:
# probability distibutions:
# p(s_1); p(o_n | s_n); p(a_n | s_n); p(r_n | s_n, a_n); p(s_n | s_n_minus_1, a_n_minus_1)

# In this example,
# p(s_1) is a categorical distribution where s_1 can take one of k states (thus parameterized by k X 1 parameters )
# p(o_n | s_n) is a categorical distibution where o_n can take one of d states, conditioned on the state taken by s_n (thus parameterized by k X d params)
# p(a_n | s_n) is a categorical distibution where a_n can take one of A states, conditioned on the state taken by s_n (thus parameterized by k X A params)
# p(r_n | s_n, a_n) is a gaussian distribution whose mu and sigma are conditioned on state taken by s_n and a_n (thus parameterized by k X A X 2 params)
# p(s_n | s_n_minus_1, a_n_minus_1) is a categorical distribution where s_n can take one of k states, conditioned on the state taken by s_n_minus_1 and a_n_minus_1 (thus parameterised by k X A X k params)


import numpy as np
import torch
import torch.distributions as tdist
import torch.nn.functional as F
from tqdm import tqdm


# forward-backward algo (scaled) used to calculate responsibilities in E step
def forward_backward(dataset, n_states, seq_len, start_probabilities, action_probabilities, transition_probabilities, emission_probabilities, reward_mus, reward_sigmas):
    N = seq_len
    S = n_states
    alpha = torch.zeros((N,S))
    beta = torch.zeros((N,S))
    scale_factor = torch.zeros(N)

    ## forward pass
    # alpha for first observation
    curr_obs, curr_action, curr_reward, = dataset[0]
    curr_obs, curr_action = int(curr_obs), int(curr_action)
    for j in range(S):
        alpha[0][j] = start_probabilities[j] * \
                      emission_probabilities[j][curr_obs] * \
                      action_probabilities[j][curr_action] * \
                      torch.exp(tdist.Normal(loc=reward_mus[j][curr_action], scale=reward_sigmas[j][curr_action]).log_prob(curr_reward))
    # rescale / normalize
    sf = sum(alpha[0])
    alpha[0] /= sf
    scale_factor[0] = sf # store scale factor (used for backward pass)

    # iterate for subsequent dataset
    for i in range(1, N):
        curr_obs, curr_action, curr_reward = dataset[i]
        curr_obs, curr_action = int(curr_obs), int(curr_action)
        prev_action = int(dataset[i-1][1])
        for j in range(S):
            for k in range(S):
                alpha[i][j] += alpha[i-1][k] * transition_probabilities[k][prev_action][j] * emission_probabilities[j][curr_obs] * \
                               action_probabilities[j][curr_action] * \
                               torch.exp(tdist.Normal(loc=reward_mus[j][curr_action], scale=reward_sigmas[j][curr_action]).log_prob(curr_reward))

        # rescale / normalize
        sf = sum(alpha[i])
        alpha[i] /= sf
        scale_factor[i] = sf # store scale factor (used for backward pass)

    ## backward pass

    # init beta (accounting for the explicit end state)
    beta[N-1] = 1

    # iterate to calculate rest of the beta values
    for i in range(N-2, -1, -1):
        next_obs, next_action, next_reward = dataset[i+1]
        next_obs, next_action = int(next_obs), int(next_action)
        curr_action = int(dataset[i][1])
        for j in range(S):
            for k in range(S):
                beta[i][j] += beta[i+1][k] * transition_probabilities[j][curr_action][k] * emission_probabilities[k][next_obs] * \
                              action_probabilities[k][next_action] * \
                              torch.exp(tdist.Normal(loc=reward_mus[k][next_action], scale=reward_sigmas[k][next_action]).log_prob(next_reward))

        # rescale / normalize
        beta[i] /= scale_factor[i+1] # note i+1 index of scale factor is used here


    ## calculate posterior probabilities

    p_z_given_X = torch.zeros((N,S)) # marginal posterior, i.e., p_z_given_X[n][j] = p(zn = state[j] | X)
    p_zn_znplus1_given_X = torch.zeros((N-1, S, S)) # joint posterior, i.e., p_zn_znplus1_given_X[n][j][k] = p(zn = state[j],zn+1 = state[k] | X)

    # p_X (data likelihood) when using scaled values of alpha and beta :
    p_X = torch.prod(scale_factor)

    # iteratively calculate marginal poseriors
    for i in range(N):
        # p_z_given_X when using un-scaled values of alpha and beta :
        # p_z_given_X[i] = (alpha[i] * beta[i]) / p_X

        # p_z_given_X when using scaled values of alpha and beta :
        p_z_given_X[i] = (alpha[i] * beta[i])

    # iteratively calculate joint posteriors
    for i in range(0, N-1):
        next_obs, next_action, next_reward = dataset[i+1]
        next_obs, next_action = int(next_obs), int(next_action)
        curr_action = int(dataset[i][1])
        for j in range(S):
            for k in range(S):
                p_zn_znplus1_given_X[i][j][k] = alpha[i][j] * beta[i+1][k] * transition_probabilities[j][curr_action][k] * \
                                                emission_probabilities[k][next_obs] * action_probabilities[k][next_action] * \
                                                torch.exp(tdist.Normal(loc=reward_mus[k][next_action], scale=reward_sigmas[k][next_action]).log_prob(next_reward))


        # normalization factor when using scaled values
        p_zn_znplus1_given_X[i] /= scale_factor[i+1]

    return p_z_given_X, p_zn_znplus1_given_X




# function implementing MLE
def mle(dataset, n_states, n_obs, n_actions, seq_len, n_iters, n_seq, batch_size, lr):

    # init params
    start_scores = np.random.rand(n_states)
    action_scores = np.random.rand(n_states, n_actions)
    transition_scores = np.random.rand(n_states, n_actions, n_states)
    emission_scores = np.random.rand(n_states, n_obs)
    reward_mus = np.random.rand(n_states, n_actions)
    reward_sigmas_raw = np.random.rand(n_states, n_actions)

    start_scores = torch.from_numpy(start_scores).requires_grad_()
    action_scores = torch.from_numpy(action_scores).requires_grad_()
    transition_scores = torch.from_numpy(transition_scores).requires_grad_()
    emission_scores = torch.from_numpy(emission_scores).requires_grad_()
    reward_mus = torch.from_numpy(reward_mus).requires_grad_()
    reward_sigmas_raw = torch.from_numpy(reward_sigmas_raw).requires_grad_()

    # container for maintaining responsibilities
    responsibilities_marginal = torch.zeros((seq_len, n_states)) # gamma(zn) = p(zn | X)
    responsibilities_joint = torch.zeros((seq_len-1, n_states, n_states)) # epsilon(zn, znplus1) = p(zn, znplus1 | X)

    # optimizer for M step
    params = list([start_scores, action_scores, transition_scores, emission_scores, reward_mus, reward_sigmas_raw])
    optimizer = torch.optim.Adam(params, lr=lr)

    # start iterations
    for iter in tqdm(range(n_iters)):

        # fetch batch
        idx = np.arange(n_seq)
        np.random.shuffle(idx)
        idx = idx[:batch_size]
        dataset_batch = dataset[idx]

        loss = 0

        # (re)calculate probabilities from (updated) scores
        start_probs = F.softmax(start_scores, dim=0)
        action_probs = F.softmax(action_scores, dim=1)
        transition_probs = F.softmax(transition_scores, dim=2)
        emission_probs = F.softmax(emission_scores, dim=1)
        reward_sigmas = F.relu(reward_sigmas_raw) + 1e-8

        for i in range(batch_size):

            ## E step

            # calculate responsibilities
            with torch.no_grad():
                responsibilities_marginal, responsibilities_joint = forward_backward(dataset_batch[i], n_states, seq_len, start_probs, action_probs, transition_probs, emission_probs, reward_mus, reward_sigmas)

            ## M step

            # responsibilities should be treated as constants in the M step - detach from computation graph
            # otherwise it creates a circular relationship for params and autograd fails (autograd interprets it as an extra backward pass)
            # responsibilities_marginal = responsibilities_marginal.detach()
            # responsibilities_joint = responsibilities_joint.detach()

            m_step_objective = 0

            for k in range(n_states):
                m_step_objective += responsibilities_marginal[0][k] * torch.log(start_probs[k])

            for n in range(seq_len-1):
                for j in range(n_states):
                    for k in range(n_states):
                        m_step_objective += responsibilities_joint[n][j][k] * torch.log(transition_probs[j][int(dataset_batch[i,n,1])][k])

            for k in range(n_states):
                # p(o_n | s_nk)
                m_step_objective += torch.dot( responsibilities_marginal[:, k], tdist.Categorical(emission_probs[k]).log_prob(dataset_batch[i,:,0]).float() )

                # p(a_n | s_nk)
                m_step_objective += torch.dot( responsibilities_marginal[:, k], tdist.Categorical(action_probs[k]).log_prob(dataset_batch[i,:,1]).float() )

            for n in range(seq_len):
                for k in range(n_states):
                    # p(r_n | s_nk, a_nk)
                    m_step_objective += responsibilities_marginal[n, k] * tdist.Normal( loc=reward_mus[k][int(dataset_batch[i,n,1])], scale=reward_sigmas[k][int(dataset_batch[i,n,1])] ).log_prob(dataset_batch[i,n,2]).float()

            loss += -m_step_objective

        # take gradient step
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        if iter % 100 == 0:
            print('iter:{} \t loss:{:3f}'.format(iter, loss.item()))

    return start_probs, action_probs, transition_probs, emission_probs, reward_mus, reward_sigmas




# main
def main(seed):
    # hyperparams
    n_states = 2 # K
    n_obs = 3 # D
    n_actions = 2 # A
    seq_len = 10
    n_iters = 1000
    n_seq = 10000
    batch_size = 50
    lr = 1e-1
    random_seed = seed

    # set random seed
    np.random.seed(random_seed)
    torch.manual_seed(random_seed)

    # true parameters to be estimated
    starting_state_probs_true = [0.6, 0.4] # shape = (1, K)
    action_probs_true = [[0.8, 0.2], [0.6, 0.4]] # aka policy; shape = (K, A)
    transition_probs_true = np.zeros((n_states, n_actions, n_states)) # shape = (K, A, K)
    transition_probs_true[0][0] = [.1, .9]
    transition_probs_true[0][1] = [.2, .8]
    transition_probs_true[1][0] = [.7, .3]
    transition_probs_true[1][1] = [.6, .4]
    emission_probs_true = [[0.5, 0.4, 0.1], [0.1, 0.3, 0.6]] # shape (K, D)
    reward_mus_true = [[8., 4.], [2., 10.]] # shape = (K, A)
    reward_sigmas_true = [[.5, .5], [1., 1.]] # shape = (K, A)

    ## generate dataset

    dataset = np.zeros((n_seq, seq_len, 3)) # each term in a sequence is a tuple (obs, action, reward)

    for i in range(n_seq):

        # starting sample
        s_1 = np.random.choice(n_states, p=starting_state_probs_true)
        o_1 = np.random.choice(n_obs, p=emission_probs_true[s_1])
        a_1 = np.random.choice(n_actions, p=action_probs_true[s_1])
        r_1 = np.random.normal(loc=reward_mus_true[s_1][a_1], scale=reward_sigmas_true[s_1][a_1])
        dataset[i][0] = o_1, a_1, r_1

        # iterate for subsequent samples
        s_n_minus_1 = s_1
        a_n_minus_1 = a_1
        for n in range(1, seq_len):
            s_n = np.random.choice(n_states, p=transition_probs_true[s_n_minus_1][a_n_minus_1])
            o_n = np.random.choice(n_obs, p=emission_probs_true[s_n])
            a_n = np.random.choice(n_actions, p=action_probs_true[s_n])
            r_n = np.random.normal(loc=reward_mus_true[s_n][a_n], scale=reward_sigmas_true[s_n][a_n])
            dataset[i][n] = o_n, a_n, r_n
            # for next data point
            s_n_minus_1 = s_n
            a_n_minus_1 = a_n

    # print(dataset)

    dataset = torch.from_numpy(dataset)
    # dataset = torch.from_numpy(dataset)

    starting_state_probs_est, action_probs_est, transition_probs_est, emission_probs_est, reward_mus_est, reward_sigmas_est = mle(dataset, n_states, n_obs, n_actions, seq_len, n_iters, n_seq, batch_size, lr)

    print('------ start_probs -------')
    print(starting_state_probs_est.data.numpy())
    print('------ action_probs -------')
    print(action_probs_est.data.numpy())
    print('------ transition_probs -------')
    print(transition_probs_est.data.numpy())
    print('------ emission_probs -------')
    print(emission_probs_est.data.numpy())
    print('------ reward_mus -------')
    print(reward_mus_est)
    print('------ reward_sigmas -------')
    print(reward_sigmas_est)


if __name__ == '__main__':
    # random_seeds = [0,1,13,42,69,420,2048]
    random_seeds = [13]
    for s in random_seeds:
        main(s)
