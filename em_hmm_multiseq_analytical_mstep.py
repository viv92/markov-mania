# Program implementing EM on HMM using:
# 1. forward-backward algorithm for E step
# 2. the analytically derived equations for M step


# In this example,
# p(z_1) is a categorical distribution where z_1 can take one of k states (thus parameterized by k X 1 parameters )
# p(z_n|z_n-1) is a categorical distribution where z_n can take one of k states conditioned upon the state taken by z_n-1 (thus parameterized by k X k parameters)
# p(x_n|z_n) is a categorical distribution where x_n can take one of d states conditioned upon the state taken by z_n (thus parameterized by d X k parameters)

import numpy as np
import torch
import torch.distributions as tdist
from tqdm import tqdm


# forward-backward algo (scaled) used to calculate responsibilities in E step
def forward_backward(dataset, n_latent_classes, seq_len, start_probabilities, transition_probabilities, emission_probabilities):
    N = seq_len
    S = n_latent_classes
    alpha = torch.zeros((N,S))
    beta = torch.zeros((N,S))
    scale_factor = torch.zeros(N)

    ## forward pass
    # alpha for first observation
    curr_obs = dataset[0]
    for j in range(S):
        alpha[0][j] = start_probabilities[j] * emission_probabilities[j][curr_obs]
    # rescale / normalize
    sf = sum(alpha[0])
    alpha[0] /= sf
    scale_factor[0] = sf # store scale factor (used for backward pass)

    # iterate for subsequent dataset
    for i in range(1, N):
        curr_obs = dataset[i]
        for j in range(S):
            for k in range(S):
                alpha[i][j] += alpha[i-1][k] * transition_probabilities[k][j] * emission_probabilities[j][curr_obs]
        # rescale / normalize
        sf = sum(alpha[i])
        alpha[i] /= sf
        scale_factor[i] = sf # store scale factor (used for backward pass)

    ## backward pass

    # init beta (accounting for the explicit end state)
    beta[N-1] = 1

    # iterate to calculate rest of the beta values
    for i in range(N-2, -1, -1):
        next_obs = dataset[i+1]
        for j in range(S):
            for k in range(S):
                beta[i][j] += beta[i+1][k] * transition_probabilities[j][k] * emission_probabilities[k][next_obs]
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
        next_obs = dataset[i+1]
        for j in range(S):
            for k in range(S):
                p_zn_znplus1_given_X[i][j][k] = alpha[i][j] * beta[i+1][k] * transition_probabilities[j][k] * emission_probabilities[k][next_obs]
        # normalization factor when using scaled values
        p_zn_znplus1_given_X[i] /= scale_factor[i+1]

    return p_z_given_X, p_zn_znplus1_given_X




# function implementing EM
def em(dataset, n_latent_classes, n_obs_classes, seq_len, n_seq, n_iters):

    # init params
    start_probs = np.random.dirichlet(np.ones(n_latent_classes))
    transition_probs = np.zeros((n_latent_classes, n_latent_classes))
    emission_probs = np.zeros((n_latent_classes, n_obs_classes))
    for k in range(n_latent_classes):
        transition_probs[k] = np.random.dirichlet(np.ones(n_latent_classes))
        emission_probs[k] = np.random.dirichlet(np.ones(n_obs_classes))

    # warm starting emission probs
    # emission_probs = np.array([[.5, .4, .1], [.1, .3, .6]])

    start_probs = torch.from_numpy(start_probs)
    transition_probs = torch.from_numpy(transition_probs)
    emission_probs = torch.from_numpy(emission_probs)

    # container for maintaining responsibilities
    responsibilities_marginal = torch.zeros((n_seq, seq_len, n_latent_classes)) # gamma(zn) = p(zn | X)
    responsibilities_joint = torch.zeros((n_seq, seq_len-1, n_latent_classes, n_latent_classes)) # epsilon(zn, znplus1) = p(zn, znplus1 | X)

    # start iterations
    for iter in tqdm(range(n_iters)):

        ## E step

        # calculate responsibilities
        for r in range(n_seq):
            responsibilities_marginal[r], responsibilities_joint[r] = forward_backward(dataset[r], n_latent_classes, seq_len, start_probs, transition_probs, emission_probs)

        ## M step

        # calculate start_probs
        sumR = torch.sum(responsibilities_marginal, axis=0)
        for k in range(n_latent_classes):
            start_probs[k] = sumR[0][k] / torch.sum(sumR[0]) # according to bishop eq(13.124)

        # calculate transition probs
        sumR = torch.sum(responsibilities_joint, axis=0)
        sumN = torch.sum(sumR, axis=0) # sumN.shape = (n_latent_classes, n_latent_classes)
        sumKN = torch.sum(sumN, axis=1) # sumKN.shape = (n_latent_classes,)
        for j in range(n_latent_classes):
            for k in range(n_latent_classes):
                transition_probs[j][k] = sumN[j][k] / sumKN[j] # according to bishop eq(13.125)

        # calculate emission_probs (catagorical distribution)
        for k in range(n_latent_classes):
            for i in range(n_obs_classes):
                numerator = 0
                denominator = 0
                for r in range(n_seq):
                    xi = (dataset[r] == i).float() # indicator vector for obs_class = i
                    numerator += torch.dot(responsibilities_marginal[r, :, k], xi)
                    denominator += torch.sum(responsibilities_marginal[r, :, k])
                emission_probs[k][i] = numerator / denominator # according to bishop eq(13.126)

    return start_probs, transition_probs, emission_probs




# main
def main(seed):
    # hyperparams
    n_latent_classes = 2 # K
    n_obs_classes = 3 # D
    n_seq = 100 # R
    seq_len = 20
    n_iters = 100
    random_seed = seed

    # set random seed
    np.random.seed(random_seed)
    torch.manual_seed(random_seed)

    # true parameters to be estimated
    starting_latent_class_probs_true = [0.6, 0.4] # shape = (1,K)
    transition_latent_class_probs_true = [[0.7, 0.3], [0.4, 0.6]] # shape = (K,K)
    emission_probs_true = [[0.5, 0.4, 0.1], [0.1, 0.3, 0.6]] # shape (K,D)

    ## generate dataset

    dataset = np.zeros((n_seq, seq_len))

    for r in range(n_seq):
        # starting sample
        z_1 = np.random.choice(n_latent_classes, p=starting_latent_class_probs_true)
        x_1 = np.random.choice(n_obs_classes, p=emission_probs_true[z_1])
        dataset[r][0] = x_1

        # iterate for subsequent sampels
        z_n_minus_1 = z_1
        for n in range(1, seq_len):
            z_n = np.random.choice(n_latent_classes, p=transition_latent_class_probs_true[z_n_minus_1])
            x_n = np.random.choice(n_obs_classes, p=emission_probs_true[z_n])
            dataset[r][n] = x_n
            # for next data point
            z_n_minus_1 = z_n

    # print(dataset)

    dataset = torch.from_numpy(dataset).int()

    starting_latent_class_probs_est, transition_latent_class_probs_est, emission_probs_est = em(dataset, n_latent_classes, n_obs_classes, seq_len, n_seq, n_iters)

    print(starting_latent_class_probs_est.data.numpy())
    print(transition_latent_class_probs_est.data.numpy())
    print(emission_probs_est.data.numpy())


if __name__ == '__main__':
    random_seeds = [0,1,13,42,69,420,2048]
    # random_seeds = [13]
    for s in random_seeds:
        main(s)
