# Program implementing EM on HMM using:
# 1. forward-backward algorithm for E step
# 2. autograd for M step (update params via gradient descent on the M step objective - gradients obtained using autograd)

# In this example,
# p(z_1) is a categorical distribution where z_1 can take one of k states (thus parameterized by k X 1 parameters )
# p(z_n|z_n-1) is a categorical distribution where z_n can take one of k states conditioned upon the state taken by z_n-1 (thus parameterized by k X k parameters)
# p(x_n|z_n) is a categorical distribution where x_n can take one of d states conditioned upon the state taken by z_n (thus parameterized by d X k parameters)

import numpy as np
import torch
import torch.distributions as tdist
import torch.nn.functional as F
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
def em(dataset, n_latent_classes, n_obs_classes, seq_len, n_iters):

    # init params
    start_scores = np.random.rand(n_latent_classes)
    transition_scores = np.random.rand(n_latent_classes, n_latent_classes)
    emission_scores = np.random.rand(n_latent_classes, n_obs_classes)

    start_scores = torch.from_numpy(start_scores).requires_grad_()
    transition_scores = torch.from_numpy(transition_scores).requires_grad_()
    emission_scores = torch.from_numpy(emission_scores).requires_grad_()

    # container for maintaining responsibilities
    responsibilities_marginal = torch.zeros((seq_len, n_latent_classes)) # gamma(zn) = p(zn | X)
    responsibilities_joint = torch.zeros((seq_len-1, n_latent_classes, n_latent_classes)) # epsilon(zn, znplus1) = p(zn, znplus1 | X)

    # optimizer for M step
    params = list([start_scores, transition_scores, emission_scores])
    optimizer = torch.optim.Adam(params, lr=1e-2)

    # start iterations
    for iter in tqdm(range(n_iters)):

        # calculate probabilities from scores
        start_probs = F.softmax(start_scores, dim=0)
        transition_probs = F.softmax(transition_scores, dim=1)
        emission_probs = F.softmax(emission_scores, dim=1)

        ## E step

        # calculate responsibilities
        responsibilities_marginal, responsibilities_joint = forward_backward(dataset, n_latent_classes, seq_len, start_probs, transition_probs, emission_probs)

        ## M step

        # responsibilities should be treated as constants in the M step - detach from computation graph
        # otherwise it creates a circular relationship for params and autograd fails (autograd interprets it as an extra backward pass)
        responsibilities_marginal = responsibilities_marginal.detach()
        responsibilities_joint = responsibilities_joint.detach()

        # number of gradient steps to be made (aternatively use a stopping criteria for checking convergence)
        grad_steps = 20

        for g_step in range(grad_steps):

            m_step_objective = 0

            for k in range(n_latent_classes):
                m_step_objective += responsibilities_marginal[0][k] * torch.log(start_probs[k])

            sumN = torch.sum(responsibilities_joint, axis=0)
            for j in range(n_latent_classes):
                for k in range(n_latent_classes):
                    m_step_objective += sumN[j][k] * torch.log(transition_probs[j][k])


            # emission_probs_vector = torch.zeros((seq_len, n_latent_classes))
            # for n in range(seq_len):
            #     emission_probs_vector[n] = emission_probs[:, dataset[n]]
            for k in range(n_latent_classes):
                m_step_objective += torch.dot( responsibilities_marginal[:, k], tdist.Categorical(emission_probs[k]).log_prob(dataset).float() )

            loss = -m_step_objective
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            # recalculate probabilities from scores after gradient update
            start_probs = F.softmax(start_scores, dim=0)
            transition_probs = F.softmax(transition_scores, dim=1)
            emission_probs = F.softmax(emission_scores, dim=1)

        # if iter % 10 == 0:
        #     print('iter:{} \t loss:{:3f}'.format(iter, loss.item()))

    return start_probs, transition_probs, emission_probs




# main
def main(seed):
    # hyperparams
    n_latent_classes = 2 # K
    n_obs_classes = 3 # D
    seq_len = 100
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

    dataset = np.zeros(seq_len)

    # starting sample
    z_1 = np.random.choice(n_latent_classes, p=starting_latent_class_probs_true)
    x_1 = np.random.choice(n_obs_classes, p=emission_probs_true[z_1])
    dataset[0] = x_1

    # iterate for subsequent samples
    z_n_minus_1 = z_1
    for n in range(1, seq_len):
        z_n = np.random.choice(n_latent_classes, p=transition_latent_class_probs_true[z_n_minus_1])
        x_n = np.random.choice(n_obs_classes, p=emission_probs_true[z_n])
        dataset[n] = x_n
        # for next data point
        z_n_minus_1 = z_n

    # print(dataset)

    dataset = torch.from_numpy(dataset).int()

    starting_latent_class_probs_est, transition_latent_class_probs_est, emission_probs_est = em(dataset, n_latent_classes, n_obs_classes, seq_len, n_iters)

    print(starting_latent_class_probs_est.data.numpy())
    print(transition_latent_class_probs_est.data.numpy())
    print(emission_probs_est.data.numpy())


if __name__ == '__main__':
    random_seeds = [0,1,13,42,69,420,2048]
    # random_seeds = [13]
    for s in random_seeds:
        main(s)
