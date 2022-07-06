# Program implementing maximum likelihood estimation on Markov Reward Process using autograd on MLE objective

# In this example,
# p(z_1) is a categorical distribution where z_1 can take one of k states (thus parameterized by k X 1 parameters )
# p(z_n|z_n-1) is a categorical distribution where z_n can take one of k states conditioned upon the state taken by z_n-1 (thus parameterized by k X k-1 parameters)
# p(r_n|z_n) is a gaussian distribution where r_n can take a real number value conditioned upon the state taken by z_n (thus parameterized by k X 2 params for mus and sigmas)

# TODO: optimize the implementation using vectorized implementation (alias transition probs into 1D categorical vector and modify dataset accordingly - then use batching with tdist.Categorical)

import numpy as np
import torch
import torch.distributions as tdist
import torch.nn.functional as F
from tqdm import tqdm


# function implementing MLE
def mle(dataset, n_classes, seq_len, n_iters, n_seq):

    # init params
    start_scores = np.random.rand(n_classes)
    transition_scores = np.random.rand(n_classes, n_classes)
    emission_mus = np.random.rand(n_classes)
    emission_sigmas = np.random.rand(n_classes)

    start_scores = torch.from_numpy(start_scores).requires_grad_()
    transition_scores = torch.from_numpy(transition_scores).requires_grad_()
    emission_mus = torch.from_numpy(emission_mus).requires_grad_()
    emission_sigmas = torch.from_numpy(emission_sigmas).requires_grad_()

    # optimizer
    params = list([start_scores, transition_scores, emission_mus, emission_sigmas])
    optimizer = torch.optim.Adam(params, lr=1e-2)

    # start iterations
    for iter in tqdm(range(n_iters)):

        # calculate probabilities from scores
        start_probs = F.softmax(start_scores, dim=0)
        transition_probs = F.softmax(transition_scores, dim=1)

        # number of gradient steps to be made (aternatively use a stopping criteria for checking convergence)
        num_samples = 10

        for _ in range(num_samples):

            # sample a sequence (index) from dataset
            m = np.random.randint(n_seq)

            z_0, r_0 = dataset[m][0]
            z_0 = z_0.int()
            objective = torch.log(start_probs[z_0]) + tdist.Normal(loc=emission_mus[z_0], scale=emission_sigmas[z_0]).log_prob(r_0)

            for n in range(1, seq_len):
                objective += torch.log(transition_probs[dataset[m][n-1][0].int()][dataset[m][n][0].int()])
                z_n, r_n = dataset[m][n]
                z_n = z_n.int()
                objective += tdist.Normal(loc=emission_mus[z_n], scale=emission_sigmas[z_n]).log_prob(r_n)


            loss = -objective
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            # recalculate probabilities from scores after gradient update
            start_probs = F.softmax(start_scores, dim=0)
            transition_probs = F.softmax(transition_scores, dim=1)

        # if iter % 10 == 0:
        #     print('iter:{} \t loss:{:3f}'.format(iter, loss.item()))

    return start_probs, transition_probs, emission_mus, emission_sigmas




# main
def main(seed):
    # hyperparams
    n_classes = 2 # K
    n_seq = 100 # M
    seq_len = 50 # N
    n_iters = 100
    random_seed = seed

    # set random seed
    np.random.seed(random_seed)
    torch.manual_seed(random_seed)

    # true parameters to be estimated
    starting_class_probs_true = [0.6, 0.4] # shape = (1,K)
    transition_class_probs_true = [[0.7, 0.3], [0.4, 0.6]] # shape = (K,K)
    emission_mus = [2., 10.]
    emission_sigmas = [.5, .5]

    ## generate dataset

    dataset = np.zeros((n_seq, seq_len, 2)) # each entry is a tuple (z_n, r_n)

    for m in range(n_seq):

        # starting sample
        z_1 = np.random.choice(n_classes, p=starting_class_probs_true)
        r_1 = np.random.normal(loc=emission_mus[z_1], scale=emission_sigmas[z_1])
        dataset[m][0] = [z_1, r_1]

        # iterate for subsequent sampels
        z_n_minus_1 = z_1
        for n in range(1, seq_len):
            z_n = np.random.choice(n_classes, p=transition_class_probs_true[z_n_minus_1])
            r_n = np.random.normal(loc=emission_mus[z_n], scale=emission_sigmas[z_n])
            dataset[m][n] = [z_n, r_n]
            # for next data point
            z_n_minus_1 = z_n

    # dataset = torch.from_numpy(dataset).int()
    dataset = torch.from_numpy(dataset)

    starting_class_probs_est, transition_class_probs_est, emission_mus_est, emission_sigmas_est = mle(dataset, n_classes, seq_len, n_iters, n_seq)

    print(starting_class_probs_est.data.numpy())
    print(transition_class_probs_est.data.numpy())
    print(emission_mus_est.data.numpy())
    print(emission_sigmas_est.data.numpy())


if __name__ == '__main__':
    random_seeds = [0,1,13,42,69,420,2048]
    # random_seeds = [13]
    for s in random_seeds:
        main(s)
