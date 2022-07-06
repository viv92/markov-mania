# Program implementing maximum likelihood estimation on Markov Process with continuous state using autograd on MLE objective

# In this example,
# p(z_1) is a gaussian distribution, mean = u_0, covariance matrix = P_0
# p(z_n|z_n-1) is a gaussuan distribution, mean = A * z_n-1, covariance matrix = Tau

# TODO: optimize the implementation using vectorized implementation (alias transition probs into 1D categorical vector and modify dataset accordingly - then use batching with tdist.Categorical)

import numpy as np
import torch
from torch.distributions.multivariate_normal import MultivariateNormal
import torch.nn.functional as F
from tqdm import tqdm


# function implementing MLE
def mle(dataset, S, n_seq, seq_len, n_iters, batch_size, lr):

    # init raw params
    u_0 = np.random.rand(S)
    P_0_raw = np.random.rand(S, S) + 1
    A = np.random.rand(S, S)
    Tau_raw = np.random.rand(S, S) + 1

    u_0 = torch.from_numpy(u_0).requires_grad_()
    P_0_raw = torch.from_numpy(P_0_raw).requires_grad_()
    A = torch.from_numpy(A).requires_grad_()
    Tau_raw = torch.from_numpy(Tau_raw).requires_grad_()

    # reformulated covariance matrices to ensure PD
    P_0 = torch.matmul(P_0_raw, P_0_raw.T) + 1e-3 * torch.eye(S)
    Tau = torch.matmul(Tau_raw, Tau_raw.T) + 1e-3 * torch.eye(S)

    # optimizer
    params = list([u_0, P_0_raw, A, Tau_raw])
    optimizer = torch.optim.Adam(params, lr=lr)

    # start iterations
    for iter in tqdm(range(n_iters)):

        # fetch minibatch
        idx = np.arange(n_seq)
        np.random.shuffle(idx)
        idx = idx[:batch_size]
        dataset_batch = dataset[idx]

        # loss = 0
        #
        # for r in range(batch_size):
        #     print('-------r: ', r)
        #
        #     objective = MultivariateNormal(loc=u_0, covariance_matrix=P_0).log_prob(dataset_batch[r, 0])
        #
        #     for n in range(1, seq_len):
        #         print('---------n: ', n)
        #
        #         objective += MultivariateNormal(loc=torch.matmul(A, dataset_batch[r, n-1]), covariance_matrix=Tau).log_prob(dataset_batch[r, n])
        #
        #     loss += -objective

        objective = torch.sum( MultivariateNormal(loc=u_0, covariance_matrix=P_0).log_prob(dataset_batch[:, 0]) )

        for n in range(1, seq_len):
            objective += torch.sum( MultivariateNormal(loc=torch.matmul(dataset_batch[:, n-1], A.T), covariance_matrix=Tau).log_prob(dataset_batch[:, n]) )

        loss = -objective

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        # reformulate covariance matrices after gradient update
        P_0 = torch.matmul(P_0_raw, P_0_raw.T) + 1e-3 * torch.eye(S)
        Tau = torch.matmul(Tau_raw, Tau_raw.T) + 1e-3 * torch.eye(S)

        if iter % 1000 == 0:
            print('iter:{} \t loss:{:3f}'.format(iter, loss.item()))

    return u_0, P_0, A, Tau




# main
def main(seed):
    # hyperparams
    S = 2 # state dim
    n_seq = 1000
    seq_len = 5
    n_iters = 10000
    batch_size = 1
    lr = 1e-1
    random_seed = seed

    # set random seed
    np.random.seed(random_seed)
    torch.manual_seed(random_seed)

    # true parameters to be estimated
    u_0 = [2., 5.] # shape = (S,)
    P_0 = [[.25, .09], [.09, .16]] # shape = (S,S)
    A = [[3.1, 0.9], [1.3, 2.7]] # shape = (S,S)
    Tau = [[.16, .09], [.09, .25]] # shape = (S,S)

    ## generate dataset

    dataset = np.zeros((n_seq, seq_len, S)) # each element is the state vector

    for r in range(n_seq):

        # starting sample
        z_1 = np.random.multivariate_normal(u_0, P_0)
        dataset[r][0] = z_1

        # iterate for subsequent sampels
        z_n_minus_1 = z_1
        for n in range(1, seq_len):
            z_n = np.random.multivariate_normal( np.matmul(A, z_n_minus_1), Tau )
            dataset[r][n] = z_n
            # for next data point
            z_n_minus_1 = z_n

    # print(dataset)

    dataset = torch.from_numpy(dataset)

    u_0_est, P_0_est, A_est, Tau_est = mle(dataset, S, n_seq, seq_len, n_iters, batch_size, lr)

    print('------ u_0_est ------')
    print(u_0_est)
    print('------ P_0_est ------')
    print(P_0_est)
    print('------ A_est ------')
    print(A_est)
    print('------ Tau_est ------')
    print(Tau_est)



if __name__ == '__main__':
    # random_seeds = [0,1,13,42,69,420,2048]
    random_seeds = [13]
    for s in random_seeds:
        main(s)
