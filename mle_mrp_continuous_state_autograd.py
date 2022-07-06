# Program implementing maximum likelihood estimation on Markov Reward Process with continuous state using autograd on MLE objective

# In this example,
# p(s_1) is a multivariate gaussian distribution, mean = u_0, covariance matrix = P_0
# p(s_n|s_n-1) is a multivariate gaussuan distribution, mean = A * s_n-1, covariance_matrix = Tau
# p(r_n|s_n) is a univariate gaussian distribution, mean = C * s_n, covariance_matrix = Sigma

# TODO: optimize the implementation using vectorized implementation (alias transition probs into 1D categorical vector and modify dataset accordingly - then use batching with tdist.Categorical)

import numpy as np
import torch
from torch.distributions.multivariate_normal import MultivariateNormal
from torch.distributions import Normal
import torch.nn.functional as F
from tqdm import tqdm


# function implementing MLE
def mle(dataset, S, n_seq, seq_len, n_iters, batch_size, lr):

    # init raw params
    u_0 = np.random.rand(S)
    P_0_raw = np.random.rand(S, S) + 1
    A = np.random.rand(S, S)
    Tau_raw = np.random.rand(S, S) + 1
    C = np.random.rand(S)
    Sigma_raw = np.random.uniform(0, 1) + 1

    u_0 = torch.from_numpy(u_0).requires_grad_()
    P_0_raw = torch.from_numpy(P_0_raw).requires_grad_()
    A = torch.from_numpy(A).requires_grad_()
    Tau_raw = torch.from_numpy(Tau_raw).requires_grad_()
    C = torch.from_numpy(C).requires_grad_()
    Sigma_raw = torch.tensor(Sigma_raw).requires_grad_()

    # reformulated covariances to ensure PD
    P_0 = torch.matmul(P_0_raw, P_0_raw.T) +  1e-3 * torch.eye(S)
    Tau = torch.matmul(Tau_raw, Tau_raw.T) + 1e-3 * torch.eye(S)
    Sigma = F.relu(Sigma_raw) + 1e-3

    # optimizer
    params = list([u_0, P_0_raw, A, Tau_raw, C, Sigma_raw])
    optimizer = torch.optim.Adam(params, lr=lr)

    # start iterations
    for iter in tqdm(range(n_iters)):

        # fetch minibatch
        idx = np.arange(n_seq)
        np.random.shuffle(idx)
        idx = idx[:batch_size]
        dataset_batch = dataset[idx]

        # ## UN-BATCHED IMPLEMENTATION
        #
        # loss = 0
        #
        # for r in range(batch_size):
        #
        #     objective = torch.sum( MultivariateNormal(loc=u_0, covariance_matrix=P_0).log_prob(dataset_batch[r, 0, :S]) )
        #     objective += torch.sum( Normal(loc=torch.matmul(dataset_batch[r, 0, :S], C), scale=Sigma).log_prob(dataset_batch[r, 0, S]) )
        #
        #     for n in range(1, seq_len):
        #         objective += torch.sum( MultivariateNormal(loc=torch.matmul(dataset_batch[r, n-1, :S], A.T), covariance_matrix=Tau).log_prob(dataset_batch[r, n, :S]) )
        #         objective += torch.sum( Normal(loc=torch.matmul(dataset_batch[r, n, :S], C), scale=Sigma).log_prob(dataset_batch[r, n, S]) )
        #
        #     loss += -objective



        ## BATCHED IMPLEMENTATION

        objective = torch.sum( MultivariateNormal(loc=u_0, covariance_matrix=P_0).log_prob(dataset_batch[:, 0, :S]) )
        objective += torch.sum( Normal(loc=torch.matmul(dataset_batch[:, 0, :S], C), scale=Sigma).log_prob(dataset_batch[:, 0, S]) )

        for n in range(1, seq_len):
            objective += torch.sum( MultivariateNormal(loc=torch.matmul(dataset_batch[:, n-1, :S], A.T), covariance_matrix=Tau).log_prob(dataset_batch[:, n, :S]) )
            objective += torch.sum( Normal(loc=torch.matmul(dataset_batch[:, n, :S], C), scale=Sigma).log_prob(dataset_batch[:, n, S]) )

        loss = -objective

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        # reformulate covariances after gradient update
        P_0 = torch.matmul(P_0_raw, P_0_raw.T) +  1e-3 * torch.eye(S)
        Tau = torch.matmul(Tau_raw, Tau_raw.T) + 1e-3 * torch.eye(S)
        Sigma = F.relu(Sigma_raw) + 1e-3

        if iter % 1000 == 0:
            print('iter:{} \t loss:{:3f}'.format(iter, loss.item()))

    return u_0, P_0, A, Tau, C, Sigma




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
    C = [1.69, 2.45] # shape = (S,R) = (S,) since reward is univariate
    Sigma = 1. # scalar since reward is univariate

    ## generate dataset

    dataset = np.zeros((n_seq, seq_len, S+1)) # each element is a concatenated vector (state, reward)

    for m in range(n_seq):

        # starting sample
        s_1 = np.random.multivariate_normal(u_0, P_0)
        r_1 = np.random.normal( np.dot(C, s_1), Sigma )
        dataset[m][0] = np.append(s_1, r_1)

        # iterate for subsequent sampels
        s_n_minus_1 = s_1
        for n in range(1, seq_len):
            s_n = np.random.multivariate_normal( np.matmul(A, s_n_minus_1), Tau )
            r_n = np.random.normal( np.dot(C, s_n), Sigma )
            dataset[m][n] = np.append(s_n, r_n)
            # for next data point
            s_n_minus_1 = s_n

    # print(dataset)

    dataset = torch.from_numpy(dataset)

    u_0_est, P_0_est, A_est, Tau_est, C_est, Sigma_est = mle(dataset, S, n_seq, seq_len, n_iters, batch_size, lr)

    print('------ u_0_est ------')
    print(u_0_est)
    print('------ P_0_est ------')
    print(P_0_est)
    print('------ A_est ------')
    print(A_est)
    print('------ Tau_est ------')
    print(Tau_est)
    print('------ C_est ------')
    print(C_est)
    print('------ Sigma_est ------')
    print(Sigma_est)



if __name__ == '__main__':
    # random_seeds = [0,1,13,42,69,420,2048]
    random_seeds = [13]
    for s in random_seeds:
        main(s)
