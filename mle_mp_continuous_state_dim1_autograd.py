# Program implementing maximum likelihood estimation on Markov Process with continuous state using autograd on MLE objective

# In this example,
# p(z_1) is a gaussian distribution, mean = u_0, covariance matrix = P_0
# p(z_n|z_n-1) is a gaussuan distribution, mean = A * z_n-1, covariance matrix = Tau

# TODO: optimize the implementation using vectorized implementation (alias transition probs into 1D categorical vector and modify dataset accordingly - then use batching with tdist.Categorical)

import numpy as np
import torch
from torch.distributions.normal import Normal
import torch.nn.functional as F
from tqdm import tqdm
from torch.distributions import Normal


# function implementing MLE
def mle(dataset, S, n_seq, seq_len, n_iters, batch_size, lr):

    # init raw params
    u_0 = np.random.uniform(0, 1)
    P_0_raw = np.random.uniform(1, 2)
    A = np.random.uniform(0, 1)
    Tau_raw = np.random.uniform(1, 2)

    u_0 = torch.tensor(u_0).requires_grad_()
    P_0_raw = torch.tensor(P_0_raw).requires_grad_()
    A = torch.tensor(A).requires_grad_()
    Tau_raw = torch.tensor(Tau_raw).requires_grad_()

    # reformulated params
    P_0 = P_0_raw * P_0_raw +  1e-8
    Tau = Tau_raw * Tau_raw + 1e-8

    # optimizer
    params = list([u_0, P_0_raw, A, Tau_raw])
    optimizer = torch.optim.Adam(params, lr=lr)

    # start iterations
    for iter in tqdm(range(n_iters)):

        # fetch minibatch
        idx = np.arange(n_seq)
        np.random.shuffle(idx)
        idx = idx[:batch_size]
        # print('idx:', idx)
        dataset_batch = dataset[idx]
        # print('dataset_batch:', dataset_batch)
        # print('u_0:', u_0)
        # print('A:', A)

        # loss = 0
        #
        # for r in range(batch_size):
        #
        #     objective = Normal(loc=u_0, scale=P_0).log_prob(dataset_batch[r, 0])
        #
        #     for n in range(1, seq_len):
        #         objective += Normal(loc=torch.matmul(A, dataset_batch[r, n-1]), scale=Tau).log_prob(dataset_batch[r, n])
        #
        #     loss += -objective

        objective = torch.sum( Normal(loc=u_0, scale=P_0).log_prob(dataset_batch[:, 0]) )
        # print('n:{} \t loss:{}'.format(0, -objective.item()))

        for n in range(1, seq_len):
            # d1 = dataset_batch[:, n-1]
            # m1 = d1 * A
            # d2 = dataset_batch[:, n]
            objective += torch.sum( Normal(loc=dataset_batch[:, n-1] * A, scale=Tau).log_prob(dataset_batch[:, n]) )
            # print('n:{} \t loss:{}'.format(n, -objective.item()))
            # print('d1:', d1.item())
            # print('m1:', m1.item())
            # print('d2:', d2.item())

        loss = -objective

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        # reformulate params after gradient update
        P_0 = P_0_raw * P_0_raw +  1e-8
        Tau = Tau_raw * Tau_raw + 1e-8

        if iter % 10 == 0:
            print('iter:{} \t loss:{:3f}'.format(iter, loss.item()))

    return u_0, P_0, A, Tau




# main
def main(seed):
    # hyperparams
    S = 1 # state dim
    n_seq = 100
    seq_len = 50
    n_iters = 100
    batch_size = 1
    lr = 1e-1
    random_seed = seed

    # set random seed
    np.random.seed(random_seed)
    torch.manual_seed(random_seed)

    # true parameters to be estimated
    u_0 = 5. # shape = (S,)
    P_0 = .25 # shape = (S,S)
    A = 1.37 # shape = (S,S)
    Tau = .16 # shape = (S,S)

    ## generate dataset

    dataset = np.zeros((n_seq, seq_len)) # each element is the state scalar

    for r in range(n_seq):

        # starting sample
        z_1 = np.random.normal(u_0, P_0)
        dataset[r][0] = z_1

        # iterate for subsequent sampels
        z_n_minus_1 = z_1
        for n in range(1, seq_len):
            z_n = np.random.normal( A*z_n_minus_1, Tau )
            dataset[r][n] = z_n
            # for next data point
            z_n_minus_1 = z_n

    # print('----- dataset -----')
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
