# Program implementing maximum likelihood estimation on Markov Decision Process with continuous state using autograd on MLE objective

# In this example,
# p(s_1) is a multivariate gaussian distribution, mean = u_0, covariance matrix = P_0
# p(a_n|s_n) is a univariate gaussian distribution, mean = FF, covariance matrix = G
# p(s_n|s_n-1, a_n-1) is a multivariate gaussuan distribution, mean = A * s_n-1 + B * a_n-1, covariance_matrix = Tau
# p(r_n|s_n, a_n-1) is a univariate gaussian distribution, mean = C * s_n + D * a_n-1, covariance_matrix = Sigma

# TODO: optimize the implementation using vectorized implementation (alias transition probs into 1D categorical vector and modify dataset accordingly - then use batching with tdist.Categorical)

import numpy as np
import torch
from torch.distributions.multivariate_normal import MultivariateNormal
from torch.distributions import Normal
import torch.nn.functional as F
from tqdm import tqdm


# function implementing MLE
def supervised_model_learning(dataset, S, n_seq, seq_len, n_iters, batch_size, lr):

    # init raw params
    u_0 = np.random.rand(S)
    FF = np.random.rand(S)
    A = np.random.rand(S, S)
    B = np.random.rand(S)
    C = np.random.rand(S)
    D = np.random.uniform(0, 1)

    u_0 = torch.from_numpy(u_0).requires_grad_()
    FF = torch.from_numpy(FF).requires_grad_()
    A = torch.from_numpy(A).requires_grad_()
    B = torch.from_numpy(B).requires_grad_()
    C = torch.from_numpy(C).requires_grad_()
    D = torch.tensor(D).requires_grad_()

    eps = 1e-3

    # optimizer
    params = list([u_0, FF, A, B, C, D])
    optimizer = torch.optim.Adam(params, lr=lr)

    # start iterations
    for iter in tqdm(range(n_iters)):

        # fetch minibatch
        idx = np.arange(n_seq)
        np.random.shuffle(idx)
        idx = idx[:batch_size]
        dataset_batch = dataset[idx]

        ## SUPERVISED LOSS FOR MODEL LEARNING

        seq_loss = 0

        mean_s_0 = u_0
        diff_s_0 = dataset_batch[:, 0, :S] - mean_s_0
        loss_s_0 = torch.trace( torch.matmul(diff_s_0, diff_s_0.T) )

        mean_a_0 = torch.matmul(dataset_batch[:, 0, :S], FF)
        diff_a_0 = dataset_batch[:, 0, S] - mean_a_0
        loss_a_0 = torch.dot(diff_a_0, diff_a_0.T)

        mean_r_0 = torch.matmul(dataset_batch[:, 0, :S], C) + (D * dataset_batch[:, 0, S])
        diff_r_0 = dataset_batch[:, 0, S+1] - mean_r_0
        loss_r_0 = torch.dot(diff_r_0, diff_r_0.T)

        start_loss = loss_s_0 + loss_a_0 + loss_r_0
        seq_loss += start_loss

        optimizer.zero_grad()
        start_loss.backward()
        optimizer.step()

        for n in range(1, seq_len):

            mean_s_n = torch.matmul(dataset_batch[:, n-1, :S], A.T) + torch.matmul(dataset_batch[:, n-1, S].unsqueeze(dim=1), B.unsqueeze(dim=0))
            diff_s_n = dataset_batch[:, n, :S] - mean_s_n
            loss_s_n = torch.trace( torch.matmul(diff_s_n, diff_s_n.T) )

            mean_a_n = torch.matmul(dataset_batch[:, n, :S], FF)
            diff_a_n = dataset_batch[:, n, S] - mean_a_n
            loss_a_n = torch.dot(diff_a_n, diff_a_n.T)

            mean_r_n = torch.matmul(dataset_batch[:, n, :S], C) + (D * dataset_batch[:, n, S])
            diff_r_n = dataset_batch[:, n, S+1] - mean_r_n
            loss_r_n = torch.dot(diff_r_n, diff_r_n.T)

            step_loss = loss_s_n + loss_a_n + loss_r_n
            seq_loss += step_loss

            optimizer.zero_grad()
            step_loss.backward()
            optimizer.step()

            # if iter % 1000 == 0:
            #     print('step_loss:{:3f}'.format(step_loss.item()))


        # optimizer.zero_grad()
        # seq_loss.backward()
        # optimizer.step()

        if iter % 1000 == 0:
            print('iter:{} \t seq_loss:{:3f}'.format(iter, seq_loss.item()))

    return u_0, FF, A, B, C, D




# main
def main(seed):
    # hyperparams
    S = 2 # state dim
    n_seq = 10000
    seq_len = 5
    n_iters = 50000
    batch_size = 100
    lr = 1e-3
    random_seed = seed

    # set random seed
    np.random.seed(random_seed)
    torch.manual_seed(random_seed)

    # true parameters to be estimated
    u_0 = [2., 5.] # shape = (S,)
    P_0 = [[.25, .09], [.09, .16]] # shape = (S,S)
    FF = [3., 7.] # shape = (S,A) = (S,) since action is univariate
    G = 0.25 # scalar since action is univariate
    A = [[3.1, 0.9], [1.3, 2.7]] # shape = (S,S)
    B = [.4, 1.8] # shape = (S,A) = (S,) since action is univariate
    Tau = [[.16, .09], [.09, .25]] # shape = (S,S)
    C = [1.69, 2.45] # shape = (S,R) = (S,) since reward is univariate
    D = 1.13 # scalar since relates a_n to r_n (both scalars)
    Sigma = 1. # scalar since reward is univariate

    ## generate dataset

    dataset = np.zeros((n_seq, seq_len, S+2)) # each element is a concatenated vector (state, action, reward)

    for m in range(n_seq):

        # starting sample
        s_1 = np.random.multivariate_normal(u_0, P_0)
        a_1 = np.random.normal( np.dot(FF, s_1), G )
        r_1 = np.random.normal( np.dot(C, s_1) + (D*a_1), Sigma )
        sa = np.append(s_1, a_1)
        dataset[m][0] = np.append(sa, r_1)

        # iterate for subsequent sampels
        s_n_minus_1, a_n_minus_1 = s_1, a_1
        for n in range(1, seq_len):
            tmp = np.matmul(A, s_n_minus_1)
            s_n = np.random.multivariate_normal( np.matmul(A, s_n_minus_1) + np.dot(B, a_n_minus_1), Tau )
            a_n = np.random.normal( np.dot(FF, s_n), G )
            r_n = np.random.normal( np.dot(C, s_n) + (D*a_n), Sigma )
            sa = np.append(s_n, a_n)
            dataset[m][n] = np.append(sa, r_n)
            # for next data point
            s_n_minus_1, a_n_minus_1 = s_n, a_n

    # print(dataset)

    dataset = torch.from_numpy(dataset)

    u_0_est, FF_est, A_est, B_est, C_est, D_est = supervised_model_learning(dataset, S, n_seq, seq_len, n_iters, batch_size, lr)

    print('------ u_0_est ------')
    print(u_0_est)

    print('------ FF_est ------')
    print(FF_est)

    print('------ A_est ------')
    print(A_est)
    print('------ B_est ------')
    print(B_est)

    print('------ C_est ------')
    print(C_est)
    print('------ D_est ------')
    print(D_est)




if __name__ == '__main__':
    # random_seeds = [0,1,13,42,69,420,2048]
    random_seeds = [13]
    for s in random_seeds:
        main(s)
