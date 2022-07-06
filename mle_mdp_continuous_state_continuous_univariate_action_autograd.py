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
def mle(dataset, S, n_seq, seq_len, n_iters, batch_size, lr, step_iters):

    # init raw params
    u_0 = np.random.rand(S)
    P_0_raw = np.random.rand(S, S) + 1
    FF = np.random.rand(S)
    G_raw = np.random.uniform(0, 1) + 1
    A = np.random.rand(S, S)
    B = np.random.rand(S)
    Tau_raw = np.random.rand(S, S) + 1
    C = np.random.rand(S)
    D = np.random.uniform(0, 1)
    Sigma_raw = np.random.uniform(0, 1) + 1

    u_0 = torch.from_numpy(u_0).requires_grad_()
    P_0_raw = torch.from_numpy(P_0_raw).requires_grad_()
    FF = torch.from_numpy(FF).requires_grad_()
    G_raw = torch.tensor(G_raw).requires_grad_()
    A = torch.from_numpy(A).requires_grad_()
    B = torch.from_numpy(B).requires_grad_()
    Tau_raw = torch.from_numpy(Tau_raw).requires_grad_()
    C = torch.from_numpy(C).requires_grad_()
    D = torch.tensor(D).requires_grad_()
    Sigma_raw = torch.tensor(Sigma_raw).requires_grad_()

    eps = 1e-3

    # reformulated covariances to ensure PD
    P_0 = torch.matmul(P_0_raw, P_0_raw.T) +  eps * torch.eye(S)
    G = F.relu(G_raw) + eps
    Tau = torch.matmul(Tau_raw, Tau_raw.T) + eps * torch.eye(S)
    Sigma = F.relu(Sigma_raw) + eps

    # optimizer
    params = list([u_0, P_0_raw, FF, G_raw, A, B, Tau_raw, C, D, Sigma_raw])
    optimizer = torch.optim.Adam(params, lr=lr)

    # start iterations
    for iter in tqdm(range(n_iters)):

        # fetch minibatch
        idx = np.arange(n_seq)
        np.random.shuffle(idx)
        idx = idx[:batch_size]
        dataset_batch = dataset[idx]

        ## BATCHED IMPLEMENTATION

        seq_loss = 0

        for si in range(step_iters):

            start_objective = torch.sum( MultivariateNormal(loc=u_0, covariance_matrix=P_0).log_prob(dataset_batch[:, 0, :S]) ) # log p(s-0)
            start_objective += torch.sum( Normal(loc=torch.matmul(dataset_batch[:, 0, :S], FF), scale=G).log_prob(dataset_batch[:, 0, S]) ) # log p(a_0|s_0)
            start_objective += torch.sum( Normal(loc=torch.matmul(dataset_batch[:, 0, :S], C) + (D * dataset_batch[:, 0, S]), scale=Sigma).log_prob(dataset_batch[:, 0, S+1]) ) # log p(r_0|s_0, a_0)

            start_loss = -start_objective

            optimizer.zero_grad()
            start_loss.backward()
            optimizer.step()

            # reformulate covariances after gradient update
            P_0 = torch.matmul(P_0_raw, P_0_raw.T) +  eps * torch.eye(S)
            G = F.relu(G_raw) + eps
            Tau = torch.matmul(Tau_raw, Tau_raw.T) + eps * torch.eye(S)
            Sigma = F.relu(Sigma_raw) + eps

        if iter % (n_iters//10) == 0:
            print('iter:{} \t start_loss:{:3f}'.format(iter, start_loss.item()))

        seq_loss += start_loss

        for n in range(1, seq_len):

            for si in range(step_iters):

                step_objective = torch.sum( MultivariateNormal(loc=torch.matmul(dataset_batch[:, n-1, :S], A.T) + torch.matmul(dataset_batch[:, n-1, S].unsqueeze(dim=1), B.unsqueeze(dim=0)), covariance_matrix=Tau).log_prob(dataset_batch[:, n, :S]) ) # log p(s_n|s_n-1, a_n-1)
                step_objective += torch.sum( Normal(loc=torch.matmul(dataset_batch[:, n, :S], FF), scale=G).log_prob(dataset_batch[:, n, S]) ) # log p(a_n|s_n)
                step_objective += torch.sum( Normal(loc=torch.matmul(dataset_batch[:, n, :S], C) + (D * dataset_batch[:, n, S]), scale=Sigma).log_prob(dataset_batch[:, n, S+1]) ) # log p(r_n|s_n, a_n)

                step_loss = -step_objective

                optimizer.zero_grad()
                step_loss.backward()
                optimizer.step()

                # reformulate covariances after gradient update
                P_0 = torch.matmul(P_0_raw, P_0_raw.T) +  eps * torch.eye(S)
                G = F.relu(G_raw) + eps
                Tau = torch.matmul(Tau_raw, Tau_raw.T) + eps * torch.eye(S)
                Sigma = F.relu(Sigma_raw) + eps

            if iter % (n_iters//10) == 0:
                print('iter:{} \t step_loss:{:3f}'.format(iter, step_loss.item()))

            seq_loss += step_loss


        # optimizer.zero_grad()
        # seq_loss.backward()
        # optimizer.step()
        #
        # # reformulate covariances after gradient update
        # P_0 = torch.matmul(P_0_raw, P_0_raw.T) +  eps * torch.eye(S)
        # G = F.relu(G_raw) + eps
        # Tau = torch.matmul(Tau_raw, Tau_raw.T) + eps * torch.eye(S)
        # Sigma = F.relu(Sigma_raw) + eps

        if iter % (n_iters//10) == 0:
            print('iter:{} \t seq_loss:{:3f}'.format(iter, seq_loss.item()))

    return u_0, P_0, FF, G, A, B, Tau, C, D, Sigma




# main
def main(seed):
    # hyperparams
    S = 2 # state dim
    n_seq = 10000
    seq_len = 5
    n_iters = 1000
    step_iters = 100
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

    u_0_est, P_0_est, FF_est, G_est, A_est, B_est, Tau_est, C_est, D_est, Sigma_est = mle(dataset, S, n_seq, seq_len, n_iters, batch_size, lr, step_iters)

    print('------ u_0_est ------')
    print(u_0_est)
    print('------ P_0_est ------')
    print(P_0_est)
    print('------ FF_est ------')
    print(FF_est)
    print('------ G_est ------')
    print(G_est)
    print('------ A_est ------')
    print(A_est)
    print('------ B_est ------')
    print(B_est)
    print('------ Tau_est ------')
    print(Tau_est)
    print('------ C_est ------')
    print(C_est)
    print('------ D_est ------')
    print(D_est)
    print('------ Sigma_est ------')
    print(Sigma_est)



if __name__ == '__main__':
    # random_seeds = [0,1,13,42,69,420,2048]
    random_seeds = [13]
    for s in random_seeds:
        main(s)
