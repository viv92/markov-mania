# Program implementing policy optimization from given finite horizon mdp with continuous state and univariate continuous action -  using iterative formula derived using modified backward pass (equivalent to bellmann equation)

# In this example, we use the given finite horizon mdp (true) parameters instead of learning them from data
# p(s_1) is a multivariate gaussian distribution, mean = u_0, covariance matrix = P_0
# p(s_n|s_n-1, a_n-1) is a multivariate gaussuan distribution, mean = A * s_n-1 + B * a_n-1, covariance_matrix = Tau
# p(r_n|s_n, a_n-1) is a univariate gaussian distribution, mean = C * s_n + D * a_n-1, covariance_matrix = Sigma

# policy to be learnt / optimized
# p(a_n|s_n) is a univariate gaussian distribution, mean = FF, covariance matrix = G

# TODO: optimize the implementation using vectorized implementation

# NOTE: this implementation assumes that all seqeunces (trajectories) from the mdp are of same length.
# In another program, we will remove this assumption by adding an explicit end_state / terminal_state in the mdp

import numpy as np
import torch
import torch.distributions as tdist
import torch.nn.functional as F
from tqdm import tqdm


# function implementing policy evaluation as a modified backward message passing over the finite horizon mdp that takes expectation of the reward at each factor node
# this is equivalent to bellmann equation for policy evaluation
# used as inner loop in policy optimization
def policy_eval(S, seq_len, df, mdp, FF):
    N = seq_len

    # unpack mdp params
    u_0, P_0, A, B, Tau, C, D, Sigma = mdp

    # start iteration for policy evaluation
    C_n, D_n = C, D
    OP_n = C_n + D_n * FF
    OP_n = OP_n.float()
    for n in range(N-1, 0, -1):
        # print('dtype(A):{} \t dtype(OP_n):{}'.format(A.dtype, OP_n.dtype))
        C_n = C + df * torch.matmul(A, OP_n)
        D_n = D + df * torch.dot(B, OP_n)
        OP_n = C_n + D_n * FF
        OP_n = OP_n.float()

    # expected return of policy over given mdp
    expected_return = torch.dot(OP_n, u_0)

    return expected_return



# function performing policy optimization
def policy_optimization(S, seq_len, df, mdp, n_iters, lr):
    # policy parameters
    FF = np.random.rand(S)

    FF = torch.from_numpy(FF).requires_grad_()

    # optimizer
    params = [FF]
    optimizer = torch.optim.Adam(lr=lr, params=params)

    for i in tqdm(range(n_iters)):
        expected_return = policy_eval(S, seq_len, df, mdp, FF)
        loss = -expected_return
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        if i % 1000 == 0:
            print('expected_return: ', expected_return.item())

    return FF, expected_return



# main
def main(seed):
    # hyperparams
    S = 2 # state dim
    seq_len = 10
    n_iters = 5000
    lr = 1e-1
    df = 1
    random_seed = seed

    # set random seed
    np.random.seed(random_seed)
    torch.manual_seed(random_seed)

    # # init poicy params - to be optimized
    # FF = [3., 7.] # shape = (S,A) = (S,) since action is univariate
    # G = 0.25 # scalar since action is univariate

    # given mdp
    u_0 = [2., 5.] # shape = (S,)
    P_0 = [[.25, .09], [.09, .16]] # shape = (S,S)
    A = [[9., 3.], [-3., -16.]] # shape = (S,S)
    B = [-4., 1.] # shape = (S,A) = (S,) since action is univariate
    Tau = [[.16, .09], [.09, .25]] # shape = (S,S)
    C = [-1, 3] # shape = (S,R) = (S,) since reward is univariate
    D = -2. # scalar since relates a_n to r_n (both scalars)
    Sigma = 1. # scalar since reward is univariate

    u_0 = torch.tensor(u_0).requires_grad_(False)
    P_0 = torch.tensor(P_0).requires_grad_(False)
    A = torch.tensor(A).requires_grad_(False)
    B = torch.tensor(B).requires_grad_(False)
    Tau = torch.tensor(Tau).requires_grad_(False)
    C = torch.tensor(C).requires_grad_(False)
    D = torch.tensor(D).requires_grad_(False)
    Sigma = torch.tensor(Sigma).requires_grad_(False)

    mdp = [u_0, P_0, A, B, Tau, C, D, Sigma]

    # find optimal policy
    FF, optimal_expected_return = policy_optimization(S, seq_len, df, mdp, n_iters, lr)

    print('-------- optimal policy mean : --------')
    print(FF)
    print('optimal_expected_return: ', optimal_expected_return.data.numpy())



if __name__ == '__main__':
    # random_seeds = [0,1,13,42,69,420,2048]
    random_seeds = [13]
    for s in random_seeds:
        main(s)
