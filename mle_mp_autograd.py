# Program implementing maximum likelihood estimation on Markov Process using autograd on MLE objective

# In this example,
# p(z_1) is a categorical distribution where z_1 can take one of k states (thus parameterized by k X 1 parameters )
# p(z_n|z_n-1) is a categorical distribution where z_n can take one of k states conditioned upon the state taken by z_n-1 (thus parameterized by k X k-1 parameters)

# TODO: optimize the implementation using vectorized implementation (alias transition probs into 1D categorical vector and modify dataset accordingly - then use batching with tdist.Categorical)

import numpy as np
import torch
import torch.distributions as tdist
import torch.nn.functional as F
from tqdm import tqdm


# function implementing MLE
def mle(dataset, n_classes, n_seq, seq_len, n_iters, batch_size, lr):

    # init params
    start_scores = np.random.rand(n_classes)
    transition_scores = np.random.rand(n_classes, n_classes)

    start_scores = torch.from_numpy(start_scores).requires_grad_()
    transition_scores = torch.from_numpy(transition_scores).requires_grad_()

    # optimizer
    params = list([start_scores, transition_scores])
    optimizer = torch.optim.Adam(params, lr=lr)

    # start iterations
    for iter in tqdm(range(n_iters)):

        # fetch minibatch
        idx = np.arange(n_seq)
        np.random.shuffle(idx)
        idx = idx[:batch_size]
        dataset_batch = dataset[idx]

        # (re)calculate probabilities from scores (after gradient update)
        start_probs = F.softmax(start_scores, dim=0)
        transition_probs = F.softmax(transition_scores, dim=1)

        objective = torch.sum(tdist.Categorical(start_probs).log_prob(dataset_batch[:, 0]))
        # objective = torch.log(start_probs[dataset[r][0]])

        for r in range(batch_size):
            for n in range(1, seq_len):
                # objective += tdist.Categorical(transition_probs[][]).log_prob(dataset_batch[:,])
                objective += torch.log(transition_probs[dataset[r][n-1]][dataset[r][n]])

        loss = -objective
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        # if iter % 10 == 0:
        #     print('iter:{} \t loss:{:3f}'.format(iter, loss.item()))

    return start_probs, transition_probs




# main
def main(seed):
    # hyperparams
    n_classes = 2 # K
    n_seq = 100
    seq_len = 50
    n_iters = 100
    batch_size = 50
    lr = 1e-2
    random_seed = seed

    # set random seed
    np.random.seed(random_seed)
    torch.manual_seed(random_seed)

    # true parameters to be estimated
    starting_class_probs_true = [0.6, 0.4] # shape = (1,K)
    transition_class_probs_true = [[0.7, 0.3], [0.4, 0.6]] # shape = (K,K)

    ## generate dataset

    dataset = np.zeros((n_seq, seq_len))

    for r in range(n_seq):

        # starting sample
        z_1 = np.random.choice(n_classes, p=starting_class_probs_true)
        dataset[r][0] = z_1

        # iterate for subsequent sampels
        z_n_minus_1 = z_1
        for n in range(1, seq_len):
            z_n = np.random.choice(n_classes, p=transition_class_probs_true[z_n_minus_1])
            dataset[r][n] = z_n
            # for next data point
            z_n_minus_1 = z_n

    dataset = torch.from_numpy(dataset).int()

    starting_class_probs_est, transition_class_probs_est = mle(dataset, n_classes, n_seq, seq_len, n_iters, batch_size, lr)

    print(starting_class_probs_est.data.numpy())
    print(transition_class_probs_est.data.numpy())


if __name__ == '__main__':
    random_seeds = [0,1,13,42,69,420,2048]
    # random_seeds = [13]
    for s in random_seeds:
        main(s)
