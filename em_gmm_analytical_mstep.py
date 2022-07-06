# Program implementing EM on GMM using the analytically derived equations for E and M steps

# In this example,
# p(z) is a categorical distribution where z can take one of k states
# p(x|z) is a gaussian distribution parameterized by (u_k, sigma_k) corresponding to the state k taken up by z

import numpy as np
import torch
import torch.distributions as tdist
from tqdm import tqdm

# function implementing EM
def em(dataset, n_classes, n_samples, n_iters):
    # init params
    mus = torch.from_numpy(np.random.rand(n_classes))
    sigmas = torch.from_numpy(np.random.rand(n_classes))
    class_probs = torch.from_numpy(np.random.dirichlet(np.ones(n_classes)))

    # container for maintaining responsibilities
    responsibilities = torch.zeros((n_samples, n_classes))

    # start iterations
    for iter in tqdm(range(n_iters)):

        ## E step

        # calculate responsibilities
        responsibilities_norm = torch.zeros(n_samples) # to store the norm - used for normalizing
        for k in range(n_classes):
            responsibilities[:, k] = class_probs[k] * torch.exp(tdist.Normal(loc=mus[k], scale=sigmas[k]).log_prob(dataset))
            responsibilities_norm += responsibilities[:, k]
        # normalize responsibilities
        for k in range(n_classes):
            responsibilities[:, k] /= responsibilities_norm

        ## M step

        # get N_k values
        N_ks = torch.zeros(n_classes)
        for k in range(n_classes):
            N_ks[k] = torch.sum(responsibilities[:, k])

        # calculate class probabilities
        for k in range(n_classes):
            class_probs[k] = N_ks[k] / n_samples

        # calculate mus
        for k in range(n_classes):
            mus[k] = torch.dot(responsibilities[:, k], dataset) / N_ks[k]

        # calculate sigmas
        for k in range(n_classes):
            sigmas[k] = torch.sqrt(torch.dot( responsibilities[:, k], torch.pow((dataset - mus[k]), 2) ) / N_ks[k])
            # sigmas[k] += 1e-8 # to ensure sigma > 0

    return class_probs, mus, sigmas




# main
def main():
    # hyperparams
    n_classes = 2
    n_samples = 1000
    n_iters = 100
    random_seed = 1

    # set random seed
    np.random.seed(random_seed)
    torch.manual_seed(random_seed)

    # true parameters to be estimated
    class_probs_true = [0.6, 0.4]
    mus_true = [2.5, 4.8]
    sigmas_true = [0.6, 0.3]

    # generate dataset
    dataset = torch.zeros(n_samples)
    for i in range(n_samples):
        sampled_class = np.random.choice(n_classes, p=class_probs_true)
        corresponding_emission_distribution = tdist.Normal(loc=mus_true[sampled_class], scale=sigmas_true[sampled_class])
        sampled_datapoint = corresponding_emission_distribution.sample()
        dataset[i] = sampled_datapoint

    class_probs_est, mus_est, sigmas_est = em(dataset, n_classes, n_samples, n_iters)

    print(class_probs_est.data.numpy())
    print(mus_est.data.numpy())
    print(sigmas_est.data.numpy())


if __name__ == '__main__':
    main()
