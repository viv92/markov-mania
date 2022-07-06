# Program implementing EM on GMM using:
# 1. the analytically derived equations for E step
# 2. autograd for M step (update params via gradient descent on the M step objective - gradients obtained using autograd)

# In this example,
# p(z) is a categorical distribution where z can take one of k states
# p(x|z) is a gaussian distribution parameterized by (u_k, sigma_k) corresponding to the state k taken up by z

import numpy as np
import torch
import torch.distributions as tdist
import torch.nn.functional as F
from tqdm import tqdm

# torch.autograd.set_detect_anomaly(True)

# function implementing EM
def em(dataset, n_classes, n_samples, n_iters):
    # init params
    mus = torch.from_numpy(np.random.rand(n_classes))
    sigmas = torch.from_numpy(np.random.rand(n_classes))
    class_scores = torch.from_numpy(np.random.rand(n_classes)) # note that these are class scores; class_probs = softmax(class_scores)
    # set requires_grad_ for the parameters
    mus.requires_grad_()
    sigmas.requires_grad_()
    class_scores.requires_grad_()

    # container for maintaining responsibilities
    responsibilities = torch.zeros((n_samples, n_classes))
    # responsibilities.requires_grad_(False)


    # optimizer for M step
    params = list([mus, sigmas, class_scores])
    optimizer = torch.optim.Adam(params, lr=1e-2)

    # start iterations
    for iter in tqdm(range(n_iters)):

        # get class probabilities
        class_probs = F.softmax(class_scores, dim=0)

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

        # responsibilities should be treated as constants in the M step - detach from computation graph
        # otherwise it creates a circular relationship for params and autograd fails (autograd interprets it as an extra backward pass)
        responsibilities_const = responsibilities.detach()

        # number of gradient steps to be made (aternatively use a stopping criteria for checking convergence)
        grad_steps = 100

        for g_step in range(grad_steps):

            m_step_objective = 0
            for k in range(n_classes):
                m_step_objective += torch.dot(responsibilities_const[:, k], ( torch.log(class_probs[k]) + tdist.Normal(loc=mus[k], scale=sigmas[k]).log_prob(dataset) ))

            loss = -m_step_objective
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            # recalculate class probabilities after gradient update
            class_probs = F.softmax(class_scores, dim=0)

        if iter % 10 == 0:
            print('iter:{} \t loss:{:3f}'.format(iter, loss.item()))


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
