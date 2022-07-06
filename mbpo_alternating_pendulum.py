# Program showcasing alternating model based policy optimization (mbpo) on a continuous state and continuous action environment (inverted pendulum environment)

# In the inverted pendulum environment -
# state dim = 3
# p(s_1) is a multivariate gaussian distribution, mean = u_0, covariance matrix = P_0
# p(a_n|s_n) is a univariate gaussian distribution, mean = FF, covariance matrix = G
# p(s_n|s_n-1, a_n-1) is a multivariate gaussuan distribution, mean = A * s_n-1 + B * a_n-1, covariance_matrix = Tau
# p(r_n|s_n, a_n-1) is a univariate gaussian distribution, mean = C * s_n + D * a_n-1, covariance_matrix = Sigma

# TODO: optimize the implementation using vectorized implementation

import numpy as np
import torch
from torch.distributions import Normal
from torch.distributions.multivariate_normal import MultivariateNormal
import torch.nn.functional as F
from tqdm import tqdm
import gym
import matplotlib.pyplot as plt


# function implementing MLE
def learn_model_using_step_mle(mdp_model_params, dataset, S, n_seq, seq_len, n_iters, batch_size, lr, step_iters):

    # unpack mdp model params
    u_0, P_0_raw, FF, G_raw, A, B, Tau_raw, C, D, Sigma_raw = mdp_model_params

    eps = 1e-8

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
    # for iter in range(n_iters):

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

        # if iter % (n_iters//10) == 0:
        #     print('iter:{} \t start_loss:{:3f}'.format(iter, start_loss.item()))

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

            # if iter % (n_iters//10) == 0:
            #     print('iter:{} \t step_loss:{:3f}'.format(iter, step_loss.item()))

            seq_loss += step_loss


        # optimizer.zero_grad()
        # seq_loss.backward()
        # optimizer.step()
        #
        # reformulate covariances after gradient update
        # P_0 = torch.matmul(P_0_raw, P_0_raw.T) +  eps * torch.eye(S)
        # G = F.relu(G_raw) + eps
        # Tau = torch.matmul(Tau_raw, Tau_raw.T) + eps * torch.eye(S)
        # Sigma = F.relu(Sigma_raw) + eps

        # if iter % (n_iters//10) == 0:
        print('iter:{} \t seq_loss:{:3f}'.format(iter, seq_loss.item()))

    return u_0, P_0_raw, FF, G_raw, A, B, Tau_raw, C, D, Sigma_raw




# function to learn model using mle objective of mdp
def learn_model_using_seq_mle(mdp_model_params, dataset, S, n_seq, seq_len, n_iters, batch_size, lr, step_iters=None):
    # unpack mdp model params
    u_0, P_0_raw, FF, G_raw, A, B, Tau_raw, C, D, Sigma_raw = mdp_model_params

    # reformulated covariances to ensure PD
    P_0 = torch.matmul(P_0_raw, P_0_raw.T) +  1e-8 * torch.eye(S)
    G = F.relu(G_raw) + 1e-8
    Tau = torch.matmul(Tau_raw, Tau_raw.T) + 1e-8 * torch.eye(S)
    Sigma = F.relu(Sigma_raw) + 1e-8

    # optimizer
    params = list([u_0, P_0_raw, FF, G_raw, A, B, Tau_raw, C, D, Sigma_raw])
    optimizer = torch.optim.Adam(params, lr=lr)

    # start iterations
    for iter in tqdm(range(n_iters)):
    # for iter in range(n_iters):


        # fetch minibatch
        idx = np.arange(n_seq)
        np.random.shuffle(idx)
        idx = idx[:batch_size]
        dataset_batch = dataset[idx]

        ## BATCHED IMPLEMENTATION

        objective = torch.sum( MultivariateNormal(loc=u_0, covariance_matrix=P_0).log_prob(dataset_batch[:, 0, :S]) ) # log p(s-0)
        objective += torch.sum( Normal(loc=torch.matmul(dataset_batch[:, 0, :S], FF), scale=G).log_prob(dataset_batch[:, 0, S]) ) # log p(a_0|s_0)
        objective += torch.sum( Normal(loc=torch.matmul(dataset_batch[:, 0, :S], C) + (D * dataset_batch[:, 0, S]), scale=Sigma).log_prob(dataset_batch[:, 0, S+1]) ) # log p(r_0|s_0, a_0)

        for n in range(1, seq_len):
            objective += torch.sum( MultivariateNormal(loc=torch.matmul(dataset_batch[:, n-1, :S], A.T) + torch.matmul(dataset_batch[:, n-1, S].unsqueeze(dim=1), B.unsqueeze(dim=0)), covariance_matrix=Tau).log_prob(dataset_batch[:, n, :S]) ) # log p(s_n|s_n-1, a_n-1)
            objective += torch.sum( Normal(loc=torch.matmul(dataset_batch[:, n, :S], FF), scale=G).log_prob(dataset_batch[:, n, S]) ) # log p(a_n|s_n)
            objective += torch.sum( Normal(loc=torch.matmul(dataset_batch[:, n, :S], C) + (D * dataset_batch[:, n, S]), scale=Sigma).log_prob(dataset_batch[:, n, S+1]) ) # log p(r_0|s_0, a_0)

        loss = -objective

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        # reformulate covariances after gradient update
        P_0 = torch.matmul(P_0_raw, P_0_raw.T) +  1e-8 * torch.eye(S)
        G = F.relu(G_raw) + 1e-8
        Tau = torch.matmul(Tau_raw, Tau_raw.T) + 1e-8 * torch.eye(S)
        Sigma = F.relu(Sigma_raw) + 1e-8

        if iter % (n_iters//10) == 0:
            print('iter:{} \t model_loss:{:3f}'.format(iter, loss.item()))

    return u_0, P_0_raw, FF, G_raw, A, B, Tau_raw, C, D, Sigma_raw




# visualize policy - for debugging
def visualize_policy(policy):
    n_obs = 16
    obs_rows = int(np.math.sqrt(n_obs))
    obs_cols = obs_rows
    nA = 4
    q_table = np.zeros((nA, obs_rows, obs_cols))
    max_q_table = np.zeros((obs_rows, obs_cols))

    for action in range(nA):
        for state in range(n_obs):
            s_r = state // obs_cols
            s_c = state % obs_cols
            q_table[action][s_r][s_c] = policy[state][action]

    fig = plt.figure()
    #timer = fig.canvas.new_timer(interval = 300) #creating a timer object and setting an interval of 3000 milliseconds
    #timer.add_callback(close_event)

    arrow_dict = {}
    arrow_dict[0] = '<'
    arrow_dict[1] = 'v'
    arrow_dict[2] = '>'
    arrow_dict[3] = '^'
    for i in range(obs_rows):
        for j in range(obs_cols):
            arg = np.argmax(q_table[:,i,j])
            arrow = arrow_dict[arg]
            plt.text(j, i, arrow, ha='center', va='center', color='red')
            max_q_table[i, j] = q_table[arg, i, j]

    plt.imshow(max_q_table)
    #timer.start()
    plt.show()



# function implementing policy evaluation as a modified backward message passing over the learnt model of the finite horizon mdp that takes expectation of the reward at each factor node
# this is equivalent to bellmann equation for policy evaluation
def policy_eval(S, seq_len, df, mdp, FF):
    N = seq_len

    # unpack mdp params
    u_0, P_0, A, B, Tau, C, D, Sigma = mdp

    # start iteration for policy evaluation
    C_n, D_n = C, D
    OP_n = C_n + D_n * FF
    # OP_n = OP_n.float()
    for n in range(N-1, 0, -1):
        # print('dtype(A):{} \t dtype(OP_n):{}'.format(A.dtype, OP_n.dtype))
        C_n = C + df * torch.matmul(A, OP_n)
        D_n = D + df * torch.dot(B, OP_n)
        OP_n = C_n + D_n * FF
        # OP_n = OP_n.float()

    # expected return of policy over given mdp
    expected_return = torch.dot(OP_n, u_0)

    return expected_return


# function performing policy optimization over the learnt model of the finite horizon mdp
def policy_optimization(FF, G_raw, S, seq_len, df, mdp, n_iters, lr):

    # optimizer
    params = [FF]
    optimizer = torch.optim.Adam(lr=lr, params=params)

    for i in tqdm(range(n_iters)):
    # for i in range(n_iters):

        expected_return = policy_eval(S, seq_len, df, mdp, FF)
        loss = -expected_return
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        if i % (n_iters//10) == 0:
            print('\niter:{} \t expected_return:{}'.format(i, expected_return.item()))

    return FF, expected_return


# epsilon greedy action
def get_epsgreedy_action(greedy_action, epsil):
    r = np.random.uniform(0, 1)
    if r < epsil:
        action = np.random.uniform(-1.99, 1.99)
    else:
        action = greedy_action
    return action


def form_step_vector(state, action, reward):
    s = np.array(state)
    sa = np.append(s, action)
    ans = np.append(sa, reward)
    return ans


# function to generate trajectory dataset using acting policy
def generate_trajectory_dataset(env, dataset, n_seq, seq_len, acting_FF, acting_G_raw, epsil, df, render):
    expected_returns = []

    for ep in range(n_seq):
        ep_return = 0
        done = False
        state = env.reset()
        ep_steps = 0
        while not done:
            if render:
                env.render()
            greedy_action = np.dot(state, acting_FF)
            action = get_epsgreedy_action(greedy_action, epsil)
            next_state, reward, done, _ = env.step([action])
            if render:
                print('ep_steps:{} \t reward:{}'.format(ep_steps, reward))
            ep_return += (df ** ep_steps) * reward
            step_vector = form_step_vector(state, action, reward)
            dataset[ep][ep_steps] = step_vector
            if done:
                # print('trace:{} \t next_state:{} \t done:{}'.format(dataset[ep][ep_steps], next_state, done))

                ## handle / formulate terminal state for model learning
                # set rest of the transitions to staying in the terminal state with zero rewards
                terminal_state = next_state
                terminal_reward = 0
                for j in range(ep_steps+1, seq_len):
                    greedy_action = np.dot(terminal_state, acting_FF)
                    action = get_epsgreedy_action(greedy_action, epsil)
                    step_vector = form_step_vector(terminal_state, action, terminal_reward)
                    dataset[ep][j] = step_vector

            state = next_state
            ep_steps += 1
            if ep_steps == seq_len:
                break

        expected_returns.append(ep_return)

    average_expected_return = sum(expected_returns) / len(expected_returns)
    return dataset, average_expected_return


# main
def main(seed):

    # hyperparams
    S = 3 # state_dim
    seq_len = 200 # 200 # N - this is an upper bound on the horizon. Mostly the horizon will be determined by the terminal state
    n_seq = 1000 # 1000 # M - number of sequences (trajectories) in the dataset
    df = 1
    model_learning_function_index = 1
    outer_iters = 20 # 20
    step_iters = 5 # 30
    n_iters_model_learning = 30 # 2
    batch_size = 100 # 1000
    lr_model_learning = 1e-3 # 1e-2
    n_iters_policy_optimization = 1000 # 1
    lr_policy_optimization = 1e-5 # 1e-3
    epsil = 0.9 # decayed
    epsil_init = epsil
    epsil_decay_step = 1/outer_iters
    random_seed = seed

    render_final_policy_episodes = False

    dict_model_learning_functions = {}
    dict_model_learning_functions[0] = ['seq_mle', learn_model_using_seq_mle]
    dict_model_learning_functions[1] = ['step_mle', learn_model_using_step_mle]
    model_learning_function_nameString, model_learning_function = dict_model_learning_functions[model_learning_function_index]

    # set random seed
    np.random.seed(random_seed)
    torch.manual_seed(random_seed)

    env = gym.make('Pendulum-v1')

    # init policy params to be learnt - start with random
    policy_FF = np.random.rand(S)
    policy_G_raw = np.random.uniform(0, 1) + 1
    policy_FF = torch.from_numpy(policy_FF).requires_grad_()
    policy_G_raw = torch.tensor(policy_G_raw).requires_grad_()

    # acting policy - frozen version of the learnt policy
    # used to gather data for model learaning
    # iteratively updated with the improved / learned policy
    acting_FF = policy_FF.clone().detach()
    acting_G_raw = policy_G_raw.clone().detach()

    # init model params
    # start with random model params - iteratively replace with improved / learned params
    u_0 = np.random.rand(S)
    P_0_raw = np.random.rand(S, S) + 1
    model_FF = np.random.rand(S)
    model_G_raw = np.random.uniform(0, 1) + 1
    A = np.random.rand(S, S)
    B = np.random.rand(S)
    Tau_raw = np.random.rand(S, S) + 1
    C = np.random.rand(S)
    D = np.random.uniform(0, 1)
    Sigma_raw = np.random.uniform(0, 1) + 1

    u_0 = torch.from_numpy(u_0).requires_grad_()
    P_0_raw = torch.from_numpy(P_0_raw).requires_grad_()
    model_FF = torch.from_numpy(model_FF).requires_grad_()
    model_G_raw = torch.tensor(model_G_raw).requires_grad_()
    A = torch.from_numpy(A).requires_grad_()
    B = torch.from_numpy(B).requires_grad_()
    Tau_raw = torch.from_numpy(Tau_raw).requires_grad_()
    C = torch.from_numpy(C).requires_grad_()
    D = torch.tensor(D).requires_grad_()
    Sigma_raw = torch.tensor(Sigma_raw).requires_grad_()

    # container to store expected return of the acting policy - used for plotting results
    acting_policy_expected_return_list = []

    for oi in tqdm(range(outer_iters)):

        # print('\nOUTER ITER: ', oi)

        # decay epsilon
        if epsil > 0:
            epsil -= epsil_decay_step

        # render final policy if flag set
        render = False
        if render_final_policy_episodes and oi == outer_iters - 1:
            render = True

        ## generate dataset using the improved acting policy
        dataset = np.zeros((n_seq, seq_len, S+2)) # each entry is a tuple (s_n, a_n, r_n)
        # print('GENERATING DATASET...')
        dataset, acting_policy_expected_return = generate_trajectory_dataset(env, dataset, n_seq, seq_len, acting_FF, acting_G_raw, epsil, df, render)

        dataset = torch.from_numpy(dataset)
        acting_policy_expected_return_list.append(acting_policy_expected_return)

        # pack the improved mdp model params
        mdp_model_params = [u_0, P_0_raw, model_FF, model_G_raw, A, B, Tau_raw, C, D, Sigma_raw]

        # learn model of the mdp from dataset of trajectories
        # print('LEARNING MODEL...')
        u_0, P_0_raw, model_FF, model_G_raw, A, B, Tau_raw, C, D, Sigma_raw = model_learning_function(mdp_model_params, dataset, S, n_seq, seq_len, n_iters_model_learning, batch_size, lr_model_learning, step_iters)

        # print('\n----starting_state_probs_est----\n', starting_state_probs_est.data.numpy())
        # print('\n----action_probs_est----\n', action_probs_est.data.numpy())
        # print('\n----transition_probs_est----\n', transition_probs_est.data.numpy())
        # print('\n----reward_mus_est----\n', reward_mus_est.data.numpy())
        # print('\n----reward_sigmas_est----\n', reward_sigmas_est.data.numpy())

        # create a frozen copy of the learnt mdp model for policy optimization

        u_0_frozen = u_0.clone().detach()
        P_0_raw_frozen = P_0_raw.clone().detach()
        model_FF_frozen = model_FF.clone().detach()
        model_G_raw_frozen = model_G_raw.clone().detach()
        A_frozen = A.clone().detach()
        B_frozen = B.clone().detach()
        Tau_raw_frozen = Tau_raw.clone().detach()
        C_frozen = C.clone().detach()
        D_frozen = D.clone().detach()
        Sigma_raw_frozen = Sigma_raw.clone().detach()

        # reformulate covariances
        P_0_frozen = torch.matmul(P_0_raw_frozen, P_0_raw_frozen.T) +  1e-8 * torch.eye(S)
        model_G_frozen = F.relu(model_G_raw_frozen) + 1e-8
        Tau_frozen = torch.matmul(Tau_raw_frozen, Tau_raw_frozen.T) + 1e-8 * torch.eye(S)
        Sigma_frozen = F.relu(Sigma_raw_frozen) + 1e-8

        # pack frozen params of the mdp - used for policy optimization
        mdp_model_frozen = [u_0_frozen, P_0_frozen, A_frozen, B_frozen, Tau_frozen, C_frozen, D_frozen, Sigma_frozen]

        # find optimal policy
        # print('LEARNING POLICY...')
        policy_scores, learnt_policy_expected_return = policy_optimization(policy_FF, policy_G_raw, S, seq_len, df, mdp_model_frozen, n_iters_policy_optimization, lr_policy_optimization)

        # print expected return
        # print('learnt_policy_expected_return: ', learnt_policy_expected_return.data.numpy())

        # acting policy - frozen version of the learnt policy
        # used to gather data for model learaning
        # iteratively updated with the improved / learned policy
        acting_FF = policy_FF.clone().detach()
        acting_G_raw = policy_G_raw.clone().detach()

        # if oi % 1 == 0:
        #     # visualize policy
        #     visualize_policy(acting_policy)

    # plot acting policy returns

    hyperparams_dict = {}
    hyperparams_dict['seed'] = seed
    hyperparams_dict['seq_len'] = seq_len
    hyperparams_dict['n_seq'] = n_seq
    hyperparams_dict['model_lr_func'] = model_learning_function_nameString
    hyperparams_dict['step_iters'] = step_iters
    hyperparams_dict['outer_iters'] = outer_iters
    hyperparams_dict['batch_size'] = batch_size
    hyperparams_dict['n_iters_model_learning'] = n_iters_model_learning
    hyperparams_dict['n_iters_policy_optimization'] = n_iters_policy_optimization
    hyperparams_dict['lr_model_learning'] = lr_model_learning
    hyperparams_dict['lr_policy_optimization'] = lr_policy_optimization
    hyperparams_dict['df'] = df
    hyperparams_dict['epsil'] = epsil_init
    hyperparams_string = ""
    for k,v in hyperparams_dict.items():
        hyperparams_string += "_" + k + ':' + str(v)

    plt.plot(acting_policy_expected_return_list)
    plt.ylabel('Average Episode Return')
    plt.xlabel('Training Epoch')
    plt.title('Average Episode Return versus Training Epochs')
    plt.savefig('plots/mbpo_alternating_pendulum' + hyperparams_string + '_.png')




if __name__ == '__main__':
    # random_seeds = [0,1,13,42,69,420,2048]
    random_seeds = [13]
    for s in random_seeds:
        main(s)
