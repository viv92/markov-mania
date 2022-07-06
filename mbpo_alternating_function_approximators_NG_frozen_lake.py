# Program implementing model based policy optimization from a dataset of trajectories generated from the frozen lake environment - using modified backward pass - equivalent to bellmann equation

# In the frozen lake environment -
# p(s_1) is a categorical distribution where s_1 can take one of k states (thus parameterized by k X 1 parameters )
# p(a_n|s_n) is a categorical distribution where a_n can take one of A values (thus parameterized by k X A-1 parameters)
# p(s_n|s_n-1, a_n-1) is a categorical distribution where s_n can take one of k states conditioned upon the state taken by s_n-1 and the action value a_n-1 (thus parameterized by k X A X k-1 parameters)
# p(r_n|s_n, a_n) is a gaussian distribution where r_n can take a real number value conditioned upon the state taken by s_n and the action a_n-1 (thus parameterized by k X A X 2 params for mus and sigmas)

# TODO: optimize the implementation using vectorized implementation

import numpy as np
import torch
import torch.nn as nn
import torch.distributions as tdist
import torch.nn.functional as F
from tqdm import tqdm
import gym
import matplotlib.pyplot as plt
from copy import deepcopy

# torch.autograd.set_detect_anomaly(True)

# function to learn model using mle objective of mdp
def learn_model_using_mle(mdp_model, dataset, n_states, n_actions, seq_len, n_iters, lr, device):

    # unpack mdp model params
    start_probs_model, action_probs_model, transition_probs_model, reward_mu_model, reward_sigma_model = mdp_model

    # params
    params = []
    for model in mdp_model:
        params += list(model.parameters())

    # optimizer
    optimizer = torch.optim.Adam(params=params, lr=lr)

    # start iterations
    for iter in tqdm(range(n_iters)):

        n_seq = len(dataset)

        for m in range(n_seq):

            # get values from function approximators

            # start_prob = start_prob_model( torch.tensor([dataset[m][0][0]]).float().to(device) )
            # action_prob = action_probs_model( torch.tensor([dataset[m][0][0], dataset[m][0][1]]).float().to(device) )
            # reward_mu = reward_mu_model( torch.tensor([dataset[m][0][0], dataset[m][0][1]]).float().to(device) )
            # reward_sigma = reward_sigma_model( torch.tensor([dataset[m][0][0], dataset[m][0][1]]).float().to(device) )
            # reward_sigma_pos = reward_sigma + 1e-8

            start_probs = start_probs_model( torch.tensor([42]).float().to(device) )
            action_probs = action_probs_model( torch.tensor([dataset[m][0][0]]).float().to(device) )
            reward_mu = reward_mu_model( torch.tensor([dataset[m][0][0], dataset[m][0][1]]).float().to(device) )
            reward_sigma = reward_sigma_model( torch.tensor([dataset[m][0][0], dataset[m][0][1]]).float().to(device) )
            reward_sigma_pos = reward_sigma + 1e-8

            objective = torch.log(start_probs)[dataset[m][0][0]] + \
                        torch.log(action_probs)[dataset[m][0][1]] + \
                        tdist.Normal(loc=reward_mu, scale=reward_sigma_pos).log_prob(dataset[m][0][2].to(device))


            for n in range(1, seq_len):

                # get values from function approximators

                # transition_probs = transition_probs_model( torch.tensor([dataset[m][n-1][0], dataset[m][n-1][1]]).float().to(device) )
                # transition_prob = transition_probs[ dataset[m][n][0] ]
                # action_prob = action_prob_model( torch.tensor([dataset[m][n][0], dataset[m][n][1]]).float().to(device) )
                # reward_mu = reward_mu_model( torch.tensor([dataset[m][n][0], dataset[m][n][1]]).float().to(device) )
                # reward_sigma = reward_sigma_model( torch.tensor([dataset[m][n][0], dataset[m][n][1]]).float().to(device) )
                # reward_sigma_pos = reward_sigma + 1e-8

                transition_probs = transition_probs_model( torch.tensor([dataset[m][n-1][0], dataset[m][n-1][1]]).float().to(device) )
                action_probs = action_probs_model( torch.tensor([dataset[m][n][0]]).float().to(device) )
                reward_mu = reward_mu_model( torch.tensor([dataset[m][n][0], dataset[m][n][1]]).float().to(device) )
                reward_sigma = reward_sigma_model( torch.tensor([dataset[m][n][0], dataset[m][n][1]]).float().to(device) )
                reward_sigma_pos = reward_sigma + 1e-8

                objective += torch.log(transition_probs[dataset[m][n][0]]) + \
                             torch.log(action_probs[dataset[m][n][1]]) + \
                             tdist.Normal(loc=reward_mu, scale=reward_sigma_pos).log_prob( dataset[m][n][2].to(device) )

            loss = -objective
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            # if m % 20000 == 0:
            #     print('m:{} \t loss:{:3f}'.format(m, loss.item()))

    print('\nmodel_loss: {:3f}'.format(loss.item()))

    return start_probs_model, action_probs_model, transition_probs_model, reward_mu_model, reward_sigma_model



# visualize policy - for debugging
def visualize_policy(policy, device):
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
            val = policy( torch.tensor([state]).float().to(device) ).squeeze()
            q_table[action][s_r][s_c] = val[action]

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
def policy_eval(n_states, n_actions, seq_len, df, mdp, policy):
    N = seq_len
    action_prob_model = policy

    start_prob_model, transition_probs_model, reward_mu_model, reward_sigma_model = mdp

    # dp container for state values
    state_values = torch.zeros((N, n_states))

    # value of state s_N-1 - is always bootstrapped from the terminal state (value of terminal state is zero).
    for k in range(n_states):
        for a in range(n_actions):

            # # get values from function approximators
            # action_prob_N = action_prob_model( torch.tensor([k, a]).float().to(device) ).squeeze()
            # reward_mu_N = reward_mu_model( torch.tensor([k, a]).float().to(device) ).squeeze()

            # get values from function approximators
            action_prob_N = action_prob_model( torch.tensor([k]).float() ).squeeze()
            reward_mu_N = reward_mu_model( torch.tensor([k, a]).float() ).squeeze()

            state_values[N-1][k] += action_prob_N[a] * reward_mu_N

    # calculate state values for states s_N-2 to s_0
    # backward message passing - or bellman backup equation
    for n in range(N-2, -1, -1):
        for k in range(n_states):
            for a in range(n_actions):

                # # get values from function approximators
                # transition_probs = transition_probs_model( torch.tensor([k, a]).float().to(device) )
                # # transition_prob = transition_probs[k_next]
                # action_prob = action_prob_model( torch.tensor([k, a]).float().to(device) ).squeeze()
                # reward_mu = reward_mu_model( torch.tensor([k, a]).float().to(device) ).squeeze()

                # get values from function approximators
                transition_probs = transition_probs_model( torch.tensor([k, a]).float() )
                # transition_prob = transition_probs[k_next]
                action_prob = action_prob_model( torch.tensor([k]).float() ).squeeze()
                reward_mu = reward_mu_model( torch.tensor([k, a]).float() ).squeeze()

                state_values[n][k] += action_prob[a] * ( reward_mu * n_states + df * torch.dot(transition_probs[:], state_values[n+1,:]) )

    # expected return of policy over given mdp
    expected_return = 0
    for k in range(n_states):

        # get value from function approximator
        starting_state_prob = start_prob_model( torch.tensor([42]).float() )

        expected_return += starting_state_prob[k] * state_values[0][k]

    # print(state_values.int())

    return expected_return


# function performing policy optimization over the learnt model of the finite horizon mdp
def policy_optimization(policy, n_states, n_actions, seq_len, df, mdp, n_iters, lr):

    # optimizer
    params = policy.parameters()
    optimizer2 = torch.optim.Adam(params=params, lr=lr)

    for i in tqdm(range(n_iters)):
        expected_return = policy_eval(n_states, n_actions, seq_len, df, mdp, policy)
        loss = -expected_return
        optimizer2.zero_grad()
        loss.backward()
        optimizer2.step()

        # if i % 5 == 0:
        #     print('expected_return: ', expected_return.item())
        #     visualize_policy(policy, device)

    print('\nexpected_return: ', expected_return.item())

    return policy, expected_return


# epsilon greedy action
def get_epsgreedy_action(greedy_action, n_actions, epsil):
    r = np.random.uniform(0, 1)
    if r < epsil:
        action = np.random.randint(n_actions)
    else:
        action = greedy_action
    return action


# function to generate trajectory dataset using acting policy
def generate_trajectory_dataset(env, dataset, n_seq, seq_len, acting_policy, n_actions, epsil, df, render):
    expected_returns = []

    for ep in range(n_seq):
        ep_return = 0
        done = False
        state = env.reset()
        ep_steps = 0
        while not done:
            if render:
                env.render()
            action_probs = acting_policy( torch.tensor([state]).float() ).numpy()
            greedy_action = np.random.choice(n_actions, p=action_probs)
            action = get_epsgreedy_action(greedy_action, n_actions, epsil)
            next_state, reward, done, _ = env.step(action)
            ep_return += (df ** ep_steps) * reward
            dataset[ep][ep_steps] = [state, action, reward]
            if done:
                # print('trace:{} \t next_state:{} \t done:{}'.format(dataset[ep][ep_steps], next_state, done))

                ## handle / formulate terminal state for model learning
                # set rest of the transitions to staying in the terminal state with zero rewards
                terminal_state = next_state
                terminal_reward = 0
                for j in range(ep_steps+1, seq_len):
                    action_probs = acting_policy( torch.tensor([terminal_state]).float() ).numpy()
                    greedy_action = np.random.choice(n_actions, p=action_probs)
                    action = get_epsgreedy_action(greedy_action, n_actions, epsil)
                    dataset[ep][j] = [terminal_state, action, terminal_reward]

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
    isSlippery = False # True
    n_states = 16 # K
    n_actions = 4 # A
    seq_len = 10 # 20 # N - this is an upper bound on the horizon. Mostly the horizon will be determined by the terminal state
    n_seq = 1000 # 10000 # M - number of sequences (trajectories) in the dataset
    df = 1
    outer_iters = 10 # 150
    n_iters_model_learning = 10
    lr_model_learning = 1e-3 # 1e-2
    n_iters_policy_optimization = 100 # 1
    lr_policy_optimization = 1e-3 # 1e-1
    epsil = 1. # decayed
    epsil_init = epsil
    epsil_decay_step = 1/outer_iters

    render_final_policy_episodes = False
    use_cuda = False

    device1 = torch.device('cpu')
    device2 = torch.device('cuda:0')
    device = device1
    if use_cuda:
        device = device2

    # set random seed
    np.random.seed(seed)
    torch.manual_seed(seed)

    env = gym.make('FrozenLake-v1', desc=None,map_name="4x4", is_slippery=isSlippery)

    ## init function approximators

    # input = constant (42), output = probability vector over starting states
    start_probs_model = nn.Sequential(
                        nn.Linear(1, n_states),
                        nn.ReLU(),
                        nn.Linear(n_states, n_states),
                        nn.Softmax(dim=0)
                        ).to(device)

    # input = state, output = probability vector over actions
    action_probs_model = nn.Sequential(
                        nn.Linear(1, n_states + n_actions),
                        nn.ReLU(),
                        nn.Linear(n_states + n_actions, n_actions),
                        nn.Softmax(dim=0)
                        ).to(device)

    # input = (state, action), output = probability vector over next_states
    transition_probs_model = nn.Sequential(
                            nn.Linear(2, n_states + n_actions),
                            nn.ReLU(),
                            nn.Linear(n_states + n_actions, n_states),
                            nn.Softmax(dim=0)
                            ).to(device)

    # input = (state, action), output = mu of the reward gaussian
    reward_mu_model = nn.Sequential(
                        nn.Linear(2, n_states + n_actions),
                        nn.ReLU(),
                        nn.Linear(n_states + n_actions, 1)
                        ).to(device)

    # input = (state, action), output = sigma of the reward gaussian
    reward_sigma_model = nn.Sequential(
                            nn.Linear(2, n_states + n_actions),
                            nn.ReLU(),
                            nn.Linear(n_states + n_actions, 1),
                            nn.ReLU()
                            ).to(device)

    # policy function approximator: input = state, output = prob vector over actions
    policy = nn.Sequential(
            nn.Linear(1, n_states + n_actions),
            nn.ReLU(),
            nn.Linear(n_states + n_actions, n_actions),
            nn.Softmax(dim=0)
            ).to(device)


    # # init policy params to be learnt - start with uniformly random
    # policy_scores = np.ones((n_states, n_actions))
    # for s in range(n_states):
    #     policy_scores[s] /= n_actions
    # policy_scores = torch.from_numpy(policy_scores).requires_grad_()

    # acting policy - frozen version of the learnt policy
    # used to gather data for model learaning
    # iteratively updated with the improved / learned policy
    acting_policy = deepcopy(policy).requires_grad_(False)


    # # init model params
    # # start with random model params - iteratively replace with improved / learned params
    # start_scores = np.random.rand(n_states)
    # actions_scores = np.random.rand(n_states, n_actions)
    # transition_scores = np.random.rand(n_states, n_actions, n_states)
    # reward_mus = np.random.rand(n_states, n_actions)
    # reward_sigmas_raw = np.random.rand(n_states, n_actions)
    # start_scores = torch.from_numpy(start_scores).requires_grad_()
    # actions_scores = torch.from_numpy(actions_scores).requires_grad_()
    # transition_scores = torch.from_numpy(transition_scores).requires_grad_()
    # reward_mus = torch.from_numpy(reward_mus).requires_grad_()
    # reward_sigmas_raw = torch.from_numpy(reward_sigmas_raw).requires_grad_()

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
        dataset = np.zeros((n_seq, seq_len, 3)) # each entry is a tuple (s_n, a_n, r_n)
        # print('GENERATING DATASET...')
        dataset, acting_policy_expected_return = generate_trajectory_dataset(env, dataset, n_seq, seq_len, acting_policy, n_actions, epsil, df, render)

        dataset = torch.from_numpy(dataset).int()
        acting_policy_expected_return_list.append(acting_policy_expected_return)

        # pack the improved mdp model params
        mdp_model = [start_probs_model, action_probs_model, transition_probs_model, reward_mu_model, reward_sigma_model]

        # learn model of the mdp from dataset of trajectories
        # print('LEARNING MODEL...')
        start_probs_model, action_probs_model, transition_probs_model, reward_mu_model, reward_sigma_model = learn_model_using_mle(mdp_model, dataset, n_states, n_actions, seq_len, n_iters_model_learning, lr_model_learning, device)

        # print('\n----starting_state_probs_est----\n', starting_state_probs_est.data.numpy())
        # print('\n----action_probs_est----\n', action_probs_est.data.numpy())
        # print('\n----transition_probs_est----\n', transition_probs_est.data.numpy())
        # print('\n----reward_mus_est----\n', reward_mus_est.data.numpy())
        # print('\n----reward_sigmas_est----\n', reward_sigmas_est.data.numpy())

        # create a frozen copy of the learnt mdp model for policy optimization

        start_probs_model_frozen = deepcopy(start_probs_model).requires_grad_(False)
        transition_probs_model_frozen = deepcopy(transition_probs_model).requires_grad_(False)
        reward_mu_model_frozen = deepcopy(reward_mu_model).requires_grad_(False)
        reward_sigma_model_frozen = deepcopy(reward_sigma_model).requires_grad_(False)

        mdp_model_frozen = [start_probs_model_frozen, transition_probs_model_frozen, reward_mu_model_frozen, reward_sigma_model_frozen]

        # find optimal policy
        # print('LEARNING POLICY...')
        policy, learnt_policy_expected_return = policy_optimization(policy, n_states, n_actions, seq_len, df, mdp_model_frozen, n_iters_policy_optimization, lr_policy_optimization)

        # print expected return
        # print('learnt_policy_expected_return: ', learnt_policy_expected_return.data.numpy())

        # acting policy - frozen version of the learnt policy
        # used to gather data for model learaning
        # iteratively updated with the improved / learned policy
        acting_policy = deepcopy(policy).requires_grad_(False)

        # if oi % 1 == 0:
        #     # visualize policy
        #     visualize_policy(acting_policy)

    # plot acting policy returns

    hyperparams_dict = {}
    hyperparams_dict['slippery'] = isSlippery
    hyperparams_dict['seed'] = seed
    hyperparams_dict['seq_len'] = seq_len
    hyperparams_dict['n_seq'] = n_seq
    hyperparams_dict['outer_iters'] = outer_iters
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
    plt.savefig('plots/mbpo_alternating_functionApproximators_frozenlake' + hyperparams_string + '_.png')



    # ## generate dataset
    # dataset = np.zeros((n_seq, seq_len, 3)) # each entry is a tuple (s_n, a_n, r_n)
    # acting_policy = np.ones(n_actions) / n_actions # acting policy - uniformly random
    # print('GENERATING DATA...')
    # dataset = generate_trajectory_dataset(env, dataset, n_seq, seq_len, acting_policy, n_actions)
    # dataset = torch.from_numpy(dataset).int()
    #
    # # learn model of the mdp from dataset of trajectories
    # print('LEARNING MODEL...')
    # start_prob_model, action_prob_model, transition_probs_model, reward_mu_model, reward_sigma_model = learn_model_using_mle(dataset, n_states, n_actions, seq_len, n_iters_model_learning, lr_model_learning, device1)
    #
    # # print('\n----starting_state_probs_est----\n', starting_state_probs_est.data.numpy())
    # # print('\n----action_probs_est----\n', action_probs_est.data.numpy())
    # # print('\n----transition_probs_est----\n', transition_probs_est.data.numpy())
    # # print('\n----reward_mus_est----\n', reward_mus_est.data.numpy())
    # # print('\n----reward_sigmas_est----\n', reward_sigmas_est.data.numpy())
    #
    # # freeze learnt mdp model
    # for p in start_prob_model.parameters():
    #     p.requires_grad = False
    # for p in transition_probs_model.parameters():
    #     p.requires_grad = False
    # for p in reward_mu_model.parameters():
    #     p.requires_grad = False
    # for p in reward_sigma_model.parameters():
    #     p.requires_grad = False
    # mdp_model = [start_prob_model.to(device1), transition_probs_model.to(device1), reward_mu_model.to(device1), reward_sigma_model.to(device1)]
    #
    # # find optimal policy
    # print('CALCULATING OPTIMAL POLICY...')
    # optimal_policy, optimal_expected_return = policy_optimization(n_states, n_actions, seq_len, df, mdp_model, n_iters_policy_optimization, lr_policy_optimization, device1)
    #
    # print('-------- optimal policy : --------')
    # print(optimal_policy)
    # print('optimal_expected_return: ', optimal_expected_return.data.numpy())


if __name__ == '__main__':
    # random_seeds = [0,1,13,42,69,420,2048]
    random_seeds = [13]
    for s in random_seeds:
        main(s)
