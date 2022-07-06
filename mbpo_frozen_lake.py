# Program implementing model based policy optimization from a dataset of trajectories generated from the frozen lake environment - using modified backward pass - equivalent to bellmann equation

# In the frozen lake environment -
# p(s_1) is a categorical distribution where s_1 can take one of k states (thus parameterized by k X 1 parameters )
# p(a_n|s_n) is a categorical distribution where a_n can take one of A values (thus parameterized by k X A-1 parameters)
# p(s_n|s_n-1, a_n-1) is a categorical distribution where s_n can take one of k states conditioned upon the state taken by s_n-1 and the action value a_n-1 (thus parameterized by k X A X k-1 parameters)
# p(r_n|s_n, a_n) is a gaussian distribution where r_n can take a real number value conditioned upon the state taken by s_n and the action a_n-1 (thus parameterized by k X A X 2 params for mus and sigmas)

# TODO: optimize the implementation using vectorized implementation

import numpy as np
import torch
import torch.distributions as tdist
import torch.nn.functional as F
from tqdm import tqdm
import gym
import matplotlib.pyplot as plt


# function to learn model using mle objective of mdp
def learn_model_using_mle(dataset, n_states, n_actions, seq_len, n_iters, lr):

    # init params
    start_scores = np.random.rand(n_states)
    actions_scores = np.random.rand(n_states, n_actions)
    transition_scores = np.random.rand(n_states, n_actions, n_states)
    reward_mus = np.random.rand(n_states, n_actions)
    reward_sigmas_raw = np.random.rand(n_states, n_actions)

    # warm starting emission probs
    # emission_scores = np.array([[.5, .4, .1], [.1, .3, .6]])

    start_scores = torch.from_numpy(start_scores).requires_grad_()
    actions_scores = torch.from_numpy(actions_scores).requires_grad_()
    transition_scores = torch.from_numpy(transition_scores).requires_grad_()
    reward_mus = torch.from_numpy(reward_mus).requires_grad_()
    reward_sigmas_raw = torch.from_numpy(reward_sigmas_raw).requires_grad_()

    # optimizer
    params = list([start_scores, actions_scores, transition_scores, reward_mus, reward_sigmas_raw])
    optimizer = torch.optim.Adam(params, lr=lr)

    # start iterations
    for iter in range(n_iters):

        # calculate probabilities from scores
        start_probs = F.softmax(start_scores, dim=0)
        action_probs = F.softmax(actions_scores, dim=1)
        transition_probs = F.softmax(transition_scores, dim=2)
        # ensure reward_sigmas are positive
        reward_sigmas = F.relu(reward_sigmas_raw) + 1e-8


        n_seq = len(dataset)

        for m in tqdm(range(n_seq)):

            objective = torch.log(start_probs[ dataset[m][0][0] ]) + \
                        torch.log(action_probs[ dataset[m][0][0] ][ dataset[m][0][1] ]) + \
                        tdist.Normal(loc=reward_mus[ dataset[m][0][0] ][ dataset[m][0][1] ], scale=reward_sigmas[ dataset[m][0][0] ][ dataset[m][0][1] ]).log_prob( dataset[m][0][2] )

            for n in range(1, seq_len):
                objective += torch.log(transition_probs[ dataset[m][n-1][0] ][ dataset[m][n-1][1] ][ dataset[m][n][0] ]) + \
                             torch.log(action_probs[ dataset[m][n][0] ][ dataset[m][n][1] ]) + \
                             tdist.Normal(loc=reward_mus[ dataset[m][n][0] ][ dataset[m][n][1] ], scale=reward_sigmas[ dataset[m][n][0] ][ dataset[m][n][1] ]).log_prob( dataset[m][n][2] )

            loss = -objective
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            # recalculate probabilities from scores after gradient update
            start_probs = F.softmax(start_scores, dim=0)
            action_probs = F.softmax(actions_scores, dim=1)
            transition_probs = F.softmax(transition_scores, dim=2)
            # ensure reward_sigmas are positive
            reward_sigmas = F.relu(reward_sigmas_raw) + 1e-8

        if iter % 1 == 0:
            print('iter:{} \t loss:{:3f}'.format(iter, loss.item()))

    return start_probs, action_probs, transition_probs, reward_mus, reward_sigmas


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
def policy_eval(n_states, n_actions, seq_len, df, mdp, policy):
    N = seq_len
    action_probs = policy

    starting_state_probs, transition_probs, reward_mus, reward_sigmas = mdp

    # dp container for state values
    state_values = torch.zeros((N, n_states)).double()

    # value of state s_N-1 - is always bootstrapped from the terminal state (value of terminal state is zero).
    for k in range(n_states):
        for a in range(n_actions):
            state_values[N-1][k] += action_probs[k][a] * reward_mus[k][a]

    # calculate state values for states s_N-2 to s_0
    # backward message passing - or bellman backup equation
    for n in range(N-2, -1, -1):
        for k in range(n_states):
            for a in range(n_actions):
                state_values[n][k] += action_probs[k][a] *  reward_mus[k][a] * n_states + action_probs[k][a] * df * torch.dot( transition_probs[k,a,:], state_values[n+1,:] )

    # expected return of policy over given mdp
    expected_return = 0
    for k in range(n_states):
        expected_return += starting_state_probs[k] * state_values[0][k]

    # print(state_values.int())

    return expected_return


# function performing policy optimization over the learnt model of the finite horizon mdp
def policy_optimization(n_states, n_actions, seq_len, df, mdp, n_iters, lr):
    # policy parameter
    policy_scores = np.random.rand(n_states, n_actions)
    policy_scores = torch.from_numpy(policy_scores).requires_grad_()
    policy = F.softmax(policy_scores, dim=1)

    # optimizer
    params = [policy_scores]
    optimizer = torch.optim.Adam(lr=lr, params=params)

    for i in tqdm(range(n_iters)):
        expected_return = policy_eval(n_states, n_actions, seq_len, df, mdp, policy)
        loss = -expected_return
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        # recalculate policy from updated params
        policy = F.softmax(policy_scores, dim=1)

        if i % 5 == 0:
            print('expected_return: ', expected_return.item())
            visualize_policy(policy)

    return policy, expected_return



# function to generate trajectory dataset using acting policy
def generate_trajectory_dataset(env, dataset, n_seq, seq_len, acting_policy, n_actions):
    for ep in tqdm(range(n_seq)):
        done = False
        state = env.reset()
        ep_steps = 0
        while not done:
            action = np.random.choice(n_actions, p=acting_policy)
            next_state, reward, done, _ = env.step(action)
            dataset[ep][ep_steps] = [state, action, reward]
            if done:
                # print('trace:{} \t next_state:{} \t done:{}'.format(dataset[ep][ep_steps], next_state, done))

                ## handle / formulate terminal state for model learning
                # set rest of the transitions to staying in the terminal state with zero rewards
                terminal_state = next_state
                terminal_reward = 0
                for j in range(ep_steps+1, seq_len):
                    action = np.random.choice(n_actions, p=acting_policy)
                    dataset[ep][j] = [terminal_state, action, terminal_reward]

            state = next_state
            ep_steps += 1
            if ep_steps == seq_len:
                break
    return dataset


# main
def main(seed):
    env = gym.make('FrozenLake-v1', desc=None,map_name="4x4", is_slippery=False)

    # hyperparams
    n_states = 16 # K
    n_actions = 4 # A
    seq_len = 10 # 20 # N - this is an upper bound on the horizon. Mostly the horizon will be determined by the terminal state
    n_seq = 10000 # M - number of sequences (trajectories) in the dataset
    df = 1
    n_iters_model_learning = 10
    lr_model_learning = 1e-4
    n_iters_policy_optimization = 50
    lr_policy_optimization = 1e-1
    random_seed = seed

    # set random seed
    np.random.seed(random_seed)
    torch.manual_seed(random_seed)

    ## generate dataset
    dataset = np.zeros((n_seq, seq_len, 3)) # each entry is a tuple (s_n, a_n, r_n)
    acting_policy = np.ones(n_actions) / n_actions # acting policy - uniformly random
    print('GENERATING DATASET...')
    dataset = generate_trajectory_dataset(env, dataset, n_seq, seq_len, acting_policy, n_actions)
    dataset = torch.from_numpy(dataset).int()

    # learn model of the mdp from dataset of trajectories
    print('LEARNING MODEL...')
    starting_state_probs_est, action_probs_est, transition_probs_est, reward_mus_est, reward_sigmas_est = learn_model_using_mle(dataset, n_states, n_actions, seq_len, n_iters_model_learning, lr_model_learning)

    print('\n----starting_state_probs_est----\n', starting_state_probs_est.data.numpy())
    print('\n----action_probs_est----\n', action_probs_est.data.numpy())
    print('\n----transition_probs_est----\n', transition_probs_est.data.numpy())
    print('\n----reward_mus_est----\n', reward_mus_est.data.numpy())
    print('\n----reward_sigmas_est----\n', reward_sigmas_est.data.numpy())

    # freeze learnt model
    starting_state_probs_est = starting_state_probs_est.detach()
    transition_probs_est = transition_probs_est.detach()
    reward_mus_est = reward_mus_est.detach()
    reward_sigmas_est = reward_sigmas_est.detach()
    mdp_model = [starting_state_probs_est, transition_probs_est, reward_mus_est, reward_sigmas_est]

    # find optimal policy
    print('CALCULATING OPTIMAL POLICY...')
    optimal_policy, optimal_expected_return = policy_optimization(n_states, n_actions, seq_len, df, mdp_model, n_iters_policy_optimization, lr_policy_optimization)

    print('-------- optimal policy : --------')
    print(optimal_policy)
    print('optimal_expected_return: ', optimal_expected_return.data.numpy())


if __name__ == '__main__':
    # random_seeds = [0,1,13,42,69,420,2048]
    random_seeds = [13]
    for s in random_seeds:
        main(s)
