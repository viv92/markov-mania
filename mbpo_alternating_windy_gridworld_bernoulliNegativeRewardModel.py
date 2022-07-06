# Program implementing alternating model based policy optimization (alternating between learning a model using a learnt policy and learning a policy using a learnt model) from a dataset of trajectories generated from the windy gridworld environment - using modified backward pass - equivalent to bellmann equation

# In the windy gridworld environment -
# p(s_1) is a categorical distribution where s_1 can take one of k states (thus parameterized by k X 1 parameters )
# p(a_n|s_n) is a categorical distribution where a_n can take one of A values (thus parameterized by k X A-1 parameters)
# p(s_n|s_n-1, a_n-1) is a categorical distribution where s_n can take one of k states conditioned upon the state taken by s_n-1 and the action value a_n-1 (thus parameterized by k X A X k-1 parameters)
# p(r_n|s_n, a_n) is a bernoulli distribution where r_n can one of 2 states conditioned upon the state taken by s_n and the action a_n (thus parameterized by k X A params)

# TODO: optimize the implementation using vectorized implementation

import numpy as np
import torch
import torch.distributions as tdist
import torch.nn.functional as F
from tqdm import tqdm
import gym
import matplotlib.pyplot as plt
import gym_gridworlds

# function to convert observation into state
def gs(obs):
    state = obs[0] * 10 + obs[1]
    return state

# function to learn model using mle objective of mdp
def learn_model_using_mle(mdp_model_params, dataset, n_states, n_actions, seq_len, n_iters, lr):
    # unpack params
    start_scores, actions_scores, transition_scores, reward_scores = mdp_model_params

    # optimizer
    params = list(mdp_model_params)
    optimizer = torch.optim.Adam(params, lr=lr)

    # start iterations
    for iter in tqdm(range(n_iters)):

        n_seq = len(dataset)

        for m in range(n_seq):

            # recalculate probabilities from scores after gradient update
            start_probs = F.softmax(start_scores, dim=0)
            action_probs = F.softmax(actions_scores, dim=1)
            transition_probs = F.softmax(transition_scores, dim=2)
            reward_probs = F.sigmoid(reward_scores)

            if dataset[m][0][2] == -1: # if reward == -1
                objective = torch.log(start_probs[ dataset[m][0][0] ]) + \
                            torch.log(action_probs[ dataset[m][0][0] ][ dataset[m][0][1] ]) + \
                            torch.log(reward_probs[ dataset[m][0][0] ][ dataset[m][0][1] ])
            else: # if reward == 0
                objective = torch.log(start_probs[ dataset[m][0][0] ]) + \
                            torch.log(action_probs[ dataset[m][0][0] ][ dataset[m][0][1] ]) + \
                            torch.log(1 - reward_probs[ dataset[m][0][0] ][ dataset[m][0][1] ])


            for n in range(1, seq_len):

                if dataset[m][n][2] == -1: # if reward == 1
                    objective += torch.log(transition_probs[ dataset[m][n-1][0] ][ dataset[m][n-1][1] ][ dataset[m][n][0] ]) + \
                                 torch.log(action_probs[ dataset[m][n][0] ][ dataset[m][n][1] ]) + \
                                 torch.log(reward_probs[ dataset[m][n][0] ][ dataset[m][n][1] ])
                else: # reward == 0
                    objective += torch.log(transition_probs[ dataset[m][n-1][0] ][ dataset[m][n-1][1] ][ dataset[m][n][0] ]) + \
                                 torch.log(action_probs[ dataset[m][n][0] ][ dataset[m][n][1] ]) + \
                                 torch.log(1 - reward_probs[ dataset[m][n][0] ][ dataset[m][n][1] ])

            loss = -objective

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        # if iter % 1 == 0:
        #     print('iter:{} \t loss:{:3f}'.format(iter, loss.item()))

    print('model_loss: {:3f}'.format(loss.item()))

    return start_scores, actions_scores, transition_scores, reward_scores


# visualize policy - for debugging
# def visualize_policy(policy):
#     n_obs = 70
#     obs_rows = 7
#     obs_cols = 10
#     nA = 4
#     q_table = np.zeros((nA, obs_rows, obs_cols))
#
#     for action in range(nA):
#         for state in range(n_obs):
#             s_r = state // obs_cols
#             s_c = state % obs_cols
#             q_table[action][s_r][s_c] = policy[state][action]
#
#     fig, ax = plt.subplots(2,2)
#     #timer = fig.canvas.new_timer(interval = 300) #creating a timer object and setting an interval of 3000 milliseconds
#     #timer.add_callback(close_event)
#
#     ax[0,0].imshow(q_table[0])
#     ax[0,1].imshow(q_table[1])
#     ax[1,0].imshow(q_table[2])
#     ax[1,1].imshow(q_table[3])
#
#     arrow_dict = {}
#     arrow_dict[0] = '<'
#     arrow_dict[1] = 'v'
#     arrow_dict[2] = '>'
#     arrow_dict[3] = '^'
#     for i in range(obs_rows):
#         for j in range(obs_cols):
#             rw = q_table[:,i,j]
#             # print('i:{} j:{} rw:{}'.format(i,j,rw))
#             arg = np.argmax(q_table[:,i,j])
#             arrow = arrow_dict[arg]
#             ax[0,0].text(j, i, arrow, ha='center', va='center', color='red')
#     #timer.start()
#     plt.show()



# visualize policy - for debugging
def visualize_policy(policy):
    n_obs = 70
    obs_rows = 7
    obs_cols = 10
    nA = 4
    q_table = np.zeros((nA, obs_rows, obs_cols))

    for action in range(nA):
        for state in range(n_obs):
            s_r = state // obs_cols
            s_c = state % obs_cols
            q_table[action][s_r][s_c] = policy[state][action]

    fig = plt.figure()
    #timer = fig.canvas.new_timer(interval = 300) #creating a timer object and setting an interval of 3000 milliseconds
    #timer.add_callback(close_event)

    max_q_table = np.zeros((obs_rows, obs_cols))

    arrow_dict = {}
    arrow_dict[3] = '<'
    arrow_dict[0] = '^'
    arrow_dict[1] = '>'
    arrow_dict[2] = 'v'
    for i in range(obs_rows):
        for j in range(obs_cols):
            rw = q_table[:,i,j]
            # print('i:{} j:{} rw:{}'.format(i,j,rw))
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

    starting_state_probs, transition_probs, reward_probs = mdp

    # dp container for state values
    state_values = torch.zeros((N, n_states)).double()

    # value of state s_N-1 - is always bootstrapped from the terminal state (value of terminal state is zero).
    for k in range(n_states):
        for a in range(n_actions):
            state_values[N-1][k] += action_probs[k][a] * -reward_probs[k][a]

    # calculate state values for states s_N-2 to s_0
    # backward message passing - or bellman backup equation
    for n in range(N-2, -1, -1):
        for k in range(n_states):
            for a in range(n_actions):
                state_values[n][k] += action_probs[k][a] * -reward_probs[k][a] * n_states + action_probs[k][a] * df * torch.dot(transition_probs[k,a,:], state_values[n+1,:])

    # expected return of policy over given mdp
    expected_return = 0
    for k in range(n_states):
        expected_return += starting_state_probs[k] * state_values[0][k]

    # print(state_values.int())

    return expected_return


# function performing policy optimization over the learnt model of the finite horizon mdp
def policy_optimization(policy_scores, n_states, n_actions, seq_len, df, mdp, n_iters, lr):

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

        # if i % 5 == 0:
        #     print('expected_return: ', expected_return.item())
        #     visualize_policy2(policy)

    return policy_scores, expected_return



# epsilon greedy action
def get_epsgreedy_action(greedy_action, n_actions, epsil):
    r = np.random.uniform(0, 1)
    if r < epsil:
        action = np.random.randint(n_actions)
    else:
        action = greedy_action
    return action


# function to generate trajectory dataset using acting policy
def generate_trajectory_dataset(env, dataset, n_seq, seq_len, acting_policy, n_actions, epsil, df):
    expected_returns = []

    for ep in tqdm(range(n_seq)):
        ep_return = 0
        done = False
        obs = env.reset()
        state = gs(obs)
        ep_steps = 0
        while not done:
            # env.render()
            action_probs = acting_policy[state].numpy()
            greedy_action = np.random.choice(n_actions, p=action_probs)
            action = get_epsgreedy_action(greedy_action, n_actions, epsil)
            next_obs, reward, done, _ = env.step(action)
            ep_return += (df ** ep_steps) * reward
            next_state = gs(next_obs)
            dataset[ep][ep_steps] = [state, action, reward] # zero reward on intermediate steps
            if done:
                # print('trace:{} \t next_state:{} \t done:{}'.format(dataset[ep][ep_steps], next_state, done))

                ## handle / formulate terminal state for model learning
                # set rest of the transitions to staying in the terminal state with zero rewards
                terminal_state = next_state
                terminal_reward = 0
                for j in range(ep_steps+1, seq_len):
                    action_probs = acting_policy[terminal_state].numpy()
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
    env = gym.make('WindyGridworld-v0')

    # hyperparams
    n_states = 70 # K
    n_actions = 4 # A
    seq_len = 30 # N - this is an upper bound on the horizon. Mostly the horizon will be determined by the terminal state
    n_seq = 1000 # M - number of sequences (trajectories) in the dataset
    df = 1
    outer_iters = 10
    n_iters_model_learning = 1
    lr_model_learning = 1e-2
    n_iters_policy_optimization = 5
    lr_policy_optimization = 1 # 1e-1
    epsil = 0.9 # 0.9 # decayed
    epsil_decay_step = 1/outer_iters
    random_seed = seed

    # set random seed
    np.random.seed(random_seed)
    torch.manual_seed(random_seed)

    # init policy params to be learnt - start with uniformly random
    policy_scores = np.ones((n_states, n_actions))
    for s in range(n_states):
        policy_scores[s] /= n_actions
    policy_scores = torch.from_numpy(policy_scores).requires_grad_()

    # acting policy - frozen version of the learnt policy
    # used to gather data for model learaning
    # iteratively updated with the improved / learned policy
    acting_policy_scores = policy_scores.clone().detach()
    acting_policy = F.softmax(acting_policy_scores, dim=1)

    # init model params
    # start with random model params - iteratively replace with improved / learned params
    start_scores = np.random.rand(n_states)
    actions_scores = np.random.rand(n_states, n_actions)
    transition_scores = np.random.rand(n_states, n_actions, n_states)
    reward_scores = np.random.rand(n_states, n_actions)
    start_scores = torch.from_numpy(start_scores).requires_grad_()
    actions_scores = torch.from_numpy(actions_scores).requires_grad_()
    transition_scores = torch.from_numpy(transition_scores).requires_grad_()
    reward_scores = torch.from_numpy(reward_scores).requires_grad_()

    # container to store expected return of the acting policy - used for plotting results
    acting_policy_expected_return_list = []

    for oi in range(outer_iters):

        print('\nOUTER ITER: ', oi)

        # decay epsilon
        if epsil > 0:
            epsil -= epsil_decay_step

        ## generate dataset using the improved acting policy
        dataset = np.zeros((n_seq, seq_len, 3)) # each entry is a tuple (s_n, a_n, r_n)
        print('GENERATING DATASET...')
        dataset, acting_policy_expected_return = generate_trajectory_dataset(env, dataset, n_seq, seq_len, acting_policy, n_actions, epsil, df)

        dataset = torch.from_numpy(dataset).int()
        acting_policy_expected_return_list.append(acting_policy_expected_return)

        # pack the improved mdp model params
        mdp_model_params = [start_scores, actions_scores, transition_scores, reward_scores]

        # learn model of the mdp from dataset of trajectories
        print('LEARNING MODEL...')
        # *** TODO - input true policy scores (frozen) to the model learning
        start_scores, actions_scores, transition_scores, reward_scores = learn_model_using_mle(mdp_model_params, dataset, n_states, n_actions, seq_len, n_iters_model_learning, lr_model_learning)

        # print('\n----starting_state_probs_est----\n', starting_state_probs_est.data.numpy())
        # print('\n----action_probs_est----\n', action_probs_est.data.numpy())
        # print('\n----transition_probs_est----\n', transition_probs_est.data.numpy())
        # print('\n----reward_scores_est----\n', reward_scores_est.data.numpy())

        # create a frozen copy of the learnt mdp model for policy optimization

        start_scores_frozen = start_scores.clone().detach()
        transition_scores_frozen = transition_scores.clone().detach()
        reward_scores_frozen = reward_scores.clone().detach()

        # calculate probabilities from scores
        start_probs_frozen = F.softmax(start_scores_frozen, dim=0)
        transition_probs_frozen = F.softmax(transition_scores_frozen, dim=2)
        reward_probs_frozen = F.sigmoid(reward_scores_frozen)

        mdp_model_frozen = [start_probs_frozen, transition_probs_frozen, reward_probs_frozen]

        # find optimal policy
        print('LEARNING POLICY...')
        policy_scores, learnt_policy_expected_return = policy_optimization(policy_scores, n_states, n_actions, seq_len, df, mdp_model_frozen, n_iters_policy_optimization, lr_policy_optimization)

        # print expected return
        print('learnt_policy_expected_return: ', learnt_policy_expected_return.data.numpy())

        # acting policy - frozen version of the learnt policy
        # used to gather data for model learaning
        # iteratively updated with the improved / learned policy
        acting_policy_scores = policy_scores.clone().detach()
        acting_policy = F.softmax(acting_policy_scores, dim=1)

        # if oi % 1 == 0:
        #     # visualize policy
        #     visualize_policy(acting_policy)

    # plot acting policy returns

    hyperparams_dict = {}
    hyperparams_dict['seed'] = seed
    hyperparams_dict['seq_len'] = seq_len
    hyperparams_dict['n_seq'] = n_seq
    hyperparams_dict['outer_iters'] = outer_iters
    hyperparams_dict['n_iters_model_learning'] = n_iters_model_learning
    hyperparams_dict['n_iters_policy_optimization'] = n_iters_policy_optimization
    hyperparams_dict['lr_model_learning'] = lr_model_learning
    hyperparams_dict['lr_policy_optimization'] = lr_policy_optimization
    hyperparams_dict['df'] = df
    hyperparams_dict['epsil'] = epsil
    hyperparams_string = ""
    for k,v in hyperparams_dict.items():
        hyperparams_string += "_" + k + ':' + str(v)

    plt.plot(acting_policy_expected_return_list)
    plt.ylabel('Average Episode Return')
    plt.xlabel('Training Epoch')
    plt.title('Average Episode Return versus Training Epochs')
    plt.savefig('plots/mbpo_alternating_windy_gridworld' + hyperparams_string + '_.png')


if __name__ == '__main__':
    # random_seeds = [0,13,42]
    random_seeds = [13]
    for s in random_seeds:
        main(s)
