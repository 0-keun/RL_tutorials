import numpy as np
import random

# env setting
grid_size = 5
goal_state = (4, 4)  
obstacles = [(2, 2), (3, 2), (1, 3)]  

# initialize Q table 
Q_table = np.zeros((grid_size, grid_size, 4))

# set params
alpha = 0.1  # learning rate
gamma = 0.9  # discount factor
epsilon = 0.2  # Exploitation-Exploration balance

# define possible action
actions = ['up', 'down', 'left', 'right']
action_map = {'up': (-1, 0), 'down': (1, 0), 'left': (0, -1), 'right': (0, 1)}

# reward function
def get_reward(state):
    if state == goal_state:
        return 100
    elif state in obstacles:
        return -100
    else:
        return -1

# Check if the agent is in a valid location
def is_valid_state(state):
    x, y = state
    return 0 <= x < grid_size and 0 <= y < grid_size and state not in obstacles

# Q-learning train
def q_learning(num_episodes):
    for episode in range(num_episodes):
        state = (0, 0)  # initial state
        while state != goal_state:
            # Exploitation or Exploration
            if random.uniform(0, 1) < epsilon:
                action = random.choice(actions)  # Exploration: random action
            else:
                action = actions[np.argmax(Q_table[state[0], state[1]])]  # Exploitation: select the action with the Q_max

            # calculate next state
            new_state = (state[0] + action_map[action][0], state[1] + action_map[action][1])

            # if next state is not valid, come back to the previous state
            if not is_valid_state(new_state):
                new_state = state

            # get reward
            reward = get_reward(new_state)

            # update the Q_value
            old_value = Q_table[state[0], state[1], actions.index(action)]
            next_max = np.max(Q_table[new_state[0], new_state[1]])
            Q_table[state[0], state[1], actions.index(action)] = old_value + alpha * (reward + gamma * next_max - old_value)

            # update the state
            state = new_state

# run
q_learning(1000)  # 1000 episodes
np.set_printoptions(formatter={'float': '{:0.2f}'.format}) # Never Mind, It is only for print

print(Q_table)
