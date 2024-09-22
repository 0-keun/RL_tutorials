import numpy as np
import torch
import torch.nn as nn
import gym
from torch.distributions import Categorical
import matplotlib.pyplot as plt
import os

# visualization On / Off
viz = True

# Set the hyperparameters
GAMMA = 0.99
LR = 3e-4
EPS_CLIP = 0.2
K_EPOCH = 4
BATCH_SIZE = 64
UPDATE_TIMESTEP = 2000

# Define the neural network
class ActorCritic(nn.Module):
    def __init__(self, state_dim, action_dim, hidden_dim=64):
        super(ActorCritic, self).__init__()
        # Common layer
        self.fc1 = nn.Linear(state_dim, hidden_dim)
        # Policy network
        self.action_head = nn.Linear(hidden_dim, action_dim)
        # Value function network
        self.value_head = nn.Linear(hidden_dim, 1)
    
    def forward(self, x):
        x = torch.tanh(self.fc1(x))
        action_probs = torch.softmax(self.action_head(x), dim=-1)
        state_values = self.value_head(x)
        return action_probs, state_values

# Define the PPO Agent (same as in the training code)
class PPOAgent:
    def __init__(self, state_dim, action_dim):
        self.policy = ActorCritic(state_dim, action_dim)
        self.optimizer = torch.optim.Adam(self.policy.parameters(), lr=LR)
        self.policy_old = ActorCritic(state_dim, action_dim)
        self.policy_old.load_state_dict(self.policy.state_dict())
        self.MseLoss = nn.MSELoss()
    
    def select_action(self, state):
        state = torch.FloatTensor(state).unsqueeze(0)
        with torch.no_grad():
            action_probs, _ = self.policy_old(state)
        dist = Categorical(action_probs)
        action = dist.sample()
        return action.item(), dist.log_prob(action)

    def compute_returns(self, rewards, dones, next_value):
        returns = []
        R = next_value
        for reward, done in zip(reversed(rewards), reversed(dones)):
            if done:
                R = 0
            R = reward + GAMMA * R
            returns.insert(0, R)
        return returns

    def update(self, memory):
        # Update during training is omitted during testing
        pass

# Define the testing function
def test_agent(agent, env, episodes=10, render=True):
    agent.policy.eval()  # Switch to evaluation mode
    total_rewards = []
    
    for episode in range(1, episodes + 1):
        state, _ = env.reset()
        ep_reward = 0
        done = False
        while not done:
            action, _ = agent.select_action(state)
            state, reward, terminated, truncated, _ = env.step(action)
            done = terminated or truncated
            ep_reward += reward
            if render:
                env.render()
        total_rewards.append(ep_reward)
        print(f"Test Episode {episode}\tReward: {ep_reward}")
    
    average_reward = np.mean(total_rewards)
    print(f"\nAverage Reward over {episodes} episodes: {average_reward}")
    agent.policy.train()  # Switch back to training mode
    return total_rewards, average_reward

# Main testing loop
def main_test():
    if viz:
        env = gym.make("CartPole-v1", render_mode="human")
    else:
        env = gym.make('CartPole-v1')

    state_dim = env.observation_space.shape[0]
    action_dim = env.action_space.n
    agent = PPOAgent(state_dim, action_dim)
    
    # Set the path to the saved model
    model_path = './model/cartpole-ppo.pth'
    
    if os.path.exists(model_path):
        # Load the model weights
        agent.policy.load_state_dict(torch.load(model_path))
        agent.policy_old.load_state_dict(agent.policy.state_dict())
        print(f"Model successfully loaded from '{model_path}'.")
    else:
        print(f"Model file '{model_path}' does not exist.")
        return
    
    # Perform testing
    test_episodes = 10  # Set the number of episodes to test
    render = True  # Set whether to render the environment (useful when running in a GUI environment)
    
    # Call the testing function
    total_rewards, average_reward = test_agent(agent, env, episodes=test_episodes, render=render)
    
    env.close()

    # Visualize the test results (optional)
    plt.figure(figsize=(10,5))
    plt.plot(range(1, test_episodes + 1), total_rewards, marker='o')
    plt.xlabel('Test Episode')
    plt.ylabel('Reward')
    plt.title('PPO Agent Performance on CartPole-v1')
    plt.grid(True)
    plt.savefig('ppo_cartpole_test_rewards.png')
    plt.show()
    print("Test reward graph saved as 'ppo_cartpole_test_rewards.png'.")

if __name__ == '__main__':
    main_test()
