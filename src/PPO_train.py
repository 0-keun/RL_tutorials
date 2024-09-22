import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import gym
from torch.distributions import Categorical
from torch.utils.data.sampler import BatchSampler, SubsetRandomSampler

# visualization On / Off
viz = True

# Set the hyperparameters
GAMMA = 0.99
LR = 3e-4
EPS_CLIP = 0.2
K_EPOCH = 4
BATCH_SIZE = 64
UPDATE_TIMESTEP = 2000

# Define the neural network (using both policy and value functions)
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

# Define the PPO Agent
class PPOAgent:
    def __init__(self, state_dim, action_dim):
        self.policy = ActorCritic(state_dim, action_dim)
        self.optimizer = optim.Adam(self.policy.parameters(), lr=LR)
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
        # Convert lists to tensors
        states = torch.FloatTensor(np.array(memory['states']))
        actions = torch.LongTensor(memory['actions']).view(-1,1)
        old_log_probs = torch.FloatTensor(memory['log_probs']).view(-1,1)
        returns = torch.FloatTensor(memory['returns']).view(-1,1)
        advantages = returns - self.policy(states)[1].detach()
        
        # PPO update
        for _ in range(K_EPOCH):
            # Mini-batch training
            for index in BatchSampler(SubsetRandomSampler(range(len(states))), BATCH_SIZE, False):
                sampled_states = states[index]
                sampled_actions = actions[index]
                sampled_old_log_probs = old_log_probs[index]
                sampled_returns = returns[index]
                sampled_advantages = advantages[index]
                
                # Current policy's probabilities and value estimates
                action_probs, state_values = self.policy(sampled_states)
                dist = Categorical(action_probs)
                entropy = dist.entropy().mean()
                new_log_probs = dist.log_prob(sampled_actions.squeeze()).view(-1,1)
                
                # Calculate the ratio
                ratio = torch.exp(new_log_probs - sampled_old_log_probs)
                
                # PPO clipping
                surr1 = ratio * sampled_advantages
                surr2 = torch.clamp(ratio, 1 - EPS_CLIP, 1 + EPS_CLIP) * sampled_advantages
                loss = -torch.min(surr1, surr2) + 0.5 * self.MseLoss(state_values, sampled_returns) - 0.01 * entropy
                
                # Backpropagation and optimization
                self.optimizer.zero_grad()
                loss.mean().backward()
                self.optimizer.step()
        
        # Update the old policy
        self.policy_old.load_state_dict(self.policy.state_dict())

# Main training loop
def main():
    if viz:
        env = gym.make('CartPole-v1', render_mode="human")
    else:
        env = gym.make('CartPole-v1')
    
    state_dim = env.observation_space.shape[0]
    action_dim = env.action_space.n
    agent = PPOAgent(state_dim, action_dim)
    
    max_episodes = 1000
    max_timesteps = 300
    timestep = 0
    
    memory = {'states': [], 'actions': [], 'log_probs': [], 'rewards': [], 'dones': []}
    
    for episode in range(1, max_episodes + 1):
        state, _ = env.reset()
        ep_reward = 0
        for t in range(max_timesteps):
            env.render()
            action, log_prob = agent.select_action(state)
            next_state, reward, terminated, truncated, _ = env.step(action)
            done = terminated or truncated
            
            memory['states'].append(state)
            memory['actions'].append(action)
            memory['log_probs'].append(log_prob.item())
            memory['rewards'].append(reward)
            memory['dones'].append(done)
            
            state = next_state
            ep_reward += reward
            timestep += 1
            
            # Update every certain number of timesteps
            if timestep % UPDATE_TIMESTEP == 0:
                # Estimate the value of the last state
                with torch.no_grad():
                    _, next_value = agent.policy_old(torch.FloatTensor(state).unsqueeze(0))
                returns = agent.compute_returns(memory['rewards'], memory['dones'], next_value.item())
                memory['returns'] = returns
                
                # Perform the update
                agent.update(memory)
                
                # Clear the memory
                memory = {'states': [], 'actions': [], 'log_probs': [], 'rewards': [], 'dones': []}
            
            if done:
                break
        
        print(f"Episode {episode}\tReward: {ep_reward}")
        
        # Terminate if the goal is achieved
        if ep_reward >= 1000:
            print("Solved!")
            break

    # Save the model
    torch.save(agent.policy.state_dict(), './model/cartpole-ppo.pth')
    print("A model is saved")
    env.close()

if __name__ == '__main__':
    main()
