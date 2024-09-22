import gym
import numpy as np
import random
from collections import deque
from tensorflow.keras import models, layers, optimizers
import matplotlib.pyplot as plt

# Lists for visualization
viz = True
scores = []
epsilons = []

# Environment setting
if viz:
    env = gym.make("CartPole-v1", render_mode="human")
else:
    env = gym.make("CartPole-v1")

# Hyperparameters
state_size = env.observation_space.shape[0]
action_size = env.action_space.n
learning_rate = 0.001
gamma = 0.95  # Discount factor
epsilon = 1.0  # Exploration rate
epsilon_min = 0.01
epsilon_decay = 0.995
batch_size = 64
memory = deque(maxlen=2000)  # Experience replay memory

# Define the DQN model
def build_model():
    model = models.Sequential()
    model.add(layers.Dense(24, input_dim=state_size, activation='relu'))
    model.add(layers.Dense(24, activation='relu'))
    model.add(layers.Dense(action_size, activation='linear'))
    model.compile(loss='mse', optimizer=optimizers.Adam(learning_rate=learning_rate))
    return model

# Define the agent
class DQNAgent:
    def __init__(self):
        self.model = build_model()
        self.target_model = build_model()  # Target network
        self.update_target_model()  # Synchronize target network weights

    def update_target_model(self):
        # Copy weights from the main network to the target network
        self.target_model.set_weights(self.model.get_weights())

    def remember(self, state, action, reward, next_state, done):
        # Store experience in memory
        memory.append((state, action, reward, next_state, done))

    def act(self, state):
        # Reshape the state into a 2D array before prediction
        state = np.reshape(state, [1, state_size])
        
        if np.random.rand() <= epsilon:
            return random.randrange(action_size)
        
        q_values = self.model.predict(state)  # Use the reshaped state
        return np.argmax(q_values[0])

    def save(self, name):
        # Save the model
        self.model.save(name)

    def replay(self):
        if len(memory) < batch_size:
            return

        minibatch = random.sample(memory, batch_size)
        for state, action, reward, next_state, done in minibatch:
            target = reward
            if not done:
                # Reshape next_state into a 2D array
                next_state = np.reshape(next_state, [1, state_size])
                target = reward + gamma * np.amax(self.target_model.predict(next_state)[0])
            
            # Reshape state into a 2D array
            state = np.reshape(state, [1, state_size])
            target_f = self.model.predict(state)
            target_f[0][action] = target

            # Train the model
            self.model.fit(state, target_f, epochs=1, verbose=0)

        global epsilon
        if epsilon > epsilon_min:
            epsilon *= epsilon_decay

# Train the agent
agent = DQNAgent()
episodes = 1000
target_update_freq = 10  # Target network update frequency

for e in range(episodes):
    state = env.reset()
    if isinstance(state, tuple):
        state = state[0]  # Extract only the state value (first element of the tuple)
    # Only reshape if state is a tuple (for Gym versions >=0.26)
    if isinstance(state, tuple):
        state = np.reshape(state, [1, state_size])
    total_reward = 0

    for time in range(500):
        # Render the environment
        if viz:
            env.render()

        # Agent selects an action
        action = agent.act(state)

        # Execute the selected action in the environment
        # Handle different return value lengths (terminated and truncated)
        result = env.step(action)

        # Unpack the returned values based on their length
        if len(result) == 4:
            next_state, reward, done, info = result
        else:
            # Additional processing if the number of returned values differs
            next_state = result[0]  # Next state
            reward = result[1]      # Reward
            done = result[2]        # Done flag
            info = result[3] if len(result) > 3 else {}  # Additional info (if available)

        # Update reward
        reward = reward if not done else -10
        total_reward += reward

        # Store experience
        agent.remember(state, action, reward, next_state, done)

        # Transition to the next state
        state = next_state

        # If the episode is done
        if done:
            print(f"Episode: {e}/{episodes}, Score: {time}, Epsilon: {epsilon:.2}")
            scores.append(time)      # Record score
            epsilons.append(epsilon) # Record epsilon
            break

    # Replay to train the model
    agent.replay()

    # Periodically update the target network
    if e % target_update_freq == 0:
        agent.update_target_model()

    # Save the model at regular intervals
    if e % 50 == 0:
        agent.save(f"./RL/model/cartpole-dqn-{e}.h5")

# Save the final model
agent.save("./RL/model/cartpole-dqn-final.h5")

# Visualize training results
plt.figure(figsize=(12, 5))
plt.subplot(1, 2, 1)
plt.plot(scores)
plt.title('Scores per Episode')
plt.xlabel('Episode')
plt.ylabel('Score')

plt.subplot(1, 2, 2)
plt.plot(epsilons)
plt.title('Epsilon Decay')
plt.xlabel('Episode')
plt.ylabel('Epsilon')
plt.show()
