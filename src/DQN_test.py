import gym
import numpy as np
from tensorflow.keras.models import load_model

# Toggle visualization On / Off
viz = True

# Environment setting
if viz:
    env = gym.make("CartPole-v1", render_mode="human")
else:
    env = gym.make("CartPole-v1")

state_size = env.observation_space.shape[0]
action_size = env.action_space.n

# Load the trained model
model = load_model("./model/cartpole-dqn-100.h5")

def act(state):
    # Reshape the state into a 2D array before prediction
    state = np.reshape(state, [1, state_size])
    q_values = model.predict(state)
    return np.argmax(q_values[0])

# Number of test episodes
test_episodes = 3

for e in range(test_episodes):
    state = env.reset()
    if isinstance(state, tuple):
        state = state[0]  # Extract only the state value (first element of the tuple)
    total_reward = 0

    for time in range(500):
        if viz:
            env.render()  # Render the environment

        # Agent selects an action using the trained model
        action = act(state)

        # Execute the action in the environment
        result = env.step(action)

        # Handle the returned values based on their length (check for terminated and truncated)
        if len(result) == 4:
            next_state, reward, done, info = result
        elif len(result) == 5:
            next_state, reward, terminated, truncated, info = result
            done = terminated or truncated  # If either is True, set done to True

        total_reward += reward

        # Transition to the next state
        state = next_state

        if done:
            print(f"Episode: {e+1}/{test_episodes}, Score: {time}")
            break

env.close()
