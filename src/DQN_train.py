import gym
import numpy as np
import random
from collections import deque
from tensorflow.keras import models, layers, optimizers
import matplotlib.pyplot as plt

# list for visualization
viz = True
scores = []
epsilons = []

# environment setting
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
        self.target_model = build_model()  # 타깃 네트워크
        self.update_target_model()  # 타깃 네트워크 가중치 동기화

    def update_target_model(self):
        # 메인 네트워크의 가중치를 타깃 네트워크로 복사
        self.target_model.set_weights(self.model.get_weights())

    def remember(self, state, action, reward, next_state, done):
        # 경험 메모리 저장
        memory.append((state, action, reward, next_state, done))

    def act(self, state):
        # 상태를 예측하기 전에 2차원 배열로 변환
        state = np.reshape(state, [1, state_size])
        
        if np.random.rand() <= epsilon:
            return random.randrange(action_size)
        
        q_values = self.model.predict(state)  # 2차원으로 변환된 상태 사용
        return np.argmax(q_values[0])

    def save(self, name):
        # 모델 저장
        self.model.save(name)

    def replay(self):
        if len(memory) < batch_size:
            return

        minibatch = random.sample(memory, batch_size)
        for state, action, reward, next_state, done in minibatch:
            target = reward
            if not done:
                # next_state를 2차원 배열로 변환
                next_state = np.reshape(next_state, [1, state_size])
                target = reward + gamma * np.amax(self.target_model.predict(next_state)[0])
            
            # state도 2차원 배열로 변환
            state = np.reshape(state, [1, state_size])
            target_f = self.model.predict(state)
            target_f[0][action] = target

            # 모델 학습
            self.model.fit(state, target_f, epochs=1, verbose=0)

        global epsilon
        if epsilon > epsilon_min:
            epsilon *= epsilon_decay



# train
agent = DQNAgent()
episodes = 1000
target_update_freq = 10  # 타깃 네트워크 갱신 주기

for e in range(episodes):
    state = env.reset()
    if isinstance(state, tuple):
        state = state[0]  # 상태 값만 추출 (튜플의 첫 번째 요소)
    # 상태가 1차원 배열일 경우에만 (1, state_size)로 변환
    if isinstance(state, tuple):
        state = np.reshape(state, [1, state_size])
    total_reward = 0

    for time in range(500):
        # 환경 시각화
        if viz:
            env.render()

        # 에이전트가 행동을 선택
        action = agent.act(state)

        # 환경에서 선택한 행동을 실행
        # 행동 실행 및 반환 값 확인
        result = env.step(action)

        # 반환된 값의 길이에 따라 언패킹
        if len(result) == 4:
            next_state, reward, done, info = result
        else:
            # 반환값의 개수가 다를 경우 추가 처리
            next_state = result[0]  # 상태
            reward = result[1]      # 보상
            done = result[2]        # 종료 여부
            info = result[3] if len(result) > 3 else {}  # 추가 정보 (있는 경우만)


        # 보상 업데이트
        reward = reward if not done else -10
        total_reward += reward

        # 경험 저장
        agent.remember(state, action, reward, next_state, done)

        # 상태 전이
        state = next_state

        # 게임이 끝났을 때
        if done:
            print(f"Episode: {e}/{episodes}, Score: {time}, Epsilon: {epsilon:.2}")
            scores.append(time)  # 점수 기록
            epsilons.append(epsilon)  # epsilon 기록
            break

    # Replay를 통해 학습
    agent.replay()

    # 타깃 네트워크 주기적 업데이트
    if e % target_update_freq == 0:
        agent.update_target_model()

    # 일정 에피소드마다 모델 저장
    if e % 50 == 0:
        agent.save(f"./RL/model/cartpole-dqn-{e}.h5")

# 최종 모델 저장
agent.save("./RL/model/cartpole-dqn-final.h5")

# 학습 결과 시각화
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
