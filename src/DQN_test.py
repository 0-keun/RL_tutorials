import gym
import numpy as np
from tensorflow.keras.models import load_model

# 환경 설정
env = gym.make("CartPole-v1", render_mode="human")
state_size = env.observation_space.shape[0]
action_size = env.action_space.n

# 학습된 모델 불러오기
model = load_model("./model/cartpole-dqn-100.h5")

def act(state):
    # 상태를 예측하기 전에 2차원 배열로 변환
    state = np.reshape(state, [1, state_size])
    q_values = model.predict(state)
    return np.argmax(q_values[0])

# 테스트 에피소드 수
test_episodes = 3

for e in range(test_episodes):
    state = env.reset()
    if isinstance(state, tuple):
        state = state[0]  # 상태 값만 추출 (튜플의 첫 번째 요소)
    total_reward = 0

    for time in range(500):
        env.render()  # 환경 시각화
        
        # 에이전트가 학습된 모델로 행동 선택
        action = act(state)
        
        # 환경에서 행동을 실행
        result = env.step(action)

        # 반환되는 값의 길이에 따라 처리 (terminated와 truncated 확인)
        if len(result) == 4:
            next_state, reward, done, info = result
        elif len(result) == 5:
            next_state, reward, terminated, truncated, info = result
            done = terminated or truncated  # 둘 중 하나라도 True면 done 처리

        total_reward += reward

        # 상태 전이
        state = next_state

        if done:
            print(f"Episode: {e+1}/{test_episodes}, Score: {time}")
            break


env.close()
