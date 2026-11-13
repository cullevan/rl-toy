import gymnasium as gym
from stable_baselines3 import PPO

env = gym.make("CartPole-v1", render_mode="human")
model = PPO.load("checkpoints/ppo_cartpole", device="cpu")

try:
    for episode in range(3):
        observation, info = env.reset(seed=100 + episode)
        total_reward = 0.0

        while True:
            action, _ = model.predict(observation, deterministic=True)
            observation, reward, terminated, truncated, info = env.step(action)
            total_reward += reward

            if terminated or truncated:
                print(f"Episode {episode + 1}: reward = {total_reward}")
                break
finally:
    env.close()