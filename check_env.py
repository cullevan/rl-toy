import gymnasium as gym

env = gym.make("CartPole-v1", render_mode="human")

observation, info = env.reset(seed=42)
env.action_space.seed(42)

print("Initial observation:", observation)
print("Action space:", env.action_space)

total_reward = 0.0

try:
    for step in range(200):
        action = env.action_space.sample()

        observation, reward, terminated, truncated, info = env.step(action)
        total_reward += reward

        if terminated or truncated:
            print(f"Episode ended after {step + 1} steps.")
            break

    print("Total reward:", total_reward)
finally:
    env.close()