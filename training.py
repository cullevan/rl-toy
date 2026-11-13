from pathlib import Path

import gymnasium as gym
from stable_baselines3 import PPO
from stable_baselines3.common.evaluation import evaluate_policy
from stable_baselines3.common.monitor import Monitor

# Separate environments for learning and testing.
train_env = Monitor(gym.make("CartPole-v1"))
eval_env = Monitor(gym.make("CartPole-v1"))

model = PPO(
    "MlpPolicy",
    train_env,
    device="cpu",
    seed=42,
    verbose=1,
)

try:
    print("Training...")
    model.learn(total_timesteps=20000)

    Path("checkpoints").mkdir(exist_ok=True)
    model.save("checkpoints/ppo_cartpole")

    eval_env.reset(seed=100)
    mean_reward, std_reward = evaluate_policy(
        model,
        eval_env,
        n_eval_episodes=10,
        deterministic=True,
    )

    print(f"Evaluation reward: {mean_reward:.1f} +/- {std_reward:.1f}")
    print("Saved model: checkpoints/ppo_cartpole.zip")
finally:
    train_env.close()
    eval_env.close()