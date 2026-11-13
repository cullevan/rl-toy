import argparse
import json
import subprocess
from pathlib import Path
import sys

import gymnasium as gym
from stable_baselines3 import PPO
from stable_baselines3.common.evaluation import evaluate_policy
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.callbacks import CheckpointCallback

parser = argparse.ArgumentParser()
parser.add_argument("--name", required=True)
parser.add_argument("--seed", type=int, default=42)
parser.add_argument("--steps", type=int, default=20000)
parser.add_argument("--learning-rate", type=float, default=0.0003)
parser.add_argument("--eval-seed", type=int, default=1000)
parser.add_argument("--eval-episodes", type=int, default=20)
args = parser.parse_args()

if Path(args.name).name != args.name or args.name in {".", ".."}:
    parser.error("--name must be a single folder name")
if args.steps <= 0:
    parser.error("--steps must be positive")

run_dir = Path("runs") / args.name
run_dir.mkdir(parents=True, exist_ok=False)

config = {
    "environment": "CartPole-v1",
    "algorithm": "PPO",
    "policy": "MlpPolicy",
    "device": "cpu",
    "seed": args.seed,
    "requested_timesteps": args.steps,
    "learning_rate": args.learning_rate,
    "evaluation_seed": args.eval_seed,
    "evaluation_episodes": args.eval_episodes,
    "git_commit": subprocess.check_output(
        ["git", "rev-parse", "HEAD"], text=True
    ).strip(),
    "git_dirty": bool(subprocess.check_output(
        ["git", "status", "--porcelain"], text=True
    ).strip()),
    "checkpoint_frequency": 5000
}
(run_dir / "config.json").write_text(
    json.dumps(config, indent=2), encoding="utf-8"
)
(run_dir / "requirements.txt").write_text(
    subprocess.check_output(
        [sys.executable, "-m", "pip", "freeze"], text=True
    ),
    encoding="utf-8",
)

train_env = Monitor(
    gym.make(config["environment"]),
    filename=str(run_dir/"train.monitor.csv"),
)
eval_env = Monitor(gym.make(config["environment"]))

try:
    model = PPO(
        config["policy"],
        train_env,
        learning_rate=config["learning_rate"],
        device=config["device"],
        seed=args.seed,
        verbose=1,
    )

    checkpoint_callback = CheckpointCallback(
        save_freq=config["checkpoint_frequency"],
        save_path=str(run_dir/"checkpoints"),
        name_prefix="ppo",
        verbose=2,
    )

    model.learn(
        total_timesteps=args.steps,
        callback=checkpoint_callback,
    )
    model.save(str(run_dir / "model"))

    eval_env.reset(seed=config["evaluation_seed"])
    rewards, lengths = evaluate_policy(
        model, eval_env,
        n_eval_episodes=config["evaluation_episodes"],
        deterministic=True,
        return_episode_rewards=True,
    )

    metrics = {
        "actual_timesteps": int(model.num_timesteps),
        "evaluation_rewards": [float(r) for r in rewards],
        "evaluation_lengths": [int(n) for n in lengths],
        "mean_reward": float(sum(rewards) / len(rewards)),
    }
    (run_dir / "metrics.json").write_text(
        json.dumps(metrics, indent=2), encoding="utf-8"
    )
    print(f"Mean evaluation reward: {metrics['mean_reward']:.1f}")
    print(f"Results saved to: {run_dir}")
finally:
    train_env.close()
    eval_env.close()