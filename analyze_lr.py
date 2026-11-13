import json
from collections import Counter, defaultdict
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

input_dir = Path("hpc-runs/lr-sweep")
output_dir = Path("reports/cartpole-lr-sweep")

rates = [0.00001, 0.00005, 0.0001, 0.0005, 0.001, 0.005, 0.01]
seeds = range(42, 52)

records = []
curves = defaultdict(list)

for metrics_path in sorted(input_dir.glob("tune-lr-*/metrics.json")):
    run_dir = metrics_path.parent
    config = json.loads((run_dir / "config.json").read_text())
    metrics = json.loads(metrics_path.read_text())

    rate = float(config["learning_rate"])
    seed = int(config["seed"])
    rewards = np.asarray(metrics["evaluation_rewards"], dtype=float)

    if len(rewards) != 20 or not np.isfinite(rewards).all():
        raise ValueError(f"{run_dir.name}: invalid evaluation rewards")

    if (
        config["requested_timesteps"] != 100000
        or config["evaluation_seed"] != 1000
        or config["evaluation_episodes"] != 20
        or metrics["actual_timesteps"] < 100000
    ):
        raise ValueError(f"{run_dir.name}: unexpected experiment settings")

    records.append({
        "learning_rate": rate,
        "seed": seed,
        "mean_reward": float(rewards.mean()),
        "actual_timesteps": metrics["actual_timesteps"],
        "git_commit": config["git_commit"],
    })

    history = pd.read_csv(run_dir / "train.monitor.csv", comment="#")
    steps = history["l"].cumsum().to_numpy()
    smoothed = history["r"].rolling(20, min_periods=1).mean().to_numpy()
    curves[rate].append((steps, smoothed))

# Require exactly one completed model for every rate/seed combination.
expected = Counter({(rate, seed): 1 for rate in rates for seed in seeds})
observed = Counter((r["learning_rate"], r["seed"]) for r in records)

if observed != expected:
    raise ValueError(
        f"Missing combinations: {dict(expected - observed)}\n"
        f"Unexpected or duplicate combinations: {dict(observed - expected)}"
    )

scores = pd.DataFrame(records)
summary = (
    scores.groupby("learning_rate")["mean_reward"]
    .agg(
        runs="count",
        mean_reward="mean",
        std_across_seeds="std",
        min_reward="min",
        max_reward="max",
    )
    .reset_index()
)

output_dir.mkdir(parents=True, exist_ok=True)
scores.to_csv(output_dir / "run_scores.csv", index=False)
summary.to_csv(output_dir / "summary.csv", index=False)

# Use the time range covered by every training log.
all_curves = [curve for group in curves.values() for curve in group]
start = max(steps[0] for steps, _ in all_curves)
stop = min(100000, min(steps[-1] for steps, _ in all_curves))
grid = np.linspace(start, stop, 200)

fig, (left, right) = plt.subplots(1, 2, figsize=(13, 5))

for index, rate in enumerate(rates):
    color = f"C{index}"
    values = scores.loc[scores["learning_rate"] == rate, "mean_reward"]

    # Slight horizontal offsets make overlapping model scores visible.
    positions = rate * np.geomspace(0.96, 1.04, len(values))
    left.scatter(positions, values, color=color, alpha=0.7, s=25)
    left.errorbar(
        rate, values.mean(), yerr=values.std(ddof=1),
        fmt="o", color="black", capsize=4,
    )

    interpolated = [
        np.interp(grid, steps, rewards)
        for steps, rewards in curves[rate]
    ]
    right.plot(
        grid, np.mean(interpolated, axis=0),
        color=color, label=f"{rate:g}",
    )

left.set_xscale("log")
left.set_xticks(rates)
left.set_xticklabels([f"{rate:g}" for rate in rates], rotation=45)
left.set(
    xlabel="Learning rate",
    ylabel="Evaluation episode reward",
    title="Final validation: model scores and mean ± SD",
)

right.set(
    xlabel="Training environment steps",
    ylabel="Training reward: trailing 20-episode average",
    title="Mean training curves across 10 seeds",
)
right.legend(title="Learning rate", fontsize=8)

for axis in (left, right):
    axis.grid(alpha=0.25)

fig.tight_layout()
fig.savefig(output_dir / "comparison.png", dpi=200)

print("Verified all 70 learning-rate/seed combinations.")
print(summary.to_string(index=False))
print(f"\nReport files saved to: {output_dir}")
plt.show()