import json
from pathlib import Path
from statistics import mean, pstdev

import matplotlib.pyplot as plt
import pandas as pd

# Include completed experiments that have evaluation results.
run_dirs = sorted(
    path.parent for path in Path("runs").glob("*/metrics.json")
)

if not run_dirs:
    raise SystemExit("No completed experiments found in runs/.")

fig, ax = plt.subplots(figsize=(10, 6))

for run_dir in run_dirs:
    config = json.loads(
        (run_dir / "config.json").read_text(encoding="utf-8")
    )
    metrics = json.loads(
        (run_dir / "metrics.json").read_text(encoding="utf-8")
    )

    # Monitor's first line contains metadata beginning with "#".
    history = pd.read_csv(run_dir / "train.monitor.csv", comment="#")

    # "l" is episode length; "r" is total episode reward.
    steps = history["l"].cumsum()
    rewards = history["r"]
    smoothed = rewards.rolling(window=20, min_periods=1).mean()

    label = f"{run_dir.name} (seed {config['seed']})"
    line, = ax.plot(steps, smoothed, label=label)
    ax.plot(
        steps, rewards,
        color=line.get_color(), alpha=0.15, linewidth=0.7,
    )

    scores = metrics["evaluation_rewards"]
    print(
        f"{run_dir.name}: seed={config['seed']}, "
        f"steps={metrics['actual_timesteps']}, "
        f"evaluation={mean(scores):.1f} +/- {pstdev(scores):.1f}, "
        f"uncommitted_changes={config['git_dirty']}"
    )

ax.set(
    xlabel="Training environment steps",
    ylabel="Training episode reward",
    title="CartPole learning curves — trailing average of 20 episodes",
)
ax.legend()
ax.grid(alpha=0.3)
fig.tight_layout()

output = Path("runs") / "comparison.png"
fig.savefig(output, dpi=150)
print(f"Saved plot: {output}")
plt.show()