#!/bin/bash
#SBATCH --job-name=cartpole-tune
#SBATCH --partition=normal
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=2
#SBATCH --mem=8G
#SBATCH --time=03:00:00
#SBATCH --propagate=NONE
#SBATCH --array=0-13
#SBATCH --chdir=/home/c/cuonglv/projects/rl-toy
#SBATCH --output=runs/slurm/%x-%A_%a.out

set -euo pipefail

source "$HOME/miniforge3/etc/profile.d/conda.sh"
conda activate rl-toy

export OMP_NUM_THREADS="$SLURM_CPUS_PER_TASK"
export MKL_NUM_THREADS="$SLURM_CPUS_PER_TASK"

LEARNING_RATES=(
    0.00001
    0.00005
    0.0001
    0.0005
    0.001
    0.005
    0.01
)

for OFFSET in 0 1 2 3 4; do
    EXPERIMENT_INDEX=$((SLURM_ARRAY_TASK_ID * 5 + OFFSET))

    RATE_INDEX=$((EXPERIMENT_INDEX / 10))
    TRAIN_SEED=$((42 + EXPERIMENT_INDEX % 10))
    LEARNING_RATE="${LEARNING_RATES[$RATE_INDEX]}"

    RUN_NAME="tune-lr-${SLURM_ARRAY_JOB_ID}-${EXPERIMENT_INDEX}"

    echo "Experiment: $RUN_NAME"
    echo "Learning rate: $LEARNING_RATE"
    echo "Training seed: $TRAIN_SEED"
    hostname

    python -u run_experiment.py \
        --name "$RUN_NAME" \
        --learning-rate "$LEARNING_RATE" \
        --seed "$TRAIN_SEED" \
        --steps 100000 \
        --eval-seed 1000 \
        --eval-episodes 20
done