#!/bin/bash
#SBATCH --job-name=cartpole-cpu
#SBATCH --partition=normal
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=2
#SBATCH --mem=8G
#SBATCH --time=00:30:00
#SBATCH --propagate=NONE
#SBATCH --chdir=/home/c/cuonglv/projects/rl-toy
#SBATCH --output=runs/slurm/%x-%j.out

set -euo pipefail

source "$HOME/miniforge3/etc/profile.d/conda.sh"
conda activate rl-toy

export OMP_NUM_THREADS="$SLURM_CPUS_PER_TASK"
export MKL_NUM_THREADS="$SLURM_CPUS_PER_TASK"

echo "Job ID: $SLURM_JOB_ID"
hostname

python -u run_experiment.py \
    --name "cluster-batch-$SLURM_JOB_ID" \
    --seed 42 \
    --steps 20000