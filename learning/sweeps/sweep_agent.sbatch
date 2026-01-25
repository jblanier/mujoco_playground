#!/bin/bash
#SBATCH --job-name=wandb-sweep
#SBATCH --partition=PARTITION_NAME
#SBATCH --nodelist=NODE_NAME
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=72:00:00
#SBATCH --output=slurm-%j-sweep.out
#SBATCH --error=slurm-%j-sweep.err

# Usage:
#   sbatch --partition=<partition> --nodelist=<node> sweep_agent.sbatch <sweep_id>
#
# Example:
#   sbatch --partition=gpu --nodelist=gpu-node-01 sweep_agent.sbatch user/mjxrl-sweep/abc123

SWEEP_ID=$1

if [ -z "$SWEEP_ID" ]; then
    echo "Error: sweep_id required"
    echo "Usage: sbatch --partition=<partition> --nodelist=<node> sweep_agent.sbatch <sweep_id>"
    exit 1
fi

cd /home/jb/git/mujoco_playground/learning

# Activate conda environment (uncomment and modify path as needed)
# source ~/anaconda3/bin/activate brax
# source ~/miniconda3/bin/activate brax

echo "Starting wandb agent for sweep: ${SWEEP_ID}"
echo "Node: $(hostname)"
echo "GPU: $(nvidia-smi --query-gpu=name --format=csv,noheader)"

wandb agent ${SWEEP_ID}
