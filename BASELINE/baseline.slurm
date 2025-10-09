#!/bin/bash
#------------------------------------------------------
# Example SLURM job script with SBATCH requesting GPUs
#------------------------------------------------------
#SBATCH -J baseline_bert            # Job name
#SBATCH -o logs/baseline_bert_%j.o       # Name of stdout output file(%j expands to jobId)
#SBATCH -e logs/baseline_bert_%j.e       # Name of stderr output file(%j expands to jobId)
#SBATCH --gres=gpu:a100:1   # Request 1 GPU of 2 available on an average A100 node
#SBATCH -c 32               # Cores per task requested
#SBATCH -t 02:00:00         # Run time (hh:mm:ss) - 10 min
#SBATCH --mem-per-cpu=3G    # Memory per core demandes (96 GB = 3GB * 32 cores)

echo "Running on host: $(hostname)"
nvidia-smi

date
bash run_baseline.sh
date
