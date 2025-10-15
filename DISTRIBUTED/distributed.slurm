#!/bin/bash
#------------------------------------------------------
# Example SLURM job script with SBATCH requesting GPUs
#------------------------------------------------------
#SBATCH -J distributed            # Job name
#SBATCH -o logs/distributed_bert_%j.o       # Name of stdout output file(%j expands to jobId)
#SBATCH -e logs/distributed_bert_%j.e       # Name of stderr output file(%j expands to jobId)
#SBATCH --gres=gpu:a100:2   # Request 1 GPU of 2 available on an average A100 node
#SBATCH -c 32               # Cores per task requested
#SBATCH -t 02:00:00         # Run time (hh:mm:ss) - 10 min
#SBATCH --mem-per-cpu=3G    # Memory per core demandes (96 GB = 3GB * 32 cores)
#SBATCH -N 1                # Total # of nodes
#SBATCH --ntasks-per-node=2 

source $STORE/mypython/bin/activate

export TRANSFORMERS_CACHE=$STORE/.cache/huggingface
export HF_DATASETS_CACHE=$STORE/.cache/huggingface/datasets

date

echo "Running on host: $(hostname)"
nvidia-smi

srun $STORE/mypython/bin/python train_distributed.py

date
