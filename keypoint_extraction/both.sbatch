#!/usr/bin/env bash

#SBATCH --job-name=both
#SBATCH --mem=64G
#SBATCH --partition=leinegpu
#SBATCH --gres=gpu:a100-40g
#SBATCH --time=18:30:00
#SBATCH --cpus-per-task=8


module load Python/3.10.4


source ~/myenv/bin/activate


# Fix the environment variable
unset RANK

export CUDA_VISIBLE_DEVICES=0

# Set the environment variable for device-side assertions
export TORCH_USE_CUDA_DSA=1

# Export additional variables for your CUDA path, etc.
export CUDAToolkit_ROOT=$CUDA_HOME
export CUDA_HOME=${CUDA_HOME:-/path/to/cuda}   # HPC might define CUDA_HOME differently
export USE_CUDA=1
export USE_NINJA=1


# change to your path
cd /home/leinehome/mh-hannover.local/bakhshaj/tic_classifcation
export PYTHONPATH=$PYTHONPATH:$(pwd)

CONFIG_PATH="./configs/config_both.yaml" # change to your config path

python -u ./src/data/experiment.py \
        --config "$CONFIG_PATH" 
