#!/bin/bash
#SBATCH --job-name=llava_tune
#SBATCH --partition=gpu
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=4
#SBATCH --gpus-per-node=4
#SBATCH --gres=gpu:4
#SBATCH --cpus-per-task=8
#SBATCH --mem=0
#SBATCH --time=24:00:00
#SBATCH --output=logs/%x_%j.out

module load cuda/12.1
source ~/.bashrc
conda activate rag

srun torchrun --nproc_per_node=4 training/fine_tune_llava.py --model_name liuhaotian/llava-v1.5-7b --train_jsonl data/train.jsonl --epochs 3 --batch_size 1