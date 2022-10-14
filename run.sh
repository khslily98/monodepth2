#!/bin/bash

#SBATCH -J temp
#SBATCH -p part1
#SBATCH -N 1
#SBATCH -n 4
#SBATCH -o ./log/stdout/%x_stdout_%j.txt
#SBATCH -e ./log/stderr/%x_stderr_%j.txt
#SBATCH --gres=gpu

# source /home/cilab9/anaconda3/etc/profile.d/conda.sh
# conda activate odi_env

mkdir ./backup/$SLURM_JOB_NAME
cp trainer.py ./backup/$SLURM_JOB_NAME
cp datasets/[A-Za-z]*.py ./backup/$SLURM_JOB_NAME
cp networks/[A-Za-z]*.py ./backup/$SLURM_JOB_NAME
# python train.py --model_name $SLURM_JOB_NAME
python train.py --model_name $SLURM_JOB_NAME --load_weights_folder ./checkpoints/self_001_pretrain/models/weights_24 
# finetuned_24_100
# python eval_uap3.py --source_model 1 --target_model 2
# python eval_uap3.py --source_model 1 --target_model 3
# python eval_uap3.py --source_model 1 --target_model 4
# python eval_uap3.py --source_model 1 --target_model 5
# CUDA_VISIBLE_DEVICES=2 python eval_uap3.py --source_model 1 --target_model 6

# conda deactivate