#!/bin/bash

#SBATCH --account=sunwbgt98
#SBATCH --job-name=BASE-CC2
#SBATCH --mail-user=xysong@umich.edu
#SBATCH --mail-type=BEGIN,END,FAIL
#SBATCH --nodes=1
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1
#SBATCH --mem-per-gpu=14GB
#SBATCH --time=12:00:00
#SBATCH --output=/scratch/sunwbgt_root/sunwbgt98/xysong/jobs/BASE-CC2.log

export save=./results/joint_confidence_loss/SHVN-0.1-${RANDOM}/
mkdir -p $save
# SVHN
python ./src/run_joint_confidence.py --dataset SVHN --num_classes 8 --batch-size 64 --beta 0.1 --outf $save --dataroot ./data   2>&1 | tee  $save/log.txt
