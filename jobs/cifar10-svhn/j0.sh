#!/bin/bash

#SBATCH --account=sunwbgt98
#SBATCH --job-name=BASE-CS0
#SBATCH --mail-user=xysong@umich.edu
#SBATCH --mail-type=BEGIN,END,FAIL
#SBATCH --nodes=1
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1
#SBATCH --mem-per-gpu=14GB
#SBATCH --time=24:00:00
#SBATCH --output=/scratch/sunwbgt_root/sunwbgt98/xysong/jobs/BASE-CS0.log

export save=./results/joint_confidence_loss/CS-0.0001/
mkdir -p $save
# SVHN
python ./src/run_joint_confidence.py --dataset CIFAR10-SVHN --num_classes 10 --batch-size 64 --beta 0.0001 --outf $save --dataroot ./data   2>&1 | tee  $save/log.txt
