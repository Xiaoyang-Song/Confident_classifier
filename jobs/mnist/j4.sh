#!/bin/bash

#SBATCH --account=sunwbgt98
#SBATCH --job-name=CC-M4
#SBATCH --mail-user=xysong@umich.edu
#SBATCH --mail-type=BEGIN,END,FAIL
#SBATCH --nodes=1
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1
#SBATCH --mem-per-gpu=12GB
#SBATCH --time=12:00:00
#SBATCH --output=/scratch/sunwbgt_root/sunwbgt98/xysong/jobs/CC-M-0.00001.log

export save=./results/joint_confidence_loss/M-0.00001/
mkdir -p $save
# SVHN
python ./src/run_joint_confidence.py --dataset MNIST --num_classes 5 --num_channels 1 --batch-size 64 --beta 0.00001 --outf $save --dataroot ./data   2>&1 | tee  $save/log.txt
python ./src/test_detection.py --outf $save --dataset MNIST --out_dataset MNIST --pre_trained_net results/joint_confidence_loss/M-0.00001/model_epoch_100.pth  --num_classes 5 --num_channels 1 --dataroot ./data   2>&1 | tee  $save/log.txt
