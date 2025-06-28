#!/bin/bash

#SBATCH --account=sunwbgt0
#SBATCH --job-name=cccs
#SBATCH --mail-user=xysong@umich.edu
#SBATCH --mail-type=BEGIN,END,FAIL
#SBATCH --nodes=1
#SBATCH --partition=gpu
#SBATCH --gpus=1
#SBATCH --mem-per-gpu=16GB
#SBATCH --time=8:00:00
#SBATCH --output=/scratch/sunwbgt_root/sunwbgt98/xysong/Confident_classifier/out/ccm-0.001.log

export save=./results/joint_confidence_loss/M-0.001/
mkdir -p $save
# SVHN
python ./src/run_joint_confidence.py --dataset MNIST --num_channels 1 --num_classes 8 --batch-size 64 --beta 0.001 --outf $save --dataroot ./data   2>&1 | tee  $save/log.txt
python ./src/test_detection.py --outf $save --dataset MNIST --out_dataset MNIST --pre_trained_net results/joint_confidence_loss/M-0.001/model_epoch_100.pth  --num_classes 8 --num_channels 1 --dataroot ./data   2>&1 | tee  $save/log.txt
