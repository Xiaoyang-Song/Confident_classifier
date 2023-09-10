#!/bin/bash

#SBATCH --account=alkontar1
#SBATCH --job-name=CC-MFM0
#SBATCH --mail-user=xysong@umich.edu
#SBATCH --mail-type=BEGIN,END,FAIL
#SBATCH --nodes=1
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1
#SBATCH --mem-per-gpu=12GB
#SBATCH --time=6:00:00
#SBATCH --output=/scratch/sunwbgt_root/sunwbgt98/xysong/jobs/CC-MFM-0.0001.log

export save=./results/joint_confidence_loss/MFM-0.0001/
mkdir -p $save
# SVHN
python ./src/run_joint_confidence.py --dataset MNIST-FashionMNIST --num_classes 10 --num_channels 1 --batch-size 64 --beta 0.0001 --outf $save --dataroot ./data   2>&1 | tee  $save/log.txt
python ./src/test_detection.py --outf $save --dataset MNIST-FashionMNIST --out_dataset MNIST-FashionMNIST --pre_trained_net results/joint_confidence_loss/MFM-0.0001/model_epoch_100.pth  --num_classes 10 --num_channels 1 --dataroot ./data   2>&1 | tee  $save/log.txt
