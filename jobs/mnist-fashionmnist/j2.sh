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
#SBATCH --output=/scratch/sunwbgt_root/sunwbgt98/xysong/Confident_classifier/out/ccmfm-0.01.log


export save=./results/joint_confidence_loss/MFM-0.01/
mkdir -p $save
# SVHN
python ./src/run_joint_confidence.py --dataset MNIST-FashionMNIST --num_classes 10 --num_channels 1 --batch-size 64 --beta 0.01 --outf $save --dataroot ./data   2>&1 | tee  $save/log.txt
python ./src/test_detection.py --outf $save --dataset MNIST-FashionMNIST --out_dataset MNIST-FashionMNIST --pre_trained_net results/joint_confidence_loss/MFM-0.01/model_epoch_100.pth  --num_classes 10 --num_channels 1 --dataroot ./data   2>&1 | tee  $save/log.txt
