#!/bin/bash

#SBATCH --account=jhjin1
#SBATCH --job-name=ccsv
#SBATCH --mail-user=xysong@umich.edu
#SBATCH --mail-type=BEGIN,END,FAIL
#SBATCH --nodes=1
#SBATCH --partition=gpu
#SBATCH --gpus=1
#SBATCH --mem-per-gpu=16GB
#SBATCH --time=8:00:00
#SBATCH --output=/scratch/sunwbgt_root/sunwbgt98/xysong/Confident_classifier/out/ccsv-0.00001.log

export save=./results/joint_confidence_loss/SV-0.00001/
mkdir -p $save
# SVHN
python ./src/run_joint_confidence.py --dataset SVHN --num_classes 8 --batch-size 64 --beta 0.00001 --outf $save --dataroot ./data   2>&1 | tee  $save/log.txt
python ./src/test_detection.py --outf $save --dataset SVHN --out_dataset SVHN --pre_trained_net results/joint_confidence_loss/SV-0.00001/model_epoch_100.pth  --num_classes 8 --num_channels 3 --dataroot ./data   2>&1 | tee  $save/log.txt
