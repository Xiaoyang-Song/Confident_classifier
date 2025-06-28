#!/bin/bash

#SBATCH --account=jhjin1
#SBATCH --job-name=ccfm
#SBATCH --mail-user=xysong@umich.edu
#SBATCH --mail-type=BEGIN,END,FAIL
#SBATCH --nodes=1
#SBATCH --partition=gpu
#SBATCH --gpus=1
#SBATCH --mem-per-gpu=16GB
#SBATCH --time=8:00:00
#SBATCH --output=/scratch/sunwbgt_root/sunwbgt98/xysong/Confident_classifier/out/fashionmnist.log

export save=./results/joint_confidence_loss/FashionMNIST/
mkdir -p $save

python ./src/run_joint_confidence.py --dataset FashionMNIST --num_classes 8 --num_channels 1 \
    --batch-size 64 --beta 1 --outf $save --dataroot ./data   2>&1 | tee  $save/log.txt
python ./src/test_detection.py --outf $save --dataset FashionMNIST --out_dataset FashionMNIST \
    --pre_trained_net results/joint_confidence_loss/FashionMNIST/model_epoch_100.pth  \
    --num_classes 8 --num_channels 1 --dataroot ./data   2>&1 | tee  $save/log.txt
