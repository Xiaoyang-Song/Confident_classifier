# baseline
export save=./test/$6/
mkdir -p $save

# FashionMNIST
python ./src/test_detection.py --outf $save --dataset $1 --out_dataset $2 --pre_trained_net $3  --num_classes $4 --num_channels $5 --dataroot ./data   2>&1 | tee  $save/log.txt
# cmd line: bash scripts/test.sh  FashionMNIST  FashionMNIST results/joint_confidence_loss/17239/model_epoch_100.pth

# SVHN
# python ./src/test_detection.py --outf $save --dataset $1 --out_dataset $2 --pre_trained_net $3  --num_classes 8 --num_channels 3 --dataroot ./data   2>&1 | tee  $save/log.txt