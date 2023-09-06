# baseline
export save=./test/${RANDOM}/
mkdir -p $save

# FashionMNIST
python ./src/test_detection.py --outf $save --dataset $1 --out_dataset $2 --pre_trained_net $3  --num_classes 8 --num_channels 1 --dataroot ./data   2>&1 | tee  $save/log.txt
# cmd line: bash test.sh --dataset FashionMNIST --out_dataset FashionMNIST --pre_trained_net results/joint_confidence_loss/17239/model_epoch_100.pth