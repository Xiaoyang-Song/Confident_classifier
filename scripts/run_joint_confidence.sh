# baseline
export save=./results/joint_confidence_loss/${RANDOM}/
mkdir -p $save
# python ./src/run_joint_confidence.py --outf $save --dataroot ./data   2>&1 | tee  $save/log.txt

# SVHN
# python ./src/run_joint_confidence.py --dataset SVHN --num_classes 8 --outf $save --dataroot ./data   2>&1 | tee  $save/log.txt

# FashionMNIST
python ./src/run_joint_confidence.py --dataset FashionMNIST --num_classes 8 --num_channels 1 --batch_size 256 --outf $save --dataroot ./data   2>&1 | tee  $save/log.txt
