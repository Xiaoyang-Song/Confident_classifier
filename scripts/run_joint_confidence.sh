# baseline
export save=./results/joint_confidence_loss/${RANDOM}/
mkdir -p $save
# python ./src/run_joint_confidence.py --outf $save --dataroot ./data   2>&1 | tee  $save/log.txt
python ./src/run_joint_confidence.py --dataset cifar10 --outf $save --dataroot ./data   2>&1 | tee  $save/log.txt
