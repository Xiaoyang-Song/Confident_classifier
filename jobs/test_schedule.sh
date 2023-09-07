# Run test bash scripts
# SVHN
bash scripts/test.sh  SVHN SVHN results/joint_confidence_loss/SHVN-0.001-28561/model_epoch_100.pth 8 3 SVHN-0.001-28561
bash scripts/test.sh  SVHN SVHN results/joint_confidence_loss/SHVN-0.01-30566/model_epoch_100.pth 8 3 SVHN-0.01-30566
bash scripts/test.sh  SVHN SVHN results/joint_confidence_loss/SHVN-0.1-15528/model_epoch_100.pth 8 3 SVHN-0.1-15528
bash scripts/test.sh  SVHN SVHN results/joint_confidence_loss/SHVN-1-32674/model_epoch_100.pth 8 3 SVHN-1-32674

# CIFAR10-SVHN
bash scripts/test.sh  CIFAR10-SVHN CIFAR10-SVHN results/joint_confidence_loss/CS-0.001-9623/model_epoch_100.pth 10 3 CS-0.001-9623
bash scripts/test.sh  CIFAR10-SVHN CIFAR10-SVHN results/joint_confidence_loss/CS-0.01-14220/model_epoch_100.pth 10 3 CS-0.01-14220
bash scripts/test.sh  CIFAR10-SVHN CIFAR10-SVHN results/joint_confidence_loss/CS-0.1-6513/model_epoch_100.pth 10 3 CS-0.1-6513
bash scripts/test.sh  CIFAR10-SVHN CIFAR10-SVHN results/joint_confidence_loss/CS-1-10745/model_epoch_100.pth 10 3 CS-1-10745