#!/bin/bash

# Path configuration
# conda activate OoD

# Environment Configuration
export PYTHONPATH=$PYTHONPATH$:`pwd`

# Batch job submission

# SVHN Baseline
# sbatch jobs/svhn/j0.sh
# sbatch jobs/svhn/j1.sh
# sbatch jobs/svhn/j2.sh
# sbatch jobs/svhn/j3.sh
# sbatch jobs/svhn/j4.sh
# sbatch jobs/svhn/j5.sh

# CIFAR10-SVHN
sbatch jobs/cifar10-svhn/j0.sh
sbatch jobs/cifar10-svhn/j1.sh
sbatch jobs/cifar10-svhn/j2.sh
sbatch jobs/cifar10-svhn/j3.sh
sbatch jobs/cifar10-svhn/j4.sh
sbatch jobs/cifar10-svhn/j5.sh
sbatch jobs/cifar10-svhn/j6.sh
sbatch jobs/cifar10-svhn/j7.sh
sbatch jobs/cifar10-svhn/j8.sh

# FashionMNIST
# sbatch jobs/fashionmnist/j0.sh
# sbatch jobs/fashionmnist/j1.sh
# sbatch jobs/fashionmnist/j2.sh
# sbatch jobs/fashionmnist/j3.sh
# sbatch jobs/fashionmnist/j4.sh
# sbatch jobs/fashionmnist/j5.sh

# MNIST
# sbatch jobs/mnist/j0.sh
# sbatch jobs/mnist/j1.sh
# sbatch jobs/mnist/j2.sh
# sbatch jobs/mnist/j3.sh
# sbatch jobs/mnist/j4.sh

# MNIST-FashionMNIST
# sbatch jobs/mnist-fashionmnist/j0.sh
# sbatch jobs/mnist-fashionmnist/j1.sh
# sbatch jobs/mnist-fashionmnist/j2.sh
# sbatch jobs/mnist-fashionmnist/j3.sh

squeue -u xysong