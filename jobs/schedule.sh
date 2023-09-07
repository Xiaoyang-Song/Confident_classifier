#!/bin/bash

# Path configuration
# conda activate OoD

# Environment Configuration
export PYTHONPATH=$PYTHONPATH$:`pwd`

# Batch job submission

# SVHN Baseline
sbatch jobs/svhn/j0.sh
sbatch jobs/svhn/j1.sh
sbatch jobs/svhn/j2.sh
sbatch jobs/svhn/j3.sh

squeue -u xysong