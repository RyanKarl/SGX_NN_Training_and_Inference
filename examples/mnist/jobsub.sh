#!/bin/csh
#$ -q gpu        # Specify queue (use ‘debug’ for development)
#$ -l gpu_card=1
#$ -N job_name         # Specify job name
#$ -t 1-5:1

module load pytorch

python test_main_private_cifar.py ${SGE_TASK_ID}
