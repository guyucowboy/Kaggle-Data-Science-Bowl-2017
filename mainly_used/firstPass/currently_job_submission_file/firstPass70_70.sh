#!/bin/bash

#$ -l h_rt=400:00:00 -o out70_70.txt -e err70_70.txt -m e -V -pe omp 16

module load cuda/8.0
module load cudnn/5.1
module load python/3.6.0
module load tensorflow/r1.0_python-3.6.0

python firstpassCNN70_70.py


