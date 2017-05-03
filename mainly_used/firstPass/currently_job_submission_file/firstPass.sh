#!/bin/bash -l

#$ -l h_rt=90:00:00 -o out.txt -e err.txt -m e -V -pe omp 28

module load cuda/8.0
module load cudnn/5.1
module load python/3.6.0
module load tensorflow/r1.0_python-3.6.0

python change_percentage.py
python firstpassCNN.py

