#!/bin/bash -l

#$ -l h_rt=300:00:00 -o outNew.txt -e errNew.txt -m e -V -pe omp 28

module load cuda/8.0
module load cudnn/5.1
module load python/3.6.0
module load tensorflow/r1.0_python-3.6.0

python firstpassProcessing.py
python firstpassCNNnew.py

