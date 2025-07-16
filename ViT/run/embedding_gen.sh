#! /bin/bash

#SBATCH --job-name=embedding-gen
#SBATCH --partition=gpuq-a30
#SBATCH --nodelist=gpu[002]

source /home/s25piteam/miniconda3/bin/activate
conda activate torch116

srun --unbuffered python /home/s25piteam/UNLV-histopathology/ViT/embed_gen/embedding_gen.py
