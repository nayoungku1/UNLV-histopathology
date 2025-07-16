#! /bin/bash

#SBATCH --job-name=classification
#SBATCH --partition=gpuq-a30
#SBATCH --nodelist=gpu[001]

source /home/s25piteam/miniconda3/bin/activate
conda activate torch116

srun --unbuffered python \
	/home/s25piteam/UNLV-histopathology/ViT/classification/classifier.py \
	--num_epochs 100 \
	--num_fc_layers 3 \
	--model_type resnet


