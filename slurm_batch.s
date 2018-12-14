#!/bin/bash
#
#SBATCH --job-name=multi-neural-styles
#SBATCH --nodes=1
#SBATCH --cpus-per-task=1
#SBATCH --time=23:30:00
#SBATCH --gres=gpu:p40:1
#SBATCH --mem=12GB
#SBATCH --mail-type=END
#SBATCH --mail-user=ywn202@nyu.edu

python3 neural_style.py train --dataset /scratch/ywn202/style_train_data --style-image ./images/style_images/ --save-model-dir ./trained_models --epochs 2 --cuda 1 --log-interval 10 --batch-size 12

