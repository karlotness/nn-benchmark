#!/bin/bash
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --time=24:00:00
#SBATCH --mem=4GB
#SBATCH --job-name=MLPg_3object
#SBATCH --output=slurm_%j.out
cd /home/ct2243/Desktop/RESEARCH_ML/PhysicsExperiments


source activate symplectic


python run_experiment.py --physical_system_type gravitational --n_hidden 512 --n_object 3 --model_type MLP --dataset_size 20000

