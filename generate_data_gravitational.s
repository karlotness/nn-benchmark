#!/bin/bash
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --time=24:00:00
#SBATCH --mem=2GB
#SBATCH --job-name=EXP
#SBATCH --output=slurm_%j.out
cd /home/ct2243/Desktop/RESEARCH_ML/PhysicsExperiments


source activate symplectic


python generate_data_gravitational.py --nTraj 20000 --n 3 --interval 1 --epsilon 0.05 --pointsPerTraj 10 --randomize 1 --uniformMass 1 

