#!/bin/bash
#SBATCH --time=00:30:00
#SBATCH --nodes=2
#SBATCH --ntasks-per-node=32
##SBATCH --gres=gpu:4
#SBATCH --cpus-per-task=1
#SBATCH --partition=boost_usr_prod
#SBATCH --qos=boost_qos_dbg
##SBATCH --job-name=gpu
#SBATCH --err=err
#SBATCH --out=out
#SBATCH --account=IscrB_SCORE

##SBATCH --ntasks-per-socket=
##SBATCH --mem=494000 # memory per node out of 494000MB (481GB)

python -u main.py
