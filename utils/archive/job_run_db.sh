#!/bin/bash
#SBATCH --time=00:30:00
#SBATCH --nodes=3
#SBATCH --ntasks-per-node=4
#SBATCH --gres=gpu:4
#SBATCH --cpus-per-task=8
#SBATCH --partition=boost_usr_prod
##SBATCH --qos=boost_qos_dbg
#SBATCH --qos=normal
##SBATCH --job-name=gpu
#SBATCH --err=err
#SBATCH --out=out
#SBATCH --account=IscrB_SCORE

##SBATCH --ntasks-per-socket=
##SBATCH --mem=494000 # memory per node out of 494000MB (481GB)

python tmp7_run_db.py
