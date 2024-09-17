#!/bin/bash

### Job name on queue
#SBATCH --job-name=DRL_cyl

### Output and error files directory
#SBATCH -D .

### Output and error files
#SBATCH --output=mpi_%j.out
#SBATCH --error=mpi_%j.err

### Run configuration
### Rule: {ntasks-per-node} \times {cpus-per-task} = 80
#SBATCH --nodes=1
#SBATCH --ntasks=4
#SBATCH --ntasks-per-node=4
#SBATCH --cpus-per-task=20
#SBATCH --time=48:00:00
#SBATCH --gres=gpu:4

### Queue and account
#SBATCH --qos=acc_resa
#SBATCH --account=upc76

### Load MN% modules + DRL libraries
. ../../utils/modules-mn5-acc.sh

export SLURM_OVERLAP=1

python run.py

