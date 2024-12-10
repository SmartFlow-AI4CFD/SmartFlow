#!/bin/bash

### Job name on queue
#SBATCH --job-name=channel

### Output and error files directory
#SBATCH -D .

### Output and error files
#SBATCH --output=job_%j.out
#SBATCH --error=job_%j.err

### Run configuration
### Rule: {ntasks-per-node} \times {cpus-per-task} = 80
#SBATCH --nodes=1
#SBATCH --ntasks=2
#SBATCH --ntasks-per-node=2
#SBATCH --cpus-per-task=1
#SBATCH --time=48:00:00
##SBATCH --gres=gpu:4

### Queue and account
##SBATCH --qos=acc_resa
##SBATCH --account=upc76

### Load MN% modules + DRL libraries
#. ../../utils/modules-mn5-acc.sh

export SLURM_OVERLAP=1

python run.py

