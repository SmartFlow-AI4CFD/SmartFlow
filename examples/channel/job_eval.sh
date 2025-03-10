#!/bin/bash
# shopt -s extglob
# rm -r train __pycache__  experiment DRLsignals
# cd  train-0
# rm -r !(input*|stats*|fld_0.bin)
# cd ..

# cd  train-1
# rm -r !(input*|stats*|fld_0.bin)
# cd ..

# cd  train-2 
# rm -r !(input*|stats*|fld_0.bin)
# cd ..


# rm -r __pycache__  experiment dump_data logs models model_0.zip runs wandb

python main.py runner.mode="eval" > job.out 2> job.err
