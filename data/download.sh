#!/bin/bash

export HOME=/lustre/home/slaing

source ~/miniforge3/etc/profile.d/conda.sh
conda activate 312nets

# Execute python script
python /lustre/home/slaing/modded-nanogpt/data/fineweb.py 