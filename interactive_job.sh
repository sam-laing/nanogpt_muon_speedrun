#!/bin/bash
# filepath: /home/slaing/bias_correction_debunked/interactive_job.sh

# Default bid if not specified
bid=${1:-25}

# Fixed resource specifications
cpus=4
memory=20000
gpus=1
gpu_type="NVIDIA A100-SXM4-40GB"

echo "Requesting interactive job with the following specifications:"
echo "- Bid: $bid"
echo "- CPUs: $cpus"
echo "- Memory: $memory MB"
echo "- GPUs: $gpus"
echo "- GPU Type: $gpu_type"
echo

# Submit interactive job with bid and fixed resources
condor_submit_bid $bid -i \
  -append request_cpus=$cpus \
  -append request_memory=$memory \
  -append request_gpus=$gpus \
  -append "requirements=TARGET.CUDADeviceName==\"$gpu_type\""