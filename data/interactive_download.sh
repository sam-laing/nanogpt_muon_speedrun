#!/bin/bash
# filepath: /home/slaing/bias_correction_debunked/interactive_job.sh

# Default bid if not specified
bid=${1:-25}

# Fixed resource specifications
cpus=8
memory=50000

echo "Requesting interactive job with the following specifications:"
echo "- Bid: $bid"
echo "- CPUs: $cpus"
echo "- Memory: $memory MB"
echo

# Submit interactive job with bid and fixed resources
condor_submit_bid $bid -i \
  -append request_cpus=$cpus \
  -append request_memory=$memory \
  -append 'requirements = (HasOutboundConnectivity =?= True)'