#!/bin/bash

export AWS_BATCH_JOB_NODE_INDEX=0
export AWS_BATCH_JOB_NUM_NODES=1
export AWS_BATCH_JOB_MAIN_NODE_INDEX=0
export AWS_BATCH_JOB_ID=string


WORKSPACE=/workspace
WORKSPACE=/home/ubuntu/workspace/dgl2
IP_CONFIG="${WORKSPACE}/ip_config.txt"

## collect IP addresses and scatter: ip_config.txt
bash ${WORKSPACE}/tests/regression/generate_ip_config.sh ${WORKSPACE} ${IP_CONFIG}
#echo "ip_list: $(cat ${WORKSPACE}/ip_config.txt)"

# fetch raw data
#rank=1
## DataStore --dataset ogbn-products --rank 1 --world_size 4
#echo "Rank~${rank} fetched data onto ${workspace}/ogbn-products/chunked_data."

# graph partition
## install latest nightly build DGL
## fetch latest DGL master branch
## python3 /dgl/tools/dispatch_data.py ...

# report generation
## ReportGeneration --log /workspace/graph_partition.log_221206

echo "All is done..."