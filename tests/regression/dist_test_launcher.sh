#!/bin/bash


$(service ssh restart)

WORKSPACE="/workspace"

while getopts 'h' opt; do
    case "${opt}" in
        ?|h)
            echo "Usage: $(basename $0) [-w arg]"
            exit 1
            ;;
    esac
done

echo "Workspace: ${WORKSPACE}."
export WORKSPACE=${WORKSPACE}
export IP_CONFIG="${WORKSPACE}/ip_config.txt"
export SSH_PORT=2233

## collect IP addresses and scatter: ip_config.txt
bash /dgl/tests/regression/generate_ip_config.sh

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