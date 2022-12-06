#!/bin/bash

export AWS_BATCH_JOB_NODE_INDEX=0
export AWS_BATCH_JOB_NUM_NODES=1
export AWS_BATCH_JOB_MAIN_NODE_INDEX=0
export AWS_BATCH_JOB_ID=string


WORKSPACE=/workspace
# dist env preparation

NODE_TYPE="child"
if [ "${AWS_BATCH_JOB_MAIN_NODE_INDEX}" == "${AWS_BATCH_JOB_NODE_INDEX}" ]; then
    echo "Running on main node..."
    NODE_TYPE="main"
fi

# wait for all nodes to report IP
IP_CONFIG="${WORKSPACE}/ip_config.txt"
wait_for_nodes () {
    echo "Main node is waiting for all nodes to report IP..."
    touch ${IP_CONFIG}
    echo $(hostname -i) >> ${IP_CONFIG}
    NUM_LINES=$(sort ${IP_CONFIG} | uniq | wc -l)
    while [ "${AWS_BATCH_JOB_NUM_NODES}" -gt "${NUM_LINES}" ]
    do
        echo "${NUM_LINES} out of ${AWS_BATCH_JOB_NUM_NODES} nodes joined, check again in 1 second."
        sleep 1
        NUM_LINES=$(sort ${IP_CONFIG} | uniq | wc -l)
    done
    echo "All nodes successfully joined..."
}

# report IP to main node
report_to_main () {
    IP=$(hostname -i)
    CMD="echo ${IP} >> ${IP_CONFIG}"
    ssh ${AWS_BATCH_JOB_MAIN_NODE_PRIVATE_IPV4_ADDRESS} ${CMD}
    echo "Node~${AWS_BATCH_JOB_NODE_INDEX} has reported ${IP} to main node."
}

# share IP list with all nodes
share_to_nodes () {
    for target in $(cat ${IP_CONFIG})
    do
        scp ${IP_CONFIG} ${target}:${IP_CONFIG}
    done
    echo "Shared IP to all nodes."
}

# non-main nodes waits for IPs share
wait_for_ip_share () {
    while [ ! -f "${IP_CONFIG}" ]
    do
        echo "Node~${AWS_BATCH_JOB_NODE_INDEX} is waiting for IP share from main node. Sleep 1 second."
        sleep 1
    done
}
echo "Current node is ${NODE_TYPE}"
case ${NODE_TYPE} in
    "main")
        wait_for_nodes
        share_to_nodes
        ;;
    "child")
        report_to_main
        wait_for_ip_share
        ;;
esac
## collect IP addresses and scatter: ip_config.txt
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