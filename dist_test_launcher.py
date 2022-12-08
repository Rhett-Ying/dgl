import os
import argparse

def prepare_dgl():
    os.system(
        "pip3 install --pre dgl -f https://data.dgl.ai/wheels-test/repo.html"
    )

def export_envs():
    workspace = os.environ.get('WORKSPACE', '/workspace')
    if not os.path.isdir(workspace):
        os.makedirs(workspace)
    os.system(
        f"export WORKSPACE={workspace}"
        f" && export IP_CONFIG={workspace}/ip_config.txt"
        " && export SSH_PORT=2233"
    )

if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description="Distributed test launcher",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    _, _ = parser.parse_known_args()

    # DGL preparation
    prepare_dgl()

    # export envs
    export_envs()

    # generate ip_config.txt
    os.system(
        "bash /dgl/tests/regression/generate_ip_config.sh"
    )

