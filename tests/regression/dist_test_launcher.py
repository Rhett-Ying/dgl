import os
import argparse

def prepare_dgl():
    os.system(
        "pip3 install --pre dgl -f https://data.dgl.ai/wheels-test/repo.html"
    )

def prepare_env():
    os.system('service ssh restart')

    workspace = os.environ.get('WORKSPACE', '/workspace')
    if not os.path.isdir(workspace):
        os.makedirs(workspace)
        os.environ['WORKSPACE'] = workspace
    os.environ['IP_CONFIG'] = os.path.join(workspace, 'ip_config.txt')
    os.environ['SSH_PORT'] = 2233

def fetch_raw_data():
    workspace = os.environ.get('WORKSPACE', '/workspace')
    os.system(
        'aws s3 sync s3://dgl-data-store/test_dataset '
        f'{workspace}/test_dataset'
    )
    os.system(
        f'ls -lh {workspace}/test_dataset'
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
    prepare_env()

    # generate ip_config.txt
    os.system(
        "bash /dgl/tests/regression/generate_ip_config.sh"
    )

    # fetch raw data
    fetch_raw_data()
