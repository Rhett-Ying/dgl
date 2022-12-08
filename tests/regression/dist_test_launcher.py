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
        print(f"{workspace} is created...")

    return workspace

def fetch_raw_data():
    workspace = os.environ.get('WORKSPACE', '/workspace')
    os.system(
        'aws s3 sync s3://dgl-data-store/test_dataset '
        f'{workspace}/test_dataset'
    )

if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description="Distributed test launcher",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    _, _ = parser.parse_known_args()

    # DGL preparation
    #prepare_dgl()

    # export envs
    workspace = prepare_env()

    # fetch raw data
    fetch_raw_data()

    # generate ip_config.txt
    ip_config = os.path.join(workspace, 'ip_config.txt')
    os.system(
        f"bash /dgl/tests/regression/generate_ip_config.sh {ip_config}"
    )
    
