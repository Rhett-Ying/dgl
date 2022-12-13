import argparse
import os


if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description="Data store for distributed test",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    parser.add_argument(
        "--dataset",
        type=str,
        required=True,
        help="target dataset name to fetch"
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        required=True,
        help="output directory to save target dataset"
    )
    args, _ = parser.parse_known_args()

    os.system(
        f"aws s3 sync s3://dgl-data-store/{args.dataset} {args.output_dir}"
    )
    os.system(
        f"ls -lh {args.output_dir}"
    )

    print(f"Finished to download {args.dataset} to {args.output_dir}")
