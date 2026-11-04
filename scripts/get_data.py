# /// script
# requires-python = ">=3.12"
# dependencies = [
#     "requests>=2.34.2",
# ]
# ///

import requests
import tarfile
import sys

s_dataset_url = "https://graphchallenge.s3.amazonaws.com/synthetic/sparsechallenge_2019/dnn/neuron1024.tar.gz"
m_dataset_url = "https://graphchallenge.s3.amazonaws.com/synthetic/sparsechallenge_2019/dnn/neuron4096.tar.gz"
l_dataset_url = "https://graphchallenge.s3.amazonaws.com/synthetic/sparsechallenge_2019/dnn/neuron16384.tar.gz"
xl_dataset_url = "https://graphchallenge.s3.amazonaws.com/synthetic/sparsechallenge_2019/dnn/neuron65536.tar.gz"


def download_dataset(url, local_path):
    try:
        with requests.get(url, stream=True) as response:
            response.raise_for_status()

            with tarfile.open(fileobj=response.raw, mode="r|gz") as tar:
                tar.extractall(path=local_path, filter="data")

        print(f"Downloaded into: {local_path}")
    except requests.exceptions.RequestException as e:
        print(f"Download failed: {e}")


def main():
    if len(sys.argv) < 2:
        print("Usage: get_data <local_filename> [size]")
        return

    dataset_url = None

    if len(sys.argv) == 2:
        dataset_url = s_dataset_url
    else:
        match sys.argv[2]:
            case "s":
                dataset_url = s_dataset_url
            case "m":
                dataset_url = m_dataset_url
            case "l":
                dataset_url = l_dataset_url
            case "xl":
                dataset_url = xl_dataset_url
            case _:
                print("Usage: get_data <local_path> [size]")
                return

    print(f"Downloading from {dataset_url}...")
    download_dataset(dataset_url, sys.argv[1])


if __name__ == "__main__":
    main()
