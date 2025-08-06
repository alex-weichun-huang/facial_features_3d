import os
import argparse
import webdataset as wds
from tqdm import tqdm

def shard_npy_dataset(npy_dir, output_dir, samples_per_shard=1, file_ext=".npy"):
    os.makedirs(output_dir, exist_ok=True)
    npy_folder_names = os.listdir(npy_dir)
    # npy paths append .npy to each folder name
    npy_paths = sorted([
        os.path.join(npy_dir, f"{folder_name}/{folder_name}{file_ext}") for folder_name in npy_folder_names
        if os.path.isdir(os.path.join(npy_dir, folder_name))
    ])

    with wds.ShardWriter(os.path.join(output_dir, "shard-%03d.tar"), maxcount=samples_per_shard) as sink:
        for npy_path in tqdm(npy_paths, desc="Sharding .npy files"):
            sample_id = os.path.splitext(os.path.basename(npy_path))[0]
            with open(npy_path, "rb") as nf:
                sample = {
                    "__key__": sample_id,
                    "npy": nf.read()
                }
                sink.write(sample)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Shard .npy dataset into WebDataset format.")
    parser.add_argument("--npy_dir", type=str, required=True, help="Path to directory with .npy files.")
    parser.add_argument("--output_dir", type=str, required=True, help="Path to save .tar shards.")
    parser.add_argument("--samples_per_shard", type=int, default=1, help="Number of .npy files per shard.")
    parser.add_argument("--file_ext", type=str, default=".npy", help="File extension to look for.")
    args = parser.parse_args()

    shard_npy_dataset(args.npy_dir, args.output_dir, args.samples_per_shard, args.file_ext)
