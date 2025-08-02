import os
import argparse
import webdataset as wds
from tqdm import tqdm

def shard_video_dataset(video_dir, output_dir, samples_per_shard=1, video_ext=".mp4"):
    os.makedirs(output_dir, exist_ok=True)
    video_paths = sorted([
        os.path.join(video_dir, f) for f in os.listdir(video_dir)
        if f.lower().endswith(video_ext)
    ])
    if not video_paths:
        print(f"No video files found in {video_dir}.")
        return

    with wds.ShardWriter(output_dir, maxcount=samples_per_shard) as sink:
        for video_path in tqdm(video_paths, desc="Sharding videos"):
            video_id = os.path.splitext(os.path.basename(video_path))[0]
            with open(video_path, "rb") as vf:
                sample = {
                    "__key__": video_id,
                    "mp4": vf.read()
                }
                sink.write(sample)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Shard video dataset into WebDataset format.")
    parser.add_argument("--video_dir", type=str, required=True, help="Path to directory with video files.")
    parser.add_argument("--output_dir", type=str, required=True, help="Path to save .tar shards.")
    parser.add_argument("--samples_per_shard", type=int, default=1, help="Number of videos per shard.")
    parser.add_argument("--video_ext", type=str, default=".mp4", help="Video file extension to look for.")
    args = parser.parse_args()

    shard_video_dataset(args.video_dir, args.output_dir, args.samples_per_shard, args.video_ext)