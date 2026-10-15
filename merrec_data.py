"""
Download MerRec parquet shards from Hugging Face into the layout the scripts expect.

The dataset (https://huggingface.co/datasets/mercari-us/merrec) has one directory per
month (20230501 ... 20231001), each with ~300 parquet shards of ~70-90 MB.
Shards are written to ``<out_dir>/<month>/``, e.g. ``data/20230501/``, which is the
``--data_path`` / ``--local_dir_path`` used throughout the README.

Examples:
    # Quick experiment: first 5 shards of May 2023 (~400 MB)
    python merrec_data.py --month 20230501 --num_shards 5

    # Full month used in the paper's benchmarks (~25-30 GB)
    python merrec_data.py --month 20230501

Note: a partial download is a convenience sample. Sequences may span shards, so
some sequences can be cut short; use the full month to reproduce paper numbers.
"""
import sys
from argparse import ArgumentParser

REPO_ID = "mercari-us/merrec"
MONTHS = ["20230501", "20230601", "20230701", "20230801", "20230901", "20231001"]


def select_shards(files, month, num_shards=None):
    """Pick the parquet files for `month`, sorted, optionally only the first `num_shards`."""
    shards = sorted(f for f in files if f.startswith(f"{month}/") and f.endswith(".parquet"))
    if not shards:
        raise ValueError(f"No parquet shards found for month {month!r}.")
    if num_shards is not None:
        if num_shards < 1:
            raise ValueError("--num_shards must be >= 1")
        shards = shards[:num_shards]
    return shards


def download(month, out_dir="data", num_shards=None, revision=None):
    from huggingface_hub import HfApi, hf_hub_download

    files = HfApi().list_repo_files(REPO_ID, repo_type="dataset", revision=revision)
    shards = select_shards(files, month, num_shards)
    print(f"Downloading {len(shards)} shard(s) of {month} into {out_dir}/{month}/ ...")
    for i, shard in enumerate(shards, 1):
        # local_dir keeps the repo-relative path, i.e. <out_dir>/<month>/<shard>.parquet
        hf_hub_download(REPO_ID, shard, repo_type="dataset", revision=revision, local_dir=out_dir)
        print(f"  [{i}/{len(shards)}] {shard}")
    print("Done.")
    return [f"{out_dir}/{s}" for s in shards]


def parse_args(argv=None):
    parser = ArgumentParser(description="Download MerRec shards from Hugging Face.")
    parser.add_argument("--month", default="20230501", choices=MONTHS)
    parser.add_argument("--out_dir", default="data")
    parser.add_argument("--num_shards", type=int, default=None,
                        help="Only download the first N shards of the month (default: all).")
    parser.add_argument("--revision", default=None, help="Dataset git revision to pin (optional).")
    return parser.parse_args(argv)


if __name__ == "__main__":
    a = parse_args()
    download(a.month, a.out_dir, a.num_shards, a.revision)
    sys.exit(0)
