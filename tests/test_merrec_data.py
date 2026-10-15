import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from merrec_data import parse_args, select_shards  # noqa: E402

FILES = [
    "README.md",
    "20230501/000000000001.parquet",
    "20230501/000000000000.parquet",
    "20230501/000000000002.parquet",
    "20230601/000000000000.parquet",
]


def test_select_shards_filters_month_and_sorts():
    assert select_shards(FILES, "20230501") == [
        "20230501/000000000000.parquet",
        "20230501/000000000001.parquet",
        "20230501/000000000002.parquet",
    ]


def test_select_shards_limits_count():
    assert select_shards(FILES, "20230501", num_shards=2) == [
        "20230501/000000000000.parquet",
        "20230501/000000000001.parquet",
    ]


def test_select_shards_errors():
    with pytest.raises(ValueError):
        select_shards(FILES, "20231001")
    with pytest.raises(ValueError):
        select_shards(FILES, "20230501", num_shards=0)


def test_parse_args_defaults_match_readme_paths():
    a = parse_args([])
    assert (a.out_dir, a.month) == ("data", "20230501")
