"""Regression tests for preprocess_mtl.py (see issue #6)."""
import sys
from pathlib import Path

import pandas as pd
import pytest

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))
sys.path.insert(0, str(Path(__file__).resolve().parent))

import preprocess_mtl  # noqa: E402
from fixtures import write_fixture  # noqa: E402

HIST = [f"hist_{i}" for i in range(1, 8)]


def run(raw_dir, out_path, *extra):
    args = preprocess_mtl.parse_args(
        "test", ["--local_dir_path", str(raw_dir), "--out_path", str(out_path), *extra]
    )
    preprocess_mtl.main(args)
    return pd.read_csv(out_path)


@pytest.fixture
def raw(tmp_path):
    raw_dir = tmp_path / "raw"
    raw_dir.mkdir()
    df = write_fixture(raw_dir)
    return raw_dir, df


def test_product_and_color_ids_are_encoded_uniquely(raw, tmp_path):
    raw_dir, src = raw
    out = run(raw_dir, tmp_path / "out.csv")

    # Rows are only emitted from the MIN_SEQ_LEN-th event of each sequence onward.
    ordered = src.sort_values(["user_id", "sequence_id", "stime"])
    pos = ordered.groupby(["user_id", "sequence_id"]).cumcount()
    emitted = ordered[pos >= preprocess_mtl.MIN_SEQ_LEN - 1]
    assert len(out) == len(emitted)

    # Before the fix every product_id and color was 0.
    assert out["product_id"].nunique() == emitted["product_id"].nunique() > 1
    assert out["color"].nunique() == emitted["color"].nunique() > 1
    # Same raw value -> same encoded id, across the whole file.
    pairs = pd.DataFrame({"raw": emitted["product_id"].to_numpy(), "enc": out["product_id"].to_numpy()})
    assert pairs.groupby("raw")["enc"].nunique().max() == 1
    assert pairs.groupby("enc")["raw"].nunique().max() == 1


def test_history_window_holds_previous_products(raw, tmp_path):
    raw_dir, _ = raw
    out = run(raw_dir, tmp_path / "out.csv")

    assert out[HIST].to_numpy().max() > 0, "history window is all zeros"
    # hist_1..hist_7 of each row are the product_ids of the 7 preceding events
    # in the same sequence, so hist_7 of row k equals product_id of row k-1.
    for _, seq in out.groupby(["user_id", "sequence_id"]):
        seq = seq.reset_index(drop=True)
        for k in range(1, len(seq)):
            assert seq.loc[k, "hist_7"] == seq.loc[k - 1, "product_id"]


def test_rerun_overwrites_instead_of_appending(raw, tmp_path):
    raw_dir, _ = raw
    out_path = tmp_path / "out.csv"
    first = run(raw_dir, out_path)
    second = run(raw_dir, out_path)
    assert len(second) == len(first)


def test_use_product_id_flag_can_be_disabled(raw, tmp_path):
    raw_dir, src = raw
    assert preprocess_mtl.parse_args("t", []).use_product_id is True
    assert preprocess_mtl.parse_args("t", ["--no-use_product_id"]).use_product_id is False

    out = run(raw_dir, tmp_path / "out.csv", "--no-use_product_id")
    # With item_id history, the window contains raw item ids, not encoded product ids.
    assert set(out["hist_7"]) <= set(src["item_id"])


def test_missing_data_dir_gives_clear_error(tmp_path):
    with pytest.raises(FileNotFoundError, match="merrec_data.py"):
        run(tmp_path / "does-not-exist", tmp_path / "out.csv")
