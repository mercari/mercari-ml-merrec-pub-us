"""Small synthetic parquet fixture that mimics the MerRec raw schema."""
import random

import pandas as pd

EVENTS = ["item_view", "item_like", "item_add_to_cart_tap", "offer_make", "buy_start", "buy_comp"]


def make_raw_df(n_users=6, seqs_per_user=2, seq_len=10, n_products=25, n_colors=5, seed=0):
    rng = random.Random(seed)
    rows = []
    t = 0
    for u in range(n_users):
        for s in range(seqs_per_user):
            for _ in range(seq_len):
                t += 1
                pid = rng.randrange(n_products)
                rows.append({
                    "user_id": u,
                    "sequence_id": f"seq{u}_{s}",
                    "session_id": f"sess{u}_{s}",
                    "item_id": f"item{rng.randrange(10_000)}",
                    "event_id": rng.choice(EVENTS),
                    "price": float(rng.randrange(1, 500)),
                    "product_id": f"prod{pid}",
                    "c0_id": pid % 3,
                    "c1_id": pid % 7,
                    "c2_id": pid % 11,
                    "brand_id": pid % 13,
                    "item_condition_id": rng.randrange(1, 6),
                    "size_id": rng.randrange(20),
                    "shipper_id": rng.randrange(1, 4),
                    "color": f"color{rng.randrange(n_colors)}",
                    "stime": pd.Timestamp("2023-05-01") + pd.Timedelta(seconds=t),
                    "sequence_length": seq_len,
                })
    return pd.DataFrame(rows)


def write_fixture(dir_path, **kwargs):
    df = make_raw_df(**kwargs)
    df.to_parquet(dir_path / "part-0000.parquet", index=False)
    return df
