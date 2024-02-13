"""
Reformats the raw MerRec dataset into the MTL/CTR format.
"""

import csv
import os
import glob
import pandas as pd
import pytz
import datetime as dt
import sys
from argparse import ArgumentParser


MIN_SEQ_LEN = 8
MAX_SEQ_LEN = 22
INPUT_WINDOW_LENGTH = MIN_SEQ_LEN - 1

event_id_map = {
    "item_view":0,
    "item_like":1,
    "item_add_to_cart_tap":2,
    "offer_make":3,
    "buy_start":4,
    "buy_comp":5,
}

csv_header = [
    "user_id",
    "sequence_id",
    "session_id",
    "item_id",
    *event_id_map.keys(),
    "price",
    "product_id",
    "c0_id",
    "c1_id",
    "c2_id",
    "brand_id",
    "item_condition_id",
    "size_id",
    "shipper_id",
    "color",
    "hist_1",
    "hist_2",
    "hist_3",
    "hist_4",
    "hist_5",
    "hist_6",
    "hist_7"
]


def glob_local_parquets(local_dir_path):
    """Glob local parquet files."""
    return glob.glob(f"{local_dir_path}/*.parquet")


def parse_args(description):
    parser = ArgumentParser(description=description)
    parser.add_argument('--local_dir_path', default='./data/20230501')
    parser.add_argument('--out_path', default='./data/mtl_product.csv')
    parser.add_argument("--use_product_id", type=bool, default=True, help="True:use product_id, False:use item_id.")
    args = parser.parse_args()
    return args


def main(args):
    print(f"Input arguments: {args}")
    if args.use_product_id is True:
        input_column = 11  # Use product_id in history window
    else:
        input_column = 3  # Use item_id in history window
    start_time = pytz.utc.localize(dt.datetime.utcnow()).astimezone(pytz.timezone('US/Pacific'))
    print(f"Beginning conversion script. time: {start_time}")

    # Fetch local parquet file paths
    parquet_file_paths = glob_local_parquets(args.local_dir_path)
    
    # Read and merge parquet files
    df = pd.concat([pd.read_parquet(f) for f in parquet_file_paths])
    df = df[df.sequence_length>=8]
    df.sort_values(by=["user_id", "sequence_id", "stime"], ascending=True, inplace=True)
    print(f"Merged Dataframe size: {df.shape}.")

    product_id_map = {}
    color_id_map = {}

    i = l = 0
    discarded = 0
    while i < df.shape[0]:  # i loops across sequences

        # Beggining of a sequence
        event = df.iloc[i,:].to_dict()
        user_id = event["user_id"]  # Column 0
        sequence_id = event["sequence_id"]  # Column 1
        engagements = []

        j = i  # j loops within the sequence
        while event["user_id"] == user_id and event["sequence_id"] == sequence_id:
            event_id = event["event_id"]
            event_cells = [0] * len(event_id_map)
            event_cells[event_id_map[event_id]] = 1
            product_id = product_id_map.get(event["product_id"], len(product_id_map))
            color_id = color_id_map.get(event["color"], len(color_id_map))
            row = [
                user_id,
                sequence_id,
                event["session_id"],  # Column 2
                event["item_id"],  # Column 3
                *event_cells,
                event["price"],  # Column 10,
                product_id,  # Column 11
                event["c0_id"],  # Column 12
                event["c1_id"],  # Column 13
                event["c2_id"],  # Column 14
                event["brand_id"],  # Column 15
                event["item_condition_id"],  # Column 16
                event["size_id"],  # Column 17
                event["shipper_id"],  # Column 18
                color_id,  # Column 19
            ]
            engagements.append(row)
            j += 1
            if j >= df.shape[0]:
                break
            event = df.iloc[j,:].to_dict()

        i = j  # i is now at beginning of next sequence

        if len(engagements) < MIN_SEQ_LEN:
            # Skip this sequence if it's shorter than our specification
            discarded += 1
            continue

        # Reformat into multiple rows with sliding window input item IDs
        for k, event in enumerate(engagements):
            if k >= MIN_SEQ_LEN - 1:
                input_item_ids = [e[input_column] for e in engagements[k-INPUT_WINDOW_LENGTH:k]]
                assert len(input_item_ids) == INPUT_WINDOW_LENGTH
                event.extend(input_item_ids)
                if not os.path.exists(args.out_path):
                    if not os.path.exists(os.path.dirname(args.out_path)):
                        os.makedirs(os.path.dirname(args.out_path))
                    with open(args.out_path, 'w', newline='') as file:
                        csvwriter = csv.writer(file)
                        csvwriter.writerow(csv_header)
                        csvwriter.writerow(event)
                else:
                    with open(args.out_path, 'a', newline='') as file:
                        csvwriter = csv.writer(file)
                        csvwriter.writerow(event)
        
        # Print progress
        if i % 100 == 0:
            l += 1
            print(f"Finished processing {l-1} x 100 sequences")

    end_time = pytz.utc.localize(dt.datetime.utcnow()).astimezone(pytz.timezone('US/Pacific'))
    diff_time = end_time - start_time
    duration_in_s = diff_time.total_seconds()
    print(f"Completed running conversion script. time: {end_time}, diff: {duration_in_s} seconds, Discarded {discarded} sequences.")


if __name__ == "__main__":
    sys.exit(main(parse_args("Run MTL/CTR dataset preprocessing pipeline.")))
