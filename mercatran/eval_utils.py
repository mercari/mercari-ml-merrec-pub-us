import collections
import os
from argparse import Namespace
from enum import Enum
from typing import List

import faiss
import numpy as np
import pandas as pd
import torch
from embed import create_item_encoder_mask, create_subsequent_mask
from tokenizers import Tokenizer
from tqdm.auto import tqdm

from data import create_start_token_sequence


class DCGRelScore(Enum):
    item_id_match = 1.0
    brand_id_match = 0.3
    category_id_match = 0.3

    def __str__(self):
        return str(self.value)


class DCGRelOptions(Namespace):
    item_brand_cat = "item_brand_cat"
    raw_engagement = "raw_engagement"


class RetrievalIndex:
    def __init__(self, d_model: int, lookup_size: int):
        self.d_model = d_model
        self.lookup_size = lookup_size
        self.index = faiss.IndexIDMap(faiss.IndexFlatIP(self.d_model))
        self.item_metadata = {}
        self.map_cat_ids = np.vectorize(self.get_cat_id)
        self.map_brand_ids = np.vectorize(self.get_brand_id)

    def insert_items(self, item_embeds, item_ids, cat_ids, brand_ids):
        item_ids = torch.flatten(item_ids).cpu().numpy()
        cat_ids = torch.flatten(cat_ids).cpu().numpy()
        brand_ids = torch.flatten(brand_ids).cpu().numpy()
        self.index.add_with_ids(
            item_embeds.view(-1, self.d_model).cpu().numpy(),
            item_ids,
        )
        for b, it in enumerate(item_ids):
            self.item_metadata[int(it)] = {
                "brand_id": int(brand_ids[b]),
                "cat_id": int(cat_ids[b]),
            }

    def get_cat_id(self, i: int):
        return self.item_metadata[i]["cat_id"]

    def get_brand_id(self, i: int):
        return self.item_metadata[i]["brand_id"]

    def query_search(self, query_embed):
        _, item_ids = self.index.search(
            query_embed.cpu().numpy(),
            k=self.lookup_size,
        )
        cat_ids = self.map_cat_ids(item_ids)
        brand_ids = self.map_brand_ids(item_ids)
        return item_ids, cat_ids, brand_ids


class Evaluator:
    def __init__(
        self,
        batch_size,
        num_eval_seq,
        model,
        d_model,
        lookup_size,
        val_loader,
        eval_ks: List[int],
        tokenizer: Tokenizer,
        out_dir: str = None,
        ndcg_rel_option: str = DCGRelOptions.raw_engagement,
    ):
        self.batch_size = batch_size
        self.num_eval_seq = num_eval_seq
        self.model = model
        self.d_model = d_model
        self.lookup_size = lookup_size
        self.val_loader = val_loader
        self.eval_ks = eval_ks
        self.tokenizer = tokenizer
        self.out_dir = out_dir
        self.ndcg_rel_option = ndcg_rel_option
        if not os.path.exists(self.out_dir):
            os.makedirs(self.out_dir)

    def calc_rel(
        self,
        row,
        true_item_ids,
        true_brand_ids,
        true_category_ids,
    ):
        """Calculate rel score, potentially with partial credits."""
        # Reverse look up metadata for predicted item and ground truth items
        pred_item_id = row["item_id"]
        pred_brand_id = row["brand_id"]
        pred_category_id = row["category_id"]
        if (
            self.ndcg_rel_option == DCGRelOptions.item_brand_cat
        ):  # Hit/miss by non-engagement item ID/brand/cat matching
            if pred_item_id in true_item_ids:
                score = DCGRelScore.item_id_match.value
            else:  # item ID no match found
                score = 0.0
                if pred_brand_id in true_brand_ids:
                    score += DCGRelScore.brand_id_match.value
                if pred_category_id in true_category_ids:
                    score += DCGRelScore.category_id_match.value
        elif (
            self.ndcg_rel_option == DCGRelOptions.raw_engagement
        ):  # Hit by engaged item ID only
            if pred_item_id in true_item_ids:
                score = DCGRelScore.item_id_match.value
            else:
                score = 0.0
        return score

    def calc_ndcg(
        self,
        pred_item_ids,
        pred_brand_ids,
        pred_category_ids,
        true_item_ids,
        true_brand_ids,
        true_category_ids,
    ):
        """Calculate personalized DCG, IDCG and NDCG.
        (no between-step allowance for match criteria)
        """
        df_pred = pd.DataFrame.from_dict(
            {
                "item_id": pred_item_ids,
                "brand_id": pred_brand_ids,
                "category_id": pred_category_ids,
            }
        )
        df_pred["position"] = df_pred.reset_index().index.values + 1
        df_pred["score"] = df_pred.apply(
            lambda row: self.calc_rel(
                row,
                true_item_ids,
                true_brand_ids,
                true_category_ids,
            ),
            axis=1,
        )

        if (df_pred["score"] == 0.0).all():
            return np.zeros(df_pred.shape[0])
        else:
            df_pred["discounted_score"] = (
                np.power(2, df_pred["score"].values) - 1
            ) / np.log2(df_pred["position"].values + 1)
            df_pred["dcg"] = (
                df_pred.sort_values(by="position", ascending=True)
                .loc[:, ["discounted_score"]]
                .cumsum()
            )
            df_ideal = df_pred.copy()
            df_ideal = df_ideal.sort_values(
                by=["score", "position"], ascending=[False, True]
            ).reset_index()
            df_ideal["ideal_position"] = df_ideal.index.values + 1
            df_ideal["ideal_discounted_score"] = (
                np.power(2, df_ideal["score"].values) - 1
            ) / np.log2(df_ideal["ideal_position"].values + 1)
            df_ideal["idcg"] = (
                df_ideal.sort_values(by="ideal_position", ascending=True)
                .loc[:, ["ideal_discounted_score"]]
                .cumsum()
            )
            ndcg = df_pred["dcg"].values / df_ideal["idcg"].values
            return ndcg

    def evaluate(self, epoch_train: int, desc=""):
        """Run one full pass over validation data."""
        index = RetrievalIndex(self.d_model, self.lookup_size)
        total_cat = total_brand = total_item = 0
        total_pred = {}
        cat_pos, brand_pos, item_pos = {}, {}, {}
        cat_pos_bin, brand_pos_bin = {}, {}  # Binarized counts for recall@k
        cat_recall, brand_recall, item_recall = {}, {}, {}
        cat_precision, brand_precision, item_precision = {}, {}, {}
        cat_count, brand_count = {}, {}  # Diversity counts
        ndcg = {}
        for k in self.eval_ks:
            cat_pos[k] = collections.defaultdict(float)
            brand_pos[k] = collections.defaultdict(float)
            item_pos[k] = collections.defaultdict(float)
            cat_pos_bin[k] = collections.defaultdict(float)
            brand_pos_bin[k] = collections.defaultdict(float)
            cat_recall[k] = collections.defaultdict(float)
            brand_recall[k] = collections.defaultdict(float)
            item_recall[k] = collections.defaultdict(float)
            cat_precision[k] = collections.defaultdict(float)
            brand_precision[k] = collections.defaultdict(float)
            item_precision[k] = collections.defaultdict(float)
            cat_count[k] = collections.defaultdict(float)
            brand_count[k] = collections.defaultdict(float)
            ndcg[k] = collections.defaultdict(float)
            total_pred[k] = 0.0
        for i, batch in enumerate(
            tqdm(self.val_loader, desc=f"Epoch[{epoch_train}] Indexing batch")
        ):
            _, _, item_y, item_y_mask, category_id, brand_id, item_id = batch
            item_embeds = self.model.item_encoder(
                self.model.item_encoder_embed(item_y),
                create_item_encoder_mask(item_y_mask.unsqueeze(-2)),
            )
            index.insert_items(
                item_embeds=item_embeds,
                item_ids=item_id,
                cat_ids=category_id,
                brand_ids=brand_id,
            )

        if torch.cuda.is_available():
            torch.cuda.empty_cache()

        for i, batch in enumerate(
            tqdm(self.val_loader,
                 desc=f"Epoch[{epoch_train}] Evaluating batch")
        ):
            pred_category = collections.defaultdict(list)
            pred_brand = collections.defaultdict(list)
            pred_item = collections.defaultdict(list)
            actual_category = {}
            actual_brand = {}
            actual_item = {}

            user, user_mask, _, _, category_id, brand_id, item_id = batch
            user_enc_out = self.model.user_encoder(
                self.model.user_encoder_embed(user), user_mask.unsqueeze(-2)
            )
            dec_embed = self.model.user_decoder_embed(
                create_start_token_sequence(
                    tokenizer=self.tokenizer, batch_size=self.batch_size)
            )

            for b in range(self.batch_size):
                actual_category[b] = list(category_id.cpu().numpy()[b])
                actual_brand[b] = list(brand_id.cpu().numpy()[b])
                actual_item[b] = list(item_id.cpu().numpy()[b])

            for t in range(self.num_eval_seq):
                user_dec_out = self.model.user_decoder(
                    dec_embed,
                    user_enc_out,
                    user_mask.unsqueeze(-2),
                    create_subsequent_mask(self.batch_size, dec_embed.size(1)),
                )
                item_ids, cat_ids, brand_ids = index.query_search(
                    query_embed=user_dec_out[:, -1, :],
                )
                for b in range(self.batch_size):
                    pred_category[b].append(cat_ids[b])
                    pred_brand[b].append(brand_ids[b])
                    pred_item[b].append(item_ids[b])

                dec_embed = torch.cat(
                    (dec_embed, user_dec_out[:, -1, :].unsqueeze(1)), dim=1
                )

            for b in actual_category:
                for seq in range(self.num_eval_seq):
                    for k in self.eval_ks:
                        cat_pos[k][seq] += np.sum(
                            actual_category[b][seq] == pred_category[b][seq][0:k]  # noqa: E501
                        )
                        brand_pos[k][seq] += np.sum(
                            actual_brand[b][seq] == pred_brand[b][seq][0:k]
                        )
                        item_pos[k][seq] += np.sum(
                            actual_item[b][seq] == pred_item[b][seq][0:k]
                        )
                        cat_pos_bin[k][seq] += (
                            np.sum(
                                actual_category[b][seq] == pred_category[b][seq][0:k]  # noqa: E501
                            )
                            >= 1
                        )
                        brand_pos_bin[k][seq] += (
                            np.sum(actual_brand[b][seq] ==
                                   pred_brand[b][seq][0:k]) >= 1
                        )
                        cat_count[k][seq] += np.unique(
                            pred_category[b][seq][0:k]
                        ).shape[0]
                        brand_count[k][seq] += np.unique(
                            pred_brand[b][seq][0:k]
                        ).shape[0]
                        _ndcg = self.calc_ndcg(
                            pred_item_ids=pred_item[b][seq][0:k],
                            pred_brand_ids=pred_brand[b][seq][0:k],
                            pred_category_ids=pred_item[b][seq][0:k],
                            true_item_ids=np.array([actual_item[b][seq]]),
                            true_brand_ids=np.array([actual_brand[b][seq]]),
                            true_category_ids=np.array(
                                [actual_category[b][seq]]),
                        )
                        ndcg[k][seq] += _ndcg[min(k, len(_ndcg) - 1)]

                total_cat += len(actual_category[b])
                total_brand += len(actual_brand[b])
                total_item += len(actual_item[b])
                for k in self.eval_ks:
                    total_pred[k] += self.num_eval_seq * k

        if torch.cuda.is_available():
            torch.cuda.empty_cache()

        for k in self.eval_ks:
            for seq in range(self.num_eval_seq):
                cat_precision[k][seq] = round(
                    cat_pos[k][seq] / (total_pred[k] / self.num_eval_seq), 4
                )
                brand_precision[k][seq] = round(
                    brand_pos[k][seq] / (total_pred[k] / self.num_eval_seq), 4
                )
                item_precision[k][seq] = round(
                    item_pos[k][seq] / (total_pred[k] / self.num_eval_seq), 4
                )
                cat_recall[k][seq] = round(
                    cat_pos_bin[k][seq] / (total_cat / self.num_eval_seq), 4
                )
                brand_recall[k][seq] = round(
                    brand_pos_bin[k][seq] /
                    (total_brand / self.num_eval_seq), 4
                )
                item_recall[k][seq] = round(
                    item_pos[k][seq] / (total_item / self.num_eval_seq), 4
                )
                cat_count[k][seq] = round(
                    cat_count[k][seq] / (total_cat / self.num_eval_seq), 4
                )
                brand_count[k][seq] = round(
                    brand_count[k][seq] / (total_brand / self.num_eval_seq), 4
                )
                ndcg[k][seq] = round(
                    ndcg[k][seq] / (total_item / self.num_eval_seq), 4)
            result_k = pd.DataFrame(
                pd.concat(
                    [
                        pd.Series(cat_precision[k].values()),
                        pd.Series(brand_precision[k].values()),
                        pd.Series(item_precision[k].values()),
                        pd.Series(cat_recall[k].values()),
                        pd.Series(brand_recall[k].values()),
                        pd.Series(item_recall[k].values()),
                        pd.Series(cat_count[k].values()),
                        pd.Series(brand_count[k].values()),
                        pd.Series(ndcg[k].values()),
                    ],
                    axis=1,
                )
            )
            result_k.columns = [
                "Category P",
                "Brand P",
                "Item P",
                "Category R",
                "Brand R",
                "Item R",
                "Category C",
                "Brand C",
                "NDCG",
            ]
            print("-" * 30 + f"Metrics @{k}" + "-" * 30)
            print(result_k)
            print("-" * 70)
            if self.out_dir is not None:
                result_k.to_csv(
                    os.path.join(
                        self.out_dir,
                        f"results_at{k}_{epoch_train}_{desc}.csv"
                    )
                )
        return result_k
