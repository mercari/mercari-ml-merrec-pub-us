import logging
import os
import pathlib
import sys
from argparse import ArgumentParser
from functools import partial

import config
import torch
import torch.nn as nn
from data import (
    MASK_TOKEN,
    UserItemInteractionDataset,
    collate_batch_item,
    collate_batch_item_val,
    sequence_dataset,
    train_tokenizer,
)
from embed import (
    ItemEmbeddings,
    PositionalEncoding,
    UserEmbeddings,
    create_item_encoder_mask,
    create_user_target_mask,
)
from eval_utils import Evaluator
from model import (
    Decoder,
    DecoderLayer,
    Encoder,
    EncoderLayer,
    MultiHeadedAttention,
    PositionwiseFeedForward,
    ThreeTower,
    model_initialization,
    rate,
)
from sklearn.model_selection import train_test_split
from torch.optim.lr_scheduler import LambdaLR
from torch.utils.data import DataLoader
from tqdm.auto import tqdm


def main(args):
    model_dir = pathlib.Path(args.save_path)
    model_dir.mkdir(parents=True, exist_ok=True)
    data_path = args.data_path
    seq_dataset = sequence_dataset(path=data_path)
    tokenizer = train_tokenizer(df_train=seq_dataset)
    tokenizer.save(os.path.join(args.save_path, args.tokenizer_save_name))
    train_df, test_df = train_test_split(
        seq_dataset, test_size=args.test_frac, random_state=args.seed)
    val_df, test_df = train_test_split(
        test_df, test_size=0.5, random_state=args.seed)

    train_dataset = UserItemInteractionDataset(interactions=train_df)
    val_dataset = UserItemInteractionDataset(interactions=val_df)
    test_dataset = UserItemInteractionDataset(interactions=test_df)

    train_loader = DataLoader(
        train_dataset,
        batch_size=config.BATCH_SIZE,
        shuffle=True,
        collate_fn=partial(collate_batch_item, tokenizer=tokenizer),
        drop_last=True,
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=config.BATCH_SIZE,
        shuffle=False,
        collate_fn=partial(collate_batch_item_val, tokenizer=tokenizer),
        drop_last=True,
    )
    test_loader = DataLoader(
        test_dataset,
        batch_size=config.BATCH_SIZE,
        shuffle=False,
        collate_fn=partial(collate_batch_item_val, tokenizer=tokenizer),
        drop_last=True,
    )

    model = ThreeTower(
        Encoder(  # user_encoder
            EncoderLayer(
                config.D_MODEL,
                MultiHeadedAttention(config.NUM_HEADS, config.D_MODEL),
                PositionwiseFeedForward(
                    config.D_MODEL, config.D_FF, config.DROPOUT),
                config.DROPOUT,
            ),
            config.NUM_STACKS,
        ),
        nn.Sequential(  # user_encoder_embed
            UserEmbeddings(
                vocab_size=config.BPE_VOCAB_LIMIT,
                d_model=config.D_MODEL,
                padding_idx=tokenizer.token_to_id(MASK_TOKEN),
                max_norm=config.MAX_NORM,
            ),
            nn.Unflatten(0, (config.BATCH_SIZE, -1)),
            PositionalEncoding(config.D_MODEL, config.DROPOUT,
                               config.POSITION_MAX_LEN),
        ),
        Decoder(  # user_decoder
            DecoderLayer(
                config.D_MODEL,
                MultiHeadedAttention(config.NUM_HEADS, config.D_MODEL),
                MultiHeadedAttention(config.NUM_HEADS, config.D_MODEL),
                PositionwiseFeedForward(
                    config.D_MODEL, config.D_FF, config.DROPOUT),
                config.DROPOUT,
            ),
            config.NUM_STACKS,
        ),
        nn.Sequential(  # user_decoder_embed
            ItemEmbeddings(
                vocab_size=config.BPE_VOCAB_LIMIT,
                d_model=config.D_MODEL,
                padding_idx=tokenizer.token_to_id(MASK_TOKEN),
                max_norm=config.MAX_NORM,
            ),
            nn.Unflatten(0, (config.BATCH_SIZE, -1)),
            PositionalEncoding(config.D_MODEL, config.DROPOUT,
                               config.POSITION_MAX_LEN),
        ),
        Encoder(  # item_encoder
            EncoderLayer(
                config.D_MODEL,
                MultiHeadedAttention(config.NUM_HEADS, config.D_MODEL),
                PositionwiseFeedForward(
                    config.D_MODEL, config.D_FF, config.DROPOUT),
                config.DROPOUT,
            ),
            config.NUM_STACKS,
        ),
        nn.Sequential(  # item_encoder_embed
            ItemEmbeddings(
                vocab_size=config.BPE_VOCAB_LIMIT,
                d_model=config.D_MODEL,
                padding_idx=tokenizer.token_to_id(MASK_TOKEN),
                max_norm=config.MAX_NORM,
            ),
            nn.Unflatten(0, (config.BATCH_SIZE, -1)),
        ),
    )

    model = model_initialization(model)
    model.to(config.DEVICE)
    logging.info(f"The device is: {config.DEVICE}")
    ce_loss = nn.CrossEntropyLoss()
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=config.BASE_LR,
        betas=config.BETAS,
        eps=config.EPS,
        weight_decay=config.WEIGHT_DECAY,
        amsgrad=False,
        maximize=False,
    )

    lr_scheduler = LambdaLR(
        optimizer=optimizer,
        lr_lambda=lambda step: rate(
            step, config.D_MODEL, factor=1, warmup=config.WARMUP_STEPS
        ),
    )

    for epoch in range(config.NUM_EPOCHS):
        model.train()
        total_loss = []
        for i, batch in enumerate(
            tqdm(train_loader, desc=f"Epoch[{epoch}] Training batch")
        ):
            user, user_mask, item, item_mask, item_y, item_mask_y = batch
            user_embed, item_embed = model(
                user,
                user_mask.unsqueeze(-2),
                item,
                create_user_target_mask(item_mask.unsqueeze(-2)),
                item_y,
                create_item_encoder_mask(item_mask_y.unsqueeze(-2)),
            )
            target = torch.arange(user_embed.size(
                0) * user_embed.size(1)).to(config.DEVICE)
            user_embed = user_embed.view(-1, config.D_MODEL)
            item_embed = item_embed.view(-1, config.D_MODEL)
            res = torch.matmul(user_embed, item_embed.t())
            loss = ce_loss(res, target)
            loss.backward()
            if i % config.ACCUM_ITER == 0:
                optimizer.step()
                optimizer.zero_grad(set_to_none=True)
            lr_scheduler.step()
            total_loss.append(loss.item())

            del loss

        train_loss = sum(total_loss) / len(total_loss)
        logging.info(f"Epoch: {epoch}")
        logging.info(f"Train Loss: {train_loss}")
        if epoch % config.EVAL_EPOCHS == 0:
            torch.save(model.state_dict(), os.path.join(
                args.save_path, f"{epoch}-model.pt"))
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

            with torch.no_grad():
                model.eval()
                critic = Evaluator(
                    batch_size=config.BATCH_SIZE,
                    num_eval_seq=config.NUM_EVAL_SEQ,
                    model=model,
                    d_model=config.D_MODEL,
                    lookup_size=config.LOOKUP_SIZE,
                    val_loader=val_loader,
                    eval_ks=config.EVAL_Ks,
                    tokenizer=tokenizer,
                    out_dir=args.metrics_path,
                )
                _ = critic.evaluate(epoch_train=epoch, desc="val")
                del critic

    logging.info("Final evaluation with test set")
    with torch.no_grad():
        model.eval()
        critic = Evaluator(
            batch_size=config.BATCH_SIZE,
            num_eval_seq=config.NUM_EVAL_SEQ,
            model=model,
            d_model=config.D_MODEL,
            lookup_size=config.LOOKUP_SIZE,
            val_loader=test_loader,
            eval_ks=config.EVAL_Ks,
            tokenizer=tokenizer,
            out_dir=args.metrics_path,
        )
        _ = critic.evaluate(epoch_train=epoch, desc="final_test")


def parse_args(description):

    parser = ArgumentParser(description=description)
    parser.add_argument('--save_path', type=str, default='./checkpoint/')
    parser.add_argument('--metrics_path', type=str, default='./metrics/')
    parser.add_argument('--tokenizer_save_name', type=str,
                        default='tokenizer.json')
    parser.add_argument("--data_path", type=str, required=True)
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument("--max_len", type=int, default=22)
    parser.add_argument("--min_seq_len", type=int, default=5)
    parser.add_argument("--test_frac", type=int, default=0.3)
    args = parser.parse_args()
    return args


if __name__ == "__main__":
    logging.basicConfig(stream=sys.stdout, level=logging.INFO,
                        format='\n%(message)s')
    sys.exit(main(parse_args("Run training pipeline.")))
