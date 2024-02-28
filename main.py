# Some parts derived from: https://github.com/yuangh-x/2022-NIPS-Tenrec/blob/master/main.py
import os
import pathlib
import random
import sys
import time
from argparse import ArgumentParser

import torch
from torch.utils.tensorboard import SummaryWriter

from data import (BertTrainDataset, EvalDataset, TrainDataset, get_data_loader,
                  sequence_dataset, train_val_test_split)
from model_def.bert4rec import BERTModel
from model_def.gru4rec import GRU4Rec
from model_def.nextitnet import NextItNet
from model_def.sasrec import SASRec
from model_def.sas4infacc import SAS4infaccModel, SAS_PolicyNetGumbel
from model_def.skiprec import SkipRec, PolicyNetGumbel
from train import train_val_schedular, validator, inference_acc_schedular, paired_validator


def list_of_ints(arg):
    return [int(i) for i in arg.split(',')]


def parse_args(description):

    parser = ArgumentParser(description=description)
    parser.add_argument('--device', default='cuda')
    parser.add_argument('--save_path', type=str, default='./checkpoint/')
    parser.add_argument("--data_path", type=str, required=True)
    parser.add_argument('--seed', type=int, default=100) # seed default = 0
    parser.add_argument("--num_users", type=int, default=0)
    parser.add_argument("--num_products", type=int, default=0)
    parser.add_argument("--max_len", type=int, default=20)
    parser.add_argument("--min_seq_len", type=int, default=5)
    parser.add_argument("--pad_token", type=int, default=0)
    parser.add_argument("--train_batch_size", type=int, default=32)
    parser.add_argument("--val_batch_size", type=int, default=32)
    parser.add_argument("--test_batch_size", type=int, default=32)
    parser.add_argument("--valid_rate", type=int, default=100)
    parser.add_argument("--sample_prob", type=float, default=0.11)
    parser.add_argument("--is_parallel", type=bool, default=False)
    parser.add_argument("--model_name", type=str, required=True)
    parser.add_argument("--embedding_size", type=int, default=64)
    parser.add_argument('--hidden_size', type=int, default=64, help='Size of hidden vectors (model)')
    parser.add_argument("--epochs", type=int, default=2)
    parser.add_argument("--lr", type=float, default=0.0001)
    parser.add_argument("--weight_decay", type=float, default=0.0)
    parser.add_argument("--dropout", type=float, default=0.3)

    parser.add_argument("--task_name", type=str, default='sequence')
    parser.add_argument("--task", type=int, default=-1)

    # bert4Rec
    parser.add_argument("--mask_prob", type=float, default=0.3)
    parser.add_argument("--num_heads", type=int, default=4)
    
    # nextitnet
    parser.add_argument("--block_num", type=int, default=8)
    parser.add_argument("--dilations", type=list_of_ints, default=[1, 4])
    parser.add_argument("--kernel_size", type=int, default=3)

    # eval params
    parser.add_argument('--k', type=int, default=20, help='The number of items to measure the hit@k metric (i.e. hit@10 to see if the correct item is within the top 10 scores)')
    parser.add_argument('--metric_ks', nargs='+', type=int, default=[5, 20], help='ks for Metric@k')
    parser.add_argument('--eval', type=bool, default=True)

    # transfer learning
    parser.add_argument('--is_pretrain', type=int, default=1, help='0: mean transfer, 1: mean pretrain, 2:mean train full model without transfer')

    # inference acceleration
    parser.add_argument('--temp', type=int, default=7)
 
    args = parser.parse_args()
    return args


def get_model(args):
    if args.model_name == 'nextitnet':
        model = NextItNet(args=args)
    elif args.model_name == 'bert4rec':
        model = BERTModel(args)
    elif args.model_name == 'gru4rec':
        model = GRU4Rec(args)
    elif args.model_name == 'sasrec':
        model = SASRec(args)
    elif args.model_name == 'skiprec':
        model = (SkipRec(args), PolicyNetGumbel(args))
    elif args.model_name == 'sas4infacc':
        model = (SAS4infaccModel(args), SAS_PolicyNetGumbel(args))
    return model


def main(args):
    if args.task_name not in ['sequence', 'inference_acc']:
        raise ValueError("Invalid task option.")
    if args.model_name == 'inference_acc':
        args.test_batch_size = 1

    rng = random.Random(args.seed)
    writer = SummaryWriter()
    print("Model Name: ", args.model_name)
    model_dir = pathlib.Path(args.save_path)
    model_dir.mkdir(parents=True, exist_ok=True)
    data_path = args.data_path
    sequences, product_count, user_count = sequence_dataset(
        path=data_path, min_seq_len=args.min_seq_len, sample_prob=args.sample_prob)
    print("Len of filtered sequences: ", len(sequences))
    args.num_users = user_count
    args.num_products = product_count
    print("Num products: ", args.num_products)
    train_data, val_data, test_data = train_val_test_split(sequences=sequences)
    # sample the val dataset, since validation epoch takes a lot of time
    cnt = 0; train_data_val_small = {}; val_data_val_small = {}
    for key, _ in val_data.items():
        train_data_val_small[key] = train_data[key]
        val_data_val_small[key] = val_data[key]
        cnt += 1
        if cnt == int(len(train_data) / args.valid_rate):
            break
    if 'bert' in args.model_name:
        train_dataset = BertTrainDataset(train_data, args.max_len, args.mask_prob, args.pad_token, args.num_products, rng)
    else:
        train_dataset = TrainDataset(train_data, args.max_len, args.pad_token)
    val_dataset = EvalDataset(train_data_val_small, val_data_val_small, args.max_len, args.pad_token, args.num_products)
    test_dataset = EvalDataset(train_data, test_data, args.max_len, args.pad_token, args.num_products)
    train_loader = get_data_loader(
        train_dataset, batch_size=args.train_batch_size, is_parallel=args.is_parallel, is_train=True
    )
    val_loader = get_data_loader(
        val_dataset, batch_size=args.val_batch_size, is_parallel=args.is_parallel, is_train=False
    )
    test_loader = get_data_loader(
        test_dataset, batch_size=args.test_batch_size, is_parallel=args.is_parallel, is_train=False
    )

    # Task specific model loading
    if args.task_name == 'sequence':
        model = get_model(args).to(args.device)
    elif args.task_name == 'inference_acc':
        backbonenet, policynet = get_model(args)
        backbonenet = backbonenet.to(args.device)
        policynet = policynet.to(args.device)

    since = time.time()
    if args.task_name == 'sequence':
        _ = train_val_schedular(args.epochs, model, train_loader, val_loader, writer, args)
    elif args.task_name == 'inference_acc':
        _,_ = inference_acc_schedular(args.epochs, backbonenet, policynet, train_loader, val_loader, writer, args)
    print("Total time to train: ", time.time() - since)

    if args.eval:
        if args.task_name == 'sequence':
            best_model = torch.load(
                os.path.join(
                    args.save_path, 
                    '{}_{}_seed{}_is_pretrain_{}_best_model_lr{}_wd{}_block{}_hd{}_emb{}.pth'.format(
                        args.task_name, args.model_name, args.seed, args.is_pretrain, args.lr, 
                        args.weight_decay, args.block_num, args.hidden_size, args.embedding_size
                    )
                )
            )
            model.load_state_dict(best_model)
            model = model.to(args.device)
            since_val = time.time()
            _ = validator(0, model=model, dataloader=test_loader, writer=writer, args=args, test=False)
        elif args.task_name == 'inference_acc':
            best_policy = torch.load(
                os.path.join(
                    args.save_path,
                    '{}_{}_seed{}_lr{}_block{}_best_policynet.pth'.format(
                        args.task_name, args.model_name, args.seed, args.lr, args.block_num
                    )
                )
            )
            best_backbone = torch.load(
                os.path.join(
                    args.save_path,
                    '{}_{}_seed{}_lr{}_block{}_best_backbone.pth'.format(
                        args.task_name, args.model_name, args.seed, args.lr, args.block_num
                    )
                )
            )
            policynet.load_state_dict(best_policy)
            backbonenet.load_state_dict(best_backbone)
            policynet = policynet.to(args.device)
            backbonenet = backbonenet.to(args.device)
            since_val = time.time()
            metrics = paired_validator(0, backbonenet, policynet, test_loader, writer, args, test=True)
            print(metrics)
            print('[Inf Acc] inference_time:', backbonenet.all_time)
        print("Total time to test: ", time.time() - since_val)
    
    writer.close()


if __name__ == "__main__":
    sys.exit(main(parse_args("Run training pipeline.")))