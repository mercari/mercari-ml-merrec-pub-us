"""
Main script for training and evaluating CTR and MTL models.
Some parts of this script are derived from:
- https://github.com/yuangh-x/2022-NIPS-Tenrec/blob/master/main.py
- https://github.com/yuangh-x/2022-NIPS-Tenrec/blob/master/utils.py
"""

import os
import random
import sys
import numpy as np
from argparse import ArgumentParser

import torch

from data import (
    mtl_dataset,
    MTLDataset,
    get_data_loader,
    ctr_dataset,
)
from model_def.ctr.deepfm import DeepFM
from model_def.ctr.xdeepfm import xDeepFM
from model_def.ctr.nfm import NFM
from model_def.ctr.wdl import WDL
from model_def.ctr.afm import AFM
from model_def.ctr.dcn import DCN
from model_def.ctr.dcnmix import DCNMix
from model_def.esmm import ESMM
from model_def.mmoe import MMOE
from train import mtl_schedular, ctr_schedular


def parse_args(description):
    parser = ArgumentParser(description=description)
    parser.add_argument('--device', default='cuda')
    parser.add_argument("--task_name", type=str, default='mtl')
    parser.add_argument('--save_path', type=str, default='./checkpoint/')
    parser.add_argument("--data_path", type=str, required=True)
    parser.add_argument('--seed', type=int, default=100)
    parser.add_argument("--model_name", type=str, default='mmoe')
    parser.add_argument('--mtl_task_num', type=int, default=2, help='0:like, 1:view, 2:both tasks')
    parser.add_argument("--train_batch_size", type=int, default=32)
    parser.add_argument("--val_batch_size", type=int, default=32)
    parser.add_argument("--test_batch_size", type=int, default=32)
    parser.add_argument("--is_parallel", type=bool, default=False)
    parser.add_argument("--embedding_size", type=int, default=32)
    parser.add_argument("--lr", type=float, default=0.0001)
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument('--local_rank', type=int)
    args = parser.parse_args()
    return args


def set_seed(seed, re=True):
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed) # if you are using multi-GPU.
    if re:
        torch.backends.cudnn.benchmark = False
        torch.backends.cudnn.deterministic = True
    else:
        torch.backends.cudnn.benchmark = True
        torch.backends.cudnn.deterministic = False


def main(args):
    if args.is_parallel:
        torch.distributed.init_process_group(backend="nccl")
        torch.cuda.set_device(args.local_rank)
    set_seed(args.seed)
    print(f"Input arguments: {args}")

    # Load dataset
    if args.task_name == 'ctr':
        train, test, train_model_input, test_model_input, lf_columns, df_columns = ctr_dataset(args.data_path)
    elif args.task_name == 'mtl':
        train_data, val_data, test_data, user_feature_dict, item_feature_dict = mtl_dataset(
            args.data_path, args.mtl_task_num,
        )
        if args.mtl_task_num == 2:
            train_dataset = (train_data.iloc[:, :-2].values, train_data.iloc[:, -2].values, train_data.iloc[:, -1].values)
            val_dataset = (val_data.iloc[:, :-2].values, val_data.iloc[:, -2].values, val_data.iloc[:, -1].values)
            test_dataset = (test_data.iloc[:, :-2].values, test_data.iloc[:, -2].values, test_data.iloc[:, -1].values)
        else:
            train_dataset = (train_data.iloc[:, :-1].values, train_data.iloc[:, -1].values)
            val_dataset = (val_data.iloc[:, :-1].values, val_data.iloc[:, -1].values)
            test_dataset = (test_data.iloc[:, :-1].values, test_data.iloc[:, -1].values)
        train_dataset = MTLDataset(train_dataset, args.mtl_task_num)
        val_dataset = MTLDataset(val_dataset, args.mtl_task_num)
        test_dataset = MTLDataset(test_dataset, args.mtl_task_num)

        # dataloader
        train_dataloader = get_data_loader(train_dataset, batch_size=args.train_batch_size, is_parallel=args.is_parallel, is_train=True)
        val_dataloader = get_data_loader(val_dataset, batch_size=args.val_batch_size, is_parallel=args.is_parallel, is_train=False)
        test_dataloader = get_data_loader(test_dataset, batch_size=args.test_batch_size, is_parallel=args.is_parallel, is_train=False)

    # Load and train model
    if args.task_name == 'ctr':
        if args.model_name == 'deepfm':
            model = DeepFM(lf_columns, df_columns, task='binary', device=args.device)
        elif args.model_name == 'nfm':
            model = NFM(lf_columns, df_columns, task='binary', device=args.device)
        elif args.model_name == 'xdeepfm':
            model = xDeepFM(lf_columns, df_columns, task='binary', device=args.device)
        elif args.model_name == 'wdl':
            model = WDL(lf_columns, df_columns, task='binary', device=args.device)
        elif args.model_name == 'afm':
            model = AFM(lf_columns, df_columns, task='binary', device=args.device)
        elif args.model_name == 'dcn':
            model = DCN(lf_columns, df_columns, task='binary', device=args.device)
        elif args.model_name == 'dcnmix':
            model = DCNMix(lf_columns, df_columns, task='binary', device=args.device)
        ctr_schedular(model, train_model_input, train['item_view'].values, test_model_input, test['item_view'].values, args)
    elif args.task_name == 'mtl':
        if args.mtl_task_num == 2:
            num_task = 2
        else:
            num_task = 1
        if args.model_name == 'esmm':
            model = ESMM(user_feature_dict, item_feature_dict, emb_dim=args.embedding_size, num_task=num_task)
        else:
            model = MMOE(user_feature_dict, item_feature_dict, emb_dim=args.embedding_size, device=args.device, num_task=num_task)
        mtl_schedular(model, train_dataloader, val_dataloader, test_dataloader, args)
    else:
        raise NotImplementedError(f"task choice {args.task_name} not implemented.")


if __name__ == "__main__":
    sys.exit(main(parse_args("Run training pipeline.")))