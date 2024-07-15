# MerRec Recommendation Dataset

<img src='assets/logo.png' width='400'/>

This repository contains the experiment code for running example model benchmarks and data processing that accompanies the paper [MerRec: A Large-scale Multipurpose Mercari Dataset for Consumer-to-Consumer Recommendation Systems](https://arxiv.org/abs/2402.14230). This repository is only an example demonstration of how the MerRec dataset can be used in terms of recommendation tasks, and does not depict or reflect production implementation at Mercari.

# Benchmark Execution

## Session-based Recommendation (SBR)

In the SBR tasks, the raw data is converted to a processed sequences in the memory itself. We don't need to run pre-processing separately. Below are the commands to run various SBR models on the benchmark data.

NextItNet:

```bash
python main.py --task_name=sequence --seed=100 --model_name=nextitnet --data_path='data/20230501' --train_batch_size=32 --val_batch_size=32 --test_batch_size=32 --epochs=20 --lr=0.0001 --hidden_size=128 --block_num=8 --embedding_size=128 --kernel_size=3 --is_pretrain=1
```

Bert4Rec:

```bash
python main.py --task_name=sequence --seed=100 --model_name=bert4rec --data_path='data/20230501' --train_batch_size=32 --val_batch_size=32 --test_batch_size=32 --epochs=20 --lr=0.0001 --hidden_size=128 --block_num=16 --embedding_size=128 --num_heads=4 --mask_prob=0.3 --is_pretrain=1
```

GRU4Rec:

```bash
python main.py --task_name=sequence --seed=100 --model_name=gru4rec --data_path='data/20230501' --train_batch_size=32 --val_batch_size=32 --test_batch_size=32 --epochs=20 --lr=0.0005 --hidden_size=64 --block_num=8 --embedding_size=64 --is_pretrain=1
```

SASRec:

```bash
python main.py --task_name=sequence --seed=100 --model_name=sasrec --data_path='data/20230501' --train_batch_size=32 --val_batch_size=32 --test_batch_size=32 --epochs=20 --lr=0.0001 --hidden_size=64 --block_num=8 --embedding_size=64 --num_heads=4 --is_pretrain=1
```

## Data Preprocessing for CTR and MTL Tasks

In both CTR task and MTL task below, the raw dataset first needs to be transformed.

Based on `product_id`:

```bash
python preprocess_mtl.py --out_path='data/mtl_product.csv' --local_dir_path='data/20230501'
```

## Click-through Rate (CTR) Prediction

Attention FM (AFM):

```bash
python main_ctr_mtl.py --task_name=ctr --seed=100 --model_name=afm --data_path='data/mtl_product.csv' --train_batch_size=4096 --test_batch_size=4096 --epochs=20 --lr=0.00005
```

DeepFM:

```bash
python main_ctr_mtl.py --task_name=ctr --seed=100 --model_name=deepfm --data_path='data/mtl_product.csv' --train_batch_size=4096 --test_batch_size=4096 --epochs=20 --lr=0.00005
```

xDeepFM:

```bash
python main_ctr_mtl.py --task_name=ctr --seed=100 --model_name=xdeepfm --data_path='data/mtl_product.csv' --train_batch_size=4096 --test_batch_size=4096 --epochs=20 --lr=0.00005
```

DCN:

```bash
python main_ctr_mtl.py --task_name=ctr --seed=100 --model_name=dcn --data_path='data/mtl_product.csv' --train_batch_size=4096 --test_batch_size=4096 --epochs=20 --lr=0.00005
```

DCNv2 (DCNMIX):

```bash
python main_ctr_mtl.py --task_name=ctr --seed=100 --model_name=dcnmix --data_path='data/mtl_product.csv' --train_batch_size=4096 --test_batch_size=4096 --epochs=20 --lr=0.00005
```

NeuralFM (NFM):

```bash
python main_ctr_mtl.py --task_name=ctr --seed=100 --model_name=nfm --data_path='data/mtl_product.csv' --train_batch_size=4096 --test_batch_size=4096 --epochs=20 --lr=0.00005
```

Wide & Deep:

```bash
python main_ctr_mtl.py --task_name=ctr --seed=100 --model_name=wdl --data_path='data/mtl_product.csv' --train_batch_size=4096 --test_batch_size=4096 --epochs=20 --lr=0.00005
```

## Multi-task Learning (MTL) for Recommendation

Only `item_view` with MMOE:

```bash
python main_ctr_mtl.py --task_name=mtl --seed=100 --model_name=mmoe --data_path='data/mtl_product.csv' --train_batch_size=4096 --val_batch_size=4096 --test_batch_size=4096 --epochs=20 --lr=0.0001 --embedding_size=32 --mtl_task_num=1
```

Only `item_like` with MMOE:

```bash
python main_ctr_mtl.py --task_name=mtl --seed=100 --model_name=mmoe --data_path='data/mtl_product.csv' --train_batch_size=4096 --val_batch_size=4096 --test_batch_size=4096 --epochs=20 --lr=0.0001 --embedding_size=32 --mtl_task_num=0
```

2-task ESMM:

```bash
python main_ctr_mtl.py --task_name=mtl --seed=100 --model_name=esmm --data_path='data/mtl_product.csv' --train_batch_size=4096 --val_batch_size=4096 --test_batch_size=4096 --epochs=20 --lr=0.0001 --embedding_size=32 --mtl_task_num=2
```

2-task MMOE:

```bash
python main_ctr_mtl.py --task_name=mtl --seed=100 --model_name=mmoe --data_path='data/mtl_product.csv' --train_batch_size=4096 --val_batch_size=4096 --test_batch_size=4096 --epochs=20 --lr=0.0001 --embedding_size=32 --mtl_task_num=2
```

## Model Inference Acceleration

Skip-SASRec

```bash
python main.py --task_name=inference_acc --seed=5 --model_name=sas4infacc --data_path='data/20230501' --train_batch_size=32 --val_batch_size=32 --test_batch_size=1 --epochs=20 --lr=0.0001 --hidden_size=64 --block_num=8 --embedding_size=64 --num_heads=4 --is_pretrain=1
```

Skip-NextItNet

```bash
python main.py --task_name=inference_acc --seed=5 --model_name=skiprec --data_path='data/20230501' --train_batch_size=32 --val_batch_size=32 --test_batch_size=1 --epochs=20 --lr=0.0001 --hidden_size=128 --block_num=8 --embedding_size=128 --dilation=1,4 --kernel_size=3 --is_pretrain=1
```

# BibTeX

```bibtex
@misc{li2024merrec,
      title={MerRec: A Large-scale Multipurpose Mercari Dataset for Consumer-to-Consumer Recommendation Systems}, 
      author={Lichi Li and Zainul Abi Din and Zhen Tan and Sam London and Tianlong Chen and Ajay Daptardar},
      year={2024},
      eprint={2402.14230},
      archivePrefix={arXiv},
      primaryClass={cs.IR}
}
```

# License

- Codebase: This codebase is licensed under the MIT license.
- Dataset: The [MerRec dataset](https://huggingface.co/datasets/mercari-us/merrec) is licensed under [CC BY-NC 4.0 International](https://creativecommons.org/licenses/by-nc/4.0/legalcode.en).

# How to Contribute

Contributions are welcomed. Please read the CLA carefully before submitting your contribution to Mercari. Under any circumstances, by submitting your contribution, you are deemed to accept and agree to be bound by the terms and conditions of the CLA.

https://www.mercari.com/cla/

# Acknowledgement

We would like to thank Guanghu Yuan et al. for their work [Tenrec: A Large-scale Multipurpose Benchmark Dataset for Recommender Systems](https://arxiv.org/abs/2210.10629) and making the [code](https://github.com/yuangh-x/2022-NIPS-Tenrec) publicly available and for the extensive documentation. Many of our experiment implementation centered on `product_id` in CTR, MTL and SBR tasks derived from this work.
