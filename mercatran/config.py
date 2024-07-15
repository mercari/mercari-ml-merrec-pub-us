import torch

CATEGORY_NAME_HIERARCHY = ["c2_name", "c1_name", "c0_name"]
CATEGORY_ID_HIERARCHY = ["c2_id", "c1_id", "c0_id"]

BATCH_SIZE = 3072
BPE_VOCAB_LIMIT = 32768
MODEL_SEQ_LEN = 22
D_MODEL = 64
D_FF = 1024
NUM_HEADS = 8
NUM_STACKS = 2
POSITION_MAX_LEN = 5000
DROPOUT = 0.1
BETAS = (0.9, 0.999)
WEIGHT_DECAY = 0.01
EPS = 1e-08
MAX_NORM = 1.0
BASE_LR = 1.0
WARMUP_STEPS = 3000
NUM_EPOCHS = 21
EVAL_EPOCHS = 3
NUM_EVAL_SEQ = 4
EVAL_Ks = [5, 20]
LOOKUP_SIZE = int(max(EVAL_Ks))
ACCUM_ITER = 1
DEVICE = "cuda:0" if torch.cuda.is_available() else "cpu"

# Step LR only
DECAY_STEP = 25
GAMMA = 1.0
