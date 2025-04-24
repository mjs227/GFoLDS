
import torch
from global_filepath import GLOBAL_FP


FILEPATH, RUN = f'{GLOBAL_FP}/bert/', 'lrg_2'
CHECKPOINT_FILEPATH = FILEPATH + f'runs/run_{RUN}/checkpoints/'
OPTIMIZER_FILEPATH = FILEPATH + f'runs/run_{RUN}/optimizers/'
BASE_DATA_FILEPATH = FILEPATH + 'data/nsp/'  # no effect if RANDOMIZE_BATCHES = False
BATCH_FILEPATH = FILEPATH + f'runs/run_{RUN}/batch_data/'  # empty if RANDOMIZE_BATCHES, else folders {0, ..., N_EPOCHS}
STATS_FILEPATH = FILEPATH + f'runs/run_{RUN}/stats/'

BERT_MODEL = 'bert-base-uncased'
MODEL_TO = {'device_map': 'auto'}  # do not include device for train_ddp
TOKEN_DTYPE = None  # None => default

N_EPOCHS = 4
CHECKPOINT_RATE = 10  # in number of meta-batches (ALWAYS checkpoints at end of epoch)
CHECKPOINT_ON_ERROR = False  # checkpoint if exception occurs, regardless of checkpoint rate
CHECKPOINT_ON_KEYBOARD_INTERRUPT = False
BATCH_SIZE = 16
COLLECT_BATCHES = False
RANDOMIZE_BATCHES = True  # True => randomize during training, False => pre-randomized
PRINT_OUTPUT = False  # print loss (disable for non-debug CCR)
TQDM = False  # tqdm train loop (disable for ALL CCR) --- no effect if PRINT_OUTPUT = FALSE
PRINT_RATE = 25  # print loss every PRINT_RATE batches (no effect if PRINT_OUTPUT = false)

# AdamW for weight decay!!!
OPTIMIZER = torch.optim.AdamW  # torch.optim for train, str for train_ddp
OPTIMIZER_KWARGS = {'weight_decay': 0.01}
INIT_LR = 2e-8  # if len(LR_VALS) == 0, then lr = PEAK_LR
LR_VALS = [2e-5, 0.0]  # must be same len as LR_STEP
LR_STEP = [0.01, 1.0]

P_PROB = 0.15  # non-special token perturb prob.
M_PROB = 0.8  # mask prob. (of those perturbed)
S_PROB = 0.1  # swap prob. (of those perturbed)
MASK_ID = 103
PAD_ID = 0
