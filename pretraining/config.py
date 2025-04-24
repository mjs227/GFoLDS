
import torch
from global_filepath import GLOBAL_FP
from model.configs import SWATForMaskedLMConfig


FILEPATH, RUN = f'{GLOBAL_FP}/pretraining/', 0
CHECKPOINT_FILEPATH = f'{FILEPATH}run_{RUN}/checkpoints/'
OPTIMIZER_FILEPATH = f'{FILEPATH}run_{RUN}/optimizers/'
BASE_DATA_FILEPATH = f'{FILEPATH}data/'  # no effect if RANDOMIZE_BATCHES = False
BATCH_FILEPATH = f'{FILEPATH}run_{RUN}/batch_data/'  # empty if RANDOMIZE_BATCHES, else folders {0, ..., N_EPOCHS}
STATS_FILEPATH = f'{FILEPATH}run_{RUN}/stats/'
LOSS_WEIGHT_FILE = None

MODEL_CONFIG = SWATForMaskedLMConfig(
    swat_config__d_model=1024,
    swat_config__n_encoder_layers=10,
    swat_config__embedding_config__d_swa=1024,
    swat_config__embedding_config__n_swa_layers=2
)

MODEL_TO = {'device': 0}  # do not include device for train_ddp
TOKEN_DTYPE = None  # None => default
DDP = False  # False => train.py, True => train_ddp.py

N_EPOCHS = 5
CHECKPOINT_RATE = 10  # in number of meta-batches (ALWAYS checkpoints at end of epoch)
CHECKPOINT_ON_ERROR = True  # checkpoint if exception occurs, regardless of checkpoint rate
CHECKPOINT_ON_KEYBOARD_INTERRUPT = True
BATCH_SIZE = 16
COLLECT_BATCHES = False
RANDOMIZE_BATCHES = True  # True => randomize during training, False => pre-randomized
PRINT_OUTPUT = False  # print loss (disable for non-debug CCR)
TQDM = False  # tqdm train loop (disable for ALL CCR) --- no effect if PRINT_OUTPUT = FALSE
PRINT_RATE = 25  # print loss every PRINT_RATE batches (no effect if PRINT_OUTPUT = false)

# AdamW for weight decay!!!
OPTIMIZER = torch.optim.AdamW  # torch.optim for train, str for train_ddp
OPTIMIZER_KWARGS = {'weight_decay': 1e-5}
INIT_LR = 1e-5  # if len(LR_VALS) == 0, then lr = PEAK_LR
LR_VALS = [2e-5, 1e-5, 3e-6, 1e-6, 1e-8]  # must be same len as LR_STEP
LR_STEP = [0.2, 0.4, 0.6, 0.8, 1.0]

SPEC_TOK_P_PROB = 0.0  # special token ([MASK], [PAD], [UNK]) perturb prob.
P_PROB = 0.2  # non-special token perturb prob.
M_PROB = 1.0  # mask prob. (of those perturbed)
S_PROB = 0.0  # swap prob. (of those perturbed)
