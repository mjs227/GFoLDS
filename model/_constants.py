
from copy import deepcopy


def str_(x):
    return f'\'{x}\'' if isinstance(x, str) else str(x)


def dict_kwargs(d, default=None):
    return ({} if default is None else deepcopy(default)) if d is None else d


def type_str(obj, incl_path=False):
    if incl_path:
        return str(type(obj)).rsplit(' ', 1)[1][1:-2]

    return str(type(obj)).rsplit('.', 1)[1][:-2]


INDENT_FACTOR = 3

VOCAB_SIZE = 22077
FEATURE_SIZE = 31
MAX_SEQ_LEN = 512
SPEC_TOK_RNG = 3
PAD_TOKEN_ID = 0
MASK_TOKEN_ID = 1
IGNORE_ID = -100

DEFAULT_D_MODEL = 1024
DEFAULT_D_SWA = 1024
DEFAULT_N_ENC = 8
DEFAULT_N_SWA = 2
DEFAULT_N_HEADS = 8
DEFAULT_P_DROPOUT = 0.1

LABEL, IDX, ID = 0, 1, 2
PROHIBITED_TOKENS = {'focus_d', 'parg_d'}
LINK_ROLE_MAP = {'L-INDEX': 'INDEX', 'R-INDEX': 'INDEX', 'L-HNDL': 'HNDL', 'R-HNDL': 'HNDL', 'ARG': 'MOD'}
DEFAULT_TOKENIZER_DICT = {
    'v': {
        '[PAD]': 0,
        '[MASK]': 1,
        '[UNK]': 2
    },
    'e': {
        'ARG1': 0,
        'ARG2': 1,
        'ARG3': 2,
        'ARG4': 3,
        'MOD': 4,
        'RSTR': 5,
        'INDEX': 6,
        'HNDL': 7,
    },
    'f': {
        '[PAD]': 0,
        '[UNK]': 1
    }
}
EDGE_SIZE = len(DEFAULT_TOKENIZER_DICT['e'])
