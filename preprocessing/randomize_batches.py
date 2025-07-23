
import re
import os
import json
import random
from tqdm import tqdm
from setup import GLOBAL_FP
from argparse import ArgumentParser

ap = ArgumentParser()
ap.add_argument('-n', '--num_epochs', type=int, default=1)
ap.add_argument('-b', '--bert', help='Generate BERT pretraining data', action='store_true')
ap.add_argument('--nsp', help='Create NSP data (BERT only)', action='store_true')
ap.add_argument('-k', '--n_to_keep', help='NSP only. Default = 17460320/2 = 8730160.', type=int, default=None)
args = ap.parse_args()
assert not (args.nsp and not args.bert)
IGNORE_IDXS = {}  # to withhold validation batch

if args.n_to_keep is None:
    n_to_keep = 8730160 if args.nsp else None
else:
    assert args.nsp

if args.bert:
    if args.nsp:
        DATA_FP = f'{GLOBAL_FP}/bert/data/nsp/batch_data/'
        SRC_FP = f'{GLOBAL_FP}/bert/data/nsp/preproc_data/'
    else:
        DATA_FP = f'{GLOBAL_FP}/bert/mlm/batch_data/'
        SRC_FP = f'{GLOBAL_FP}/bert/mlm/preproc_data/parsed_files/'
else:
    DATA_FP = f'{GLOBAL_FP}/pretraining/data/batch_data/'
    SRC_FP = f'{GLOBAL_FP}/pretraining/data/parsed_graphs/parsed_files/'

max_ep = max(map(int, os.listdir(DATA_FP)), default=-1) + 1
folder_file_names = os.listdir(SRC_FP)
folder_files = []

for ff in tqdm(folder_file_names):
    ff_idx = int(re.sub(r'\D', '', ff))

    if ff_idx not in IGNORE_IDXS:
        with open(f'{SRC_FP}/{ff}', 'r') as f:
            folder_files.append(json.load(f))

file_idxs = []

for i, ff in tqdm(enumerate(folder_files)):
    file_idxs.extend((i, k) for k in range(len(ff)))

num_batch, total_len, len_cnt = len(folder_files), len(file_idxs), 0
batch_size = -(-total_len // num_batch)

if args.nsp:
    random.shuffle(file_idxs)
    file_idxs = file_idxs[:8730160]

print()
print(f'TOTAL LEN: {total_len}')
print()

for n in range(max_ep, max_ep + args.num_epochs):
    random.shuffle(file_idxs)
    os.mkdir(f'{DATA_FP}{n}')

    for i in tqdm(range(num_batch)):
        batch_out_list = [folder_files[f_idx][e_idx] for f_idx, e_idx in file_idxs[i * batch_size:(i + 1) * batch_size]]
        random.shuffle(batch_out_list)
        len_cnt += len(batch_out_list)

        with open(f'{DATA_FP}{n}/{i}', 'w') as f:
            json.dump(batch_out_list, f)

        print()
        print(f'BATCH LEN = {len(batch_out_list)}, TOTAL_LEN = {len_cnt}/{total_len}')
        print()
