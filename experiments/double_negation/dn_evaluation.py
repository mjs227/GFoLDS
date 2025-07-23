
import gc
import json
import torch
from tqdm import tqdm
from setup import GLOBAL_FP
from model.io import SWATForSCInput
from model.model import SWATForSequenceClassification


EVAL_N = 5 
EVAL_RANGE = (EVAL_N, 7)  # range of ext. neg. prefixes to eval. models on (EVAL_RANGE[0] should be > INOC_N)
BATCH_SIZE = 64
ADV_FP = GLOBAL_FP + '/double_negation/'
INIT_CHK_FP = GLOBAL_FP + '/nli/runs/gfolds/checkpoints/ep4.chk'


def generate_data(data_list):
    for x in data_list:
        yield SWATForSCInput.from_dir_graph(x['g'], targets=label_map[x['lbl']])


def evaluate(model_, eval_file, use_tqdm=True):
    eval_batches = SWATForSCInput.generate_batches(generate_data(eval_file), batch_size=BATCH_SIZE)
    tqdm_ = tqdm if use_tqdm else lambda z: z
    acc_total, acc_cnt = 0, 0

    with torch.no_grad():
        for _ in tqdm_(range(-(-len(eval_file) // BATCH_SIZE))):
            try:
                eval_batch = next(eval_batches)
                eval_out = model_(eval_batch)
                acc_total += eval_out.accuracy() * eval_batch.batch_size
                acc_cnt += eval_batch.batch_size

                del eval_batch, eval_out
            except StopIteration:
                break

    return acc_total * 100 / acc_cnt


_DEVICE = 0 if torch.cuda.is_available() else 'cpu'
label_map = {y: x for x, y in enumerate(('e', 'n', 'c'))}
eval_data, stat_file = {}, {'inoc': {}}

for n in range(EVAL_RANGE[0], EVAL_RANGE[1] + 1):
    with open(f'{ADV_FP}data/adv_{n}.json', 'r') as f:
        eval_data.update({n: json.load(f)})

if INIT_CHK_FP is not None:
    model = SWATForSequenceClassification.from_pretrained(INIT_CHK_FP)
    model.to(device=_DEVICE)
    model.eval()
    stat_file.update({'init': {}})

    for n in range(EVAL_RANGE[0], EVAL_RANGE[1] + 1):
        init_acc_n = evaluate(model, eval_data[n])
        stat_file['init'].update({n: init_acc_n})

        print()
        print(f'Initial depth-{n} accuracy: {round(init_acc_n, 3)}%')
        print()

    del model
    gc.collect()
    torch.cuda.empty_cache()

model = SWATForSequenceClassification.from_pretrained(f'{ADV_FP}checkpoints/depth{EVAL_N}.chk')
model.to(device=_DEVICE)
model.eval()

for n in range(EVAL_RANGE[0], EVAL_RANGE[1] + 1):
    inoc_acc_n = evaluate(model, eval_data[n])
    stat_file['inoc'].update({n: inoc_acc_n})

    print()
    print(f'Inoculated depth-{n} accuracy: {round(inoc_acc_n, 3)}%')
    print()

with open(f'{ADV_FP}stats/depth{EVAL_N}_eval.json', 'w') as f:
    json.dump(stat_file, f)
