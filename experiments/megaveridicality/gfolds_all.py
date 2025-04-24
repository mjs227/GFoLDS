
import gc
import os
import json
import torch
from tqdm import tqdm
from model.io import SWATForSCInput
from global_filepath import GLOBAL_FP
from model.model import SWATForSequenceClassification
from model.configs import SWATForSequenceClassificationConfig


DATA_FP = GLOBAL_FP + 'factuality/data/mv_data_bin_guc.json'
CHK_FP = GLOBAL_FP + '/pretraining/run_2/checkpoints/'
RESULTS_FP = GLOBAL_FP + '/factuality/results/gfolds.json'
PRINT_EP = True

PATIENCE = 5
LR = 1e-6
WEIGHT_DECAY = 1e-5
BATCH_SIZE = 8
DEVICE = 0
MODEL_CONFIG = SWATForSequenceClassificationConfig(
    n_classes=2,
    head_config__layer_norm=True,
    head_config__dropout_kwargs__p=0.1,
    pooling='mean'
)
SEED = 25


def cont_check(cndn, check_str):
    in_str = '' if cndn else 'y'

    while in_str not in {'yes', 'y', 'no', 'n'}:
        in_str = input(check_str + ' Continue? (y[es]/n[o]): ').strip().lower()

    if in_str[0] == 'n':
        raise KeyboardInterrupt


def generate_data(data_list):
    for x in data_list:
        yield SWATForSCInput.from_dir_graph(x['g'], targets=x['l'])


def train_eval_loop(model):
    optimizer = torch.optim.AdamW(model.parameters(), lr=LR, weight_decay=WEIGHT_DECAY)
    ep_cnt, patience_cnt, best_acc = 0, 0, 0.0

    while patience_cnt < PATIENCE:
        ep_cnt += 1

        if PRINT_EP:
            print(f'   EPOCH {ep_cnt}:')
            print()

        model.train()
        optimizer.zero_grad()
        permutation = torch.randperm(len(train_data)).tolist()
        batch_perm = map(lambda i_: train_data[i_], permutation)
        train_batches = SWATForSCInput.generate_batches(generate_data(batch_perm), batch_size=BATCH_SIZE)
        epoch_loss, loss_cnt = 0.0, 0

        for _ in tqdm_inner(range(-(-len(train_data) // BATCH_SIZE))):
            batch = next(train_batches)
            model_out = model(batch)
            loss = model_out.loss()

            loss.backward()
            optimizer.step()
            epoch_loss += loss.item()
            loss_cnt += batch.batch_size

            optimizer.zero_grad()
            del batch, model_out, loss

        if PRINT_EP:
            print()
            print(f'   LOSS: {epoch_loss / loss_cnt}')
            print()

        test_batches = SWATForSCInput.generate_batches(generate_data(test_data), batch_size=BATCH_SIZE)
        epoch_acc, acc_cnt = 0.0, 0
        model.eval()

        with torch.no_grad():
            for _ in tqdm_inner(range(-(-len(test_data) // BATCH_SIZE))):
                batch = next(test_batches)
                model_out = model(batch)

                epoch_acc += model_out.accuracy() * batch.batch_size
                acc_cnt += batch.batch_size
                del batch, model_out

        acc = epoch_acc / acc_cnt

        if acc > best_acc:
            best_acc, patience_cnt = acc, 0
        else:
            patience_cnt += 1

        if PRINT_EP:
            print()
            print(f'   ACC: {round(acc * 100, 5)}% (BEST={round(best_acc * 100, 5)}%; PATIENCE={patience_cnt})')
            print()

    del optimizer

    return best_acc, ep_cnt


torch.manual_seed(SEED)
torch.cuda.manual_seed(SEED)

cont_check(RESULTS_FP is None, 'RESULTS_FP is None (results will not be recorded...)!')
cont_check(PRINT_EP, 'PRINT_EP=True (this might get messy...).')
is_printing = RESULTS_FP is None or PRINT_EP
tqdm_outer = (lambda x: x) if is_printing else tqdm
tqdm_inner = tqdm if PRINT_EP else lambda x: x

with open(DATA_FP, 'r') as f:
    data_file = json.load(f)

train_data, test_data, res_data = data_file['train'], data_file['test'], []
checkpoints, chk_fp = [], (os.path.abspath(CHK_FP) + '/').replace('//', '/')

if os.path.isdir(chk_fp):
    ep_mbs = {}

    for chk_fn in os.listdir(chk_fp):
        ep, mb = map(int, chk_fn[2:-4].split('_mb'))

        if ep in ep_mbs.keys():
            ep_mbs[ep].append(mb)
        else:
            ep_mbs.update({ep: [mb]})

    for k_ in sorted(ep_mbs.keys()):
        checkpoints.extend((k_, w) for w in sorted(ep_mbs[k_]))
else:
    assert chk_fp.endswith('.chk/')
    chk_fp, chk_fn = chk_fp[:-1].rsplit('/', 1)
    chk_fp = chk_fp + '/'
    checkpoints.append(tuple(map(int, chk_fn[2:-4].split('_mb'))))

for ep, mb in tqdm_outer(checkpoints):
    if is_printing:
        print()
        print()
        print(f'CHECKPOINT EP={ep}, MB={mb}:')

        if PRINT_EP:
            print()
            print()

    chk_model = SWATForSequenceClassification.from_swat_model(f'{chk_fp}ep{ep}_mb{mb}.chk', MODEL_CONFIG)
    chk_model.to(device=DEVICE)
    chk_acc, chk_ep = train_eval_loop(chk_model)
    res_data.append({'acc': chk_acc, 'ep': chk_ep})

    del chk_model
    gc.collect()
    torch.cuda.empty_cache()

    if RESULTS_FP is None:
        if not PRINT_EP:
            print(f'   BEST ACC: {round(chk_acc * 100, 5)}%, NUM EPOCHS: {chk_ep}\n')
    else:
        with open(RESULTS_FP, 'w') as f:
            json.dump(res_data, f)
