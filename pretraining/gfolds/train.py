
import gc
import os
import json
import torch
import random
import pickle
import config as cfg
from re import match
from copy import deepcopy
from model.io import SWATForMLMInput
from model.model import SWATForMaskedLM


def check_fn_ow(fp, fn0, fn1):  # check filename overwrite
    fp_fns = set(os.listdir(fp))

    if fn0 + fn1 in fp_fns:
        sfx = 0

        while f'{fn0}({sfx}){fn1}' in fp_fns:
            sfx += 1

        return f'{fp}{fn0}({sfx}){fn1}'

    return fp + fn0 + fn1


def save_fn_ow(fp, fn0, fn1, save_obj, as_pkl=True):  # check filename overwrite + save
    with open(check_fn_ow(fp, fn0, fn1), 'wb' if as_pkl else 'w') as f_save:
        (pickle if as_pkl else json).dump(save_obj, f_save)


def rm_optim_backup_sd():
    optim_sd_ = next((x for x in os.listdir(OPTIM_FP) if x.startswith('last_error_')), None)

    if optim_sd_ is not None:
        os.remove(OPTIM_FP + optim_sd_)


def randomize_batches(n_batch):
    out_idxs = list(range(n_batch))
    random.shuffle(out_idxs)

    with open(BATCH_DATA_FP + 'batch_indices.json', 'w') as f_batch:
        json.dump(out_idxs, f_batch)

    return out_idxs


def record_loss_print(loss_, cnt_loss, stats_dict, mb_):  # print loss(es) & update stats dict
    record_loss_no_print(loss_, cnt_loss, stats_dict)
    mean_loss_num, mean_loss_den = 0, 0

    for a, b in stats_dict['loss']:
        mean_loss_num += a * b
        mean_loss_den += b

    print()
    print(f"{mb_}LOSS: {stats_dict['loss'][-1][0]} ({round(mean_loss_num / mean_loss_den, 3)})")
    print()


def record_loss_no_print(loss_, cnt_loss, stats_dict, *_):
    loss_rnd = round(loss_ / cnt_loss, 3)
    stats_dict['loss'].append((loss_rnd, cnt_loss))


def checkpoint(model_, optimizer_, stats_dict, ep_i, mb1, mb2):  # save model and stats dict
    save_fn_ow(STATS_FP, f'stats_ep{ep_i}_mb{mb1}-{mb2}', '.json', stats_dict, as_pkl=False)
    save_fn_ow(OPTIM_FP, f'ep{ep_i}_mb{mb2}', '.opt', optimizer_.state_dict())
    model_.save(check_fn_ow(CHK_FP, f'ep{ep_i}_mb{mb2}', '.chk'))

    rm_optim_backup_sd()
    stats_dict.clear()


CHK_FP = (os.path.abspath(cfg.CHECKPOINT_FILEPATH) + '/').replace('//', '/')
OPTIM_FP = (os.path.abspath(cfg.OPTIMIZER_FILEPATH) + '/').replace('//', '/')
BASE_DATA_FP = (os.path.abspath(cfg.BASE_DATA_FILEPATH) + '/').replace('//', '/')
BATCH_DATA_FP = (os.path.abspath(cfg.BATCH_FILEPATH) + '/').replace('//', '/')
STATS_FP = (os.path.abspath(cfg.STATS_FILEPATH) + '/').replace('//', '/')
LR_SCHEDULE = len(cfg.LR_STEP) > 0

assert all(bool(match(r'ep\d+_mb\d+\.chk$', x)) for x in os.listdir(CHK_FP))
assert all(bool(match(r'ep\d+_mb\d+\.opt$', x)) for x in os.listdir(OPTIM_FP))
assert os.path.isdir(BASE_DATA_FP) or not cfg.RANDOMIZE_BATCHES, \
    f'RANDOMIZE_BATCHES={cfg.RANDOMIZE_BATCHES}; BASE_DATA_FILEPATH={BASE_DATA_FP}'
assert os.path.isdir(BATCH_DATA_FP), f'BATCH_FILEPATH={BATCH_DATA_FP}'
assert os.path.isdir(STATS_FP), f'STATS_FILEPATH={STATS_FP}'
assert 'lr' not in cfg.OPTIMIZER_KWARGS.keys()
assert cfg.CHECKPOINT_RATE >= 0
assert all(0 < cfg.LR_STEP[i] < cfg.LR_STEP[i + 1] <= 1 for i in range(len(cfg.LR_STEP) - 1))
assert len(cfg.LR_VALS) == len(cfg.LR_STEP)
assert all(x >= 0 for x in cfg.LR_VALS)
assert cfg.N_EPOCHS > 0

if cfg.PRINT_OUTPUT:
    if cfg.TQDM:
        from tqdm import tqdm as tqdm_fn
    else:
        tqdm_fn = lambda z: z

    record_loss, print_fn = record_loss_print, lambda *args: print(*args)
else:
    record_loss, tqdm_fn, print_fn = record_loss_no_print, (lambda z: z), (lambda *_: None)

all_chk = {tuple(map(int, x[2:-4].split('_mb'))) for x in os.listdir(CHK_FP)}
chk_ep = max((x[0] for x in all_chk), default=-1)
chk_mb = max((x[1] for x in all_chk if x[0] == chk_ep), default=-1)
assert {tuple(map(int, x[2:-4].split('_mb'))) for x in os.listdir(OPTIM_FP)} == all_chk

if chk_ep == -1:
    model, chk_ep, chk_mb, optim_sd = SWATForMaskedLM.from_config(cfg.MODEL_CONFIG), 0, 0, None
    print_fn(f'INITIALIZING MODEL (NO CHECKPOINT)...\n')
else:
    load_from_chkpt = f'ep{chk_ep}_mb{chk_mb}'
    print_fn(f'INITIALIZING FROM CHECKPOINT \"{load_from_chkpt}\"...\n')

    if chk_mb == len(os.listdir(BASE_DATA_FP)) - 1:  # last item in file
        chk_ep, chk_mb = chk_ep + 1, 0

        if chk_ep == cfg.N_EPOCHS:
            raise ValueError(f'SPECIFIED NUMBER OF EPOCHS ({cfg.N_EPOCHS}) ALREADY REACHED!')
    else:
        chk_mb += 1

    model = SWATForMaskedLM.from_pretrained(CHK_FP + load_from_chkpt + '.chk')

    with open(OPTIM_FP + load_from_chkpt + '.opt', 'rb') as f:
        optim_sd = pickle.load(f)

model.to(**cfg.MODEL_TO)
model.train()

if cfg.RANDOMIZE_BATCHES:
    n_batches = len(os.listdir(BASE_DATA_FP))
    assert 0 < n_batches, 'len(os.listdir(cfg.BASE_DATA_FILEPATH))'
    assert set(map(int, os.listdir(BASE_DATA_FP))) == set(range(n_batches))

    if chk_mb == 0:  # new epoch
        batch_indices = randomize_batches(n_batches)
    else:
        with open(BATCH_DATA_FP + 'batch_indices.json', 'r') as f:
            batch_indices = json.load(f)
else:
    assert set(range(chk_ep, cfg.N_EPOCHS)) <= set(map(int, os.listdir(BATCH_DATA_FP)))
    n_batches, batch_indices = len(os.listdir(f'{BATCH_DATA_FP}{chk_ep}')), None

    for batch_fp in range(chk_ep, cfg.N_EPOCHS):
        assert set(map(int, os.listdir(f'{BATCH_DATA_FP}{batch_fp}'))) == set(range(n_batches)), str(batch_fp)

if LR_SCHEDULE:
    LR_VALS = [cfg.INIT_LR] + cfg.LR_VALS + ([cfg.LR_VALS[-1]] if cfg.LR_STEP[-1] < 1.0 else [])
    LR_STEP = [0.0] + cfg.LR_STEP + ([1.0] if cfg.LR_STEP[-1] < 1.0 else [])
    LR_FRAC = [(LR_VALS[i + 1] - LR_VALS[i]) / (LR_STEP[i + 1] - LR_STEP[i]) for i in range(len(LR_VALS) - 1)]
    chk_ep_mb, max_ep_mb = (chk_ep * n_batches) + chk_mb, cfg.N_EPOCHS * n_batches

    def lr_schedule_fn(ep_mb_):
        ep_mb_prop = (ep_mb_ + chk_ep_mb) / max_ep_mb  # chk_ep_mb > 0 => loaded from checkpoint
        val_i = next(i for i in range(len(LR_STEP) - 1) if LR_STEP[i + 1] >= ep_mb_prop)

        return LR_VALS[val_i] + ((ep_mb_prop - LR_STEP[val_i]) * LR_FRAC[val_i])

    optimizer = cfg.OPTIMIZER(model.parameters(), lr=1.0, **cfg.OPTIMIZER_KWARGS)
    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=lr_schedule_fn)
else:
    optimizer, scheduler = cfg.OPTIMIZER(model.parameters(), lr=cfg.INIT_LR, **cfg.OPTIMIZER_KWARGS), None

if optim_sd is not None:  # loaded from checkpoint
    optimizer.load_state_dict(optim_sd)

optimizer.zero_grad()
train_stats_template = {
    'loss': [],
    'print_rate': cfg.PRINT_RATE,
    'batch_size': cfg.BATCH_SIZE,
    'model_dtype': str(cfg.MODEL_TO.get('dtype', 'default')),
    'token_dtype': 'default' if cfg.TOKEN_DTYPE is None else str(cfg.TOKEN_DTYPE),
    'optimizer_kwargs': {
        k: (v if issubclass(type(v), float) or issubclass(type(v), int) else str(v))
        for k, v in cfg.OPTIMIZER_KWARGS.items()
    }
}
chk_on_err, last_chk, last_chk_ep, last_chk_mb, chk_cnt, train_stats_dict = False, None, 0, 0, 0, {}

if cfg.LOSS_WEIGHT_FILE is None:
    loss_weight = None
else:
    with open(cfg.LOSS_WEIGHT_FILE, 'r') as f:
        loss_weight = torch.tensor(
            json.load(f),
            device=model.decoder.device,
            dtype=cfg.MODEL_TO.get('dtype', torch.float32)
        )

try:
    for ep in range(chk_ep, cfg.N_EPOCHS):
        for meta_batch in range(chk_mb, n_batches):  # meta-training loop
            print_fn()
            print_fn(f'EPOCH={ep}/{cfg.N_EPOCHS - 1}; META_BATCH={meta_batch}/{n_batches - 1}:')
            print_fn()

            train_stats_dict.update({meta_batch: deepcopy(train_stats_template)})
            train_stats_dict[meta_batch].update({'lr': scheduler.get_last_lr()[0] if LR_SCHEDULE else cfg.INIT_LR})

            if cfg.RANDOMIZE_BATCHES:
                with open(f'{BASE_DATA_FP}{batch_indices[meta_batch]}', 'r') as f:
                    meta_batch_file = json.load(f)

                random.shuffle(meta_batch_file)
            else:
                with open(f'{BATCH_DATA_FP}{ep}/{meta_batch}', 'r') as f:
                    meta_batch_file = json.load(f)

            batches = SWATForMLMInput.generate_batches(
                meta_batch_file,
                batch_size=cfg.BATCH_SIZE,
                collect=cfg.COLLECT_BATCHES,
                to_after_batch_kwargs={
                    'token_dtype': cfg.TOKEN_DTYPE,
                    'target_dtype': cfg.TOKEN_DTYPE
                },
                swat_input_kwargs={
                    'spec_tok_perturb_prob': cfg.SPEC_TOK_P_PROB,
                    'perturb_prob': cfg.P_PROB,
                    'mask_prob': cfg.M_PROB,
                    'swap_prob': cfg.S_PROB
                }
            )
            mb_str, mb_loss, loss_cnt = f'(EP={ep}/{cfg.N_EPOCHS - 1}; MB={meta_batch}/{n_batches - 1}) ', 0, 0
            chk_cnt += 1

            for _ in tqdm_fn(range(-(-len(meta_batch_file) // cfg.BATCH_SIZE))):  # main training loop
                try:
                    batch = next(batches)
                    model_out = model(batch)
                    loss = model_out.loss(weight=loss_weight)

                    loss.backward()
                    optimizer.step()
                    mb_loss += loss.item()
                    loss_cnt += 1

                    optimizer.zero_grad()
                    del batch, model_out, loss

                    if loss_cnt == cfg.PRINT_RATE:
                        record_loss(mb_loss, loss_cnt, train_stats_dict[meta_batch], mb_str)
                        mb_loss, loss_cnt = 0, 0
                except StopIteration:
                    break

            # post-meta-batch cleanup:
            if LR_SCHEDULE:
                scheduler.step()
            if loss_cnt > 0:
                record_loss(mb_loss, loss_cnt, train_stats_dict[meta_batch], mb_str)
            if chk_cnt == cfg.CHECKPOINT_RATE:
                checkpoint(model, optimizer, train_stats_dict, ep, (meta_batch + 1) - chk_cnt, meta_batch)
                last_chk_ = last_chk
                last_chk, chk_cnt = None, 0
                del last_chk_
            elif cfg.CHECKPOINT_ON_ERROR or cfg.CHECKPOINT_ON_KEYBOARD_INTERRUPT:
                rm_optim_backup_sd()
                save_fn_ow(OPTIM_FP, f'last_error_ep{ep}_mb{meta_batch}', '.opt', optimizer.state_dict())

                last_chk_ep, last_chk_mb = ep, meta_batch
                last_chk_ = last_chk
                last_chk = model.save()
                del last_chk_

            gc.collect()
            torch.cuda.empty_cache()

        # post-epoch cleanup:
        if chk_cnt > 0:
            checkpoint(model, optimizer, train_stats_dict, ep, n_batches - chk_cnt, n_batches - 1)

        last_chk_ = last_chk
        last_chk, chk_cnt, chk_mb = None, 0, 0  # chk_mb = 0 in case of load from checkpoint
        del last_chk_
        gc.collect()
        torch.cuda.empty_cache()

        if cfg.RANDOMIZE_BATCHES and ep < cfg.N_EPOCHS - 1:
            batch_indices = randomize_batches(n_batches)
except KeyboardInterrupt:
    chk_on_err = cfg.CHECKPOINT_ON_KEYBOARD_INTERRUPT
    raise KeyboardInterrupt
except Exception as e:
    chk_on_err = cfg.CHECKPOINT_ON_ERROR
    raise e
finally:
    if chk_on_err:
        if last_chk is None:
            raise RuntimeError('Attempted to checkpoint on error, but no recoverable state dict found...')

        last_epoch_stats = {last_chk_mb + 1: train_stats_dict.pop(last_chk_mb + 1, {})}
        first_mb = min(train_stats_dict.keys())

        save_fn_ow(CHK_FP, f'last_error_ep{last_chk_ep}_mb{last_chk_mb}', '.chk', last_chk)
        save_fn_ow(STATS_FP, f'stats_last_error_ep{last_chk_ep}_mb{first_mb}_{last_chk_mb}', '.json', train_stats_dict)
        save_fn_ow(STATS_FP, f'stats_last_error_ep{last_chk_ep}_mb{last_chk_mb + 1}', '.json', last_epoch_stats)
    else:
        rm_optim_backup_sd()

print_fn('\n\nTRAINING COMPLETE!')
