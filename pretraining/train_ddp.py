
import gc
import re
import os
import json
import torch
import pickle
import config as cfg
from copy import deepcopy
from model.io import SWATForMLMInput
from model.model import SWATForMaskedLM
import torch.distributed as dist
import torch.multiprocessing as mp
from datetime import datetime
from collections import OrderedDict
from multiprocessing import Manager
from torch.nn.parallel import DistributedDataParallel as DDP


mp.set_start_method('spawn', force=True)


def move_state_dict(sd, device='cpu'):  # copies and moves state dict to device
    new_sd = OrderedDict()

    for k, w in sd.items():
        new_sd[k] = w.to(device)

    return new_sd


def ddp_model(rank, all_files, world_size, chk_cfg, start_iter, end_iter, batch_kwargs, checkpoint_fp, stats_fp):
    n_cpu = os.cpu_count()

    if n_cpu <= world_size:
        cpu_per_proc = n_cpu // world_size
        os.sched_setaffinity(rank, set(range(rank * cpu_per_proc, (rank + 1) * cpu_per_proc)))

    os.environ['MASTER_ADDR'] = 'localhost'
    os.environ['MASTER_PORT'] = '12355'
    dist.init_process_group('gloo', rank=rank, world_size=world_size)
    batch_size, chk_on_err, train_stats_dict, last_chk, chk_cnt = batch_kwargs['batch_size'], False, {}, None, 0

    try:
        if rank == 0:
            train_stats_template = {
                'loss': [],
                'print_rate': cfg.PRINT_RATE,
                'batch_size': cfg.BATCH_SIZE,
                'optimizer_kwargs': {
                    k: (v if isinstance(v, float) else str(v)) for k, v in cfg.OPTIMIZER_KWARGS.items()
                },
                'model_dtype': cfg.MODEL_TO.get('dtype', 'default'),
                'token_dtype': 'default' if cfg.TOKEN_DTYPE is None else str(cfg.TOKEN_DTYPE)
            }

            def checkpoint(mdl, mdl_cfg, stats_dict, mb1, mb2):  # save model and stats dict
                def check_fn_ow(fp, fn0, fn1):  # check filename overwrite
                    fp_fns, sfx = set(os.listdir(fp)), 0

                    if fn0 + fn1 in fp_fns:
                        while f'{fn0}({sfx}){fn1}' in fp_fns:
                            sfx += 1

                        return f'{fp}{fn0}({sfx}){fn1}'

                    return fp + fn0 + fn1

                mdl_fn = check_fn_ow(checkpoint_fp, f'checkpoint{mb2}', '.chk')
                sd_fn = check_fn_ow(cfg.STATS_FILEPATH, f'stats_{mb1}_{mb2}', '.json')

                with open(sd_fn, 'w') as f_sd:
                    json.dump(stats_dict, f_sd)

                with open(mdl_fn, 'wb') as f_save:
                    pickle.dump((move_state_dict(mdl.state_dict()), mdl_cfg), f_save)

            if cfg.PRINT_OUTPUT:
                def record_loss(loss_, cnt_loss, stats_dict, mb_):  # print loss(es) & update stats dict
                    loss_rnd = round(loss_ / cnt_loss, 3)
                    stats_dict['loss'].append(loss_rnd)
                    print()
                    print(f"{mb_}LOSS: {stats_dict['loss'][-1]} ({round(sum(stats_dict['loss']) / len(stats_dict['loss']), 3)})")
                    print()
            else:
                def record_loss(loss_, cnt_loss, stats_dict, *_):
                    loss_rnd = round(loss_ / cnt_loss, 3)
                    stats_dict['loss'].append(loss_rnd)
        else:
            checkpoint, record_loss = (lambda *_: None), (lambda *_: None)
            train_stats_template = None

        if cfg.LOSS_WEIGHT_FILE is None:
            loss_weight = None
        else:
            with open(cfg.LOSS_WEIGHT_FILE, 'r') as f_lw:
                loss_weight = torch.tensor(
                    json.load(f_lw),
                    device=rank,
                    dtype=cfg.MODEL_TO.get('dtype', torch.float32)
                )

        if isinstance(chk_cfg, str):
            with open(chk_cfg, 'rb') as f_chk:
                state_dict, model_cfg = pickle.load(f_chk)
        else:
            state_dict, model_cfg = None, chk_cfg

        model_ = SWATForMaskedLM.from_config(model_cfg)
        model_.to(**{**cfg.MODEL_TO, **{'device': rank}})
        model = DDP(model_)

        if state_dict is not None:
            model.load_state_dict(move_state_dict(state_dict, device=rank))

        optimizer = getattr(torch.optim, cfg.OPTIMIZER)(model.parameters(), **cfg.OPTIMIZER_KWARGS)

        for curr_iter in range(start_iter, end_iter):
            if rank == 0:
                train_stats_dict.update({curr_iter: deepcopy(train_stats_template)})

            with open(all_files[rank][curr_iter], 'r') as f_ci:
                self_file = json.load(f_ci)

            batches = SWATForMLMInput.generate_batches(self_file, **batch_kwargs)
            mb_str = '(META BATCHES ' + (', '.join(str(af[curr_iter]) for af in all_files))
            mb_str += f' [{curr_iter}/{len(all_files[rank])}]) '
            mb_loss, loss_cnt, total_cnt = 0, 0, 0
            chk_cnt += 1
            dist.barrier()

            with model.join():
                for _ in range(-(-len(self_file) // batch_kwargs['batch_size'])):  # training loop
                    try:
                        batch = next(batches)
                    except StopIteration:
                        break

                    loss = model(batch).loss(weight=loss_weight)
                    loss.backward()
                    optimizer.step()
                    mb_loss += loss.item()
                    loss_cnt += 1
                    total_cnt += 1
                    optimizer.zero_grad()

                    if rank == 0 and total_cnt % cfg.PRINT_RATE == 0:
                        record_loss(mb_loss, loss_cnt, train_stats_dict[curr_iter], mb_str)
                        mb_loss, loss_cnt = 0, 0

            if rank == 0:
                if loss_cnt > 0:
                    record_loss(mb_loss, loss_cnt, train_stats_dict[curr_iter], mb_str)

                if chk_cnt == cfg.CHECKPOINT_RATE:
                    checkpoint(model, model_cfg, train_stats_dict, (curr_iter + 1) - chk_cnt, curr_iter)
                    train_stats_dict.clear()
                    last_chk, chk_cnt = None, 0
                elif cfg.CHECKPOINT_ON_ERROR or cfg.CHECKPOINT_ON_KEYBOARD_INTERRUPT:
                    last_chk = move_state_dict(model.state_dict()), model_cfg

            gc.collect()
            torch.cuda.empty_cache()

        if rank == 0:
            if chk_cnt > 0:
                len_files = len(all_files[rank])
                checkpoint(model, model_cfg, train_stats_dict, len_files - chk_cnt, len_files - 1)

            if cfg.PRINT_OUTPUT:
                print('\n\nTRAINING COMPLETE!')

    except KeyboardInterrupt:
        chk_on_err = cfg.CHECKPOINT_ON_KEYBOARD_INTERRUPT
        raise KeyboardInterrupt
    except Exception as e:
        chk_on_err = cfg.CHECKPOINT_ON_ERROR
        raise e
    finally:
        if chk_on_err and rank == 0:
            if last_chk is not None:
                first_mb, last_mb = (lambda z: (min(z), max(z)))(train_stats_dict.keys())
                last_epoch_stats = {last_mb: train_stats_dict.pop(last_mb)}

                # with open(checkpoint_fp + 'checkpoint_last_error.chk', 'wb') as f_err:
                #     pickle.dump(last_chk, f_err)

                with open(f'{checkpoint_fp}checkpoint_last_error{last_mb - 1}.chk', 'wb') as f_err:
                    pickle.dump(last_chk, f_err)

                with open(f'{stats_fp}stats_last_error_{first_mb}_{last_mb - 1}.json', 'w') as f_err:
                    json.dump(train_stats_dict, f_err)

                with open(f'{stats_fp}stats_last_error_{last_mb}.json', 'w') as f_err:
                    json.dump(last_epoch_stats, f_err)
            else:
                print('Attempted to checkpoint on error, but no recoverable state dict found...')

        dist.destroy_process_group()


if __name__ == '__main__':
    assert cfg.DDP
    assert cfg.CHECKPOINT_RATE >= 0

    CHK_FP = (os.path.abspath(cfg.CHECKPOINT_FILEPATH) + '/').replace('//', '/')
    DATA_FP = (os.path.abspath(cfg.TRAIN_DATA_FILEPATH) + '/').replace('//', '/')
    STATS_FP = (os.path.abspath(cfg.STATS_FILEPATH) + '/').replace('//', '/')
    process_folders = sorted(os.listdir(DATA_FP), key=int)
    process_files = [sorted(os.listdir(DATA_FP + p), key=lambda x: int(re.sub(r'\D', '', x))) for p in process_folders]

    assert set(map(int, process_folders)) == set(range(len(process_folders)))
    assert 1 < len(process_folders) <= torch.cuda.device_count()
    assert len(process_folders) <= cfg.BATCH_SIZE and cfg.BATCH_SIZE % len(process_folders) == 0
    assert 'device' not in cfg.MODEL_TO and 'device_map' not in cfg.MODEL_TO

    for i, process_f in enumerate(process_files):
        assert len(process_f) == len(process_files[0]), str(i)
        assert {int(re.sub(r'\D', '', x)) for x in process_f} == set(range(len(process_f)))

    max_chk = max(
        (int(re.sub(r'\D', '', x)) for x in os.listdir(CHK_FP) if re.match(r'checkpoint\d+\.chk', x)),
        default=-1
    )
    all_batch_idxs = {int(re.sub(r'\D', '', x)) for x in process_files[0]}

    if cfg.META_BATCHES == 'all':  # configuring meta-batch range
        mb_max = max(all_batch_idxs) + 1
        META_BATCHES = next(i for i in range(max_chk + 1, mb_max) if i in all_batch_idxs), mb_max
    elif isinstance(cfg.META_BATCHES, tuple):
        assert len(cfg.META_BATCHES) == 2
        META_BATCHES = (cfg.META_BATCHES[0], cfg.META_BATCHES[1] + 1)
    else:
        META_BATCHES = (max_chk + 1, cfg.META_BATCHES + 1)

    assert META_BATCHES[1] > META_BATCHES[0]

    for i in range(len(process_files)):
        process_files[i] = list(map(lambda x: f'{DATA_FP}{i}/{x}', process_files[i]))

    if cfg.LOAD_FROM_CHECKPOINT is False:  # initializing/loading model
        if cfg.PRINT_OUTPUT:
            print(f'INITIALIZING MODEL (NO CHECKPOINT)...\n')

        chk_config = cfg.MODEL_CONFIG
    else:
        if cfg.LOAD_FROM_CHECKPOINT is True:
            chk_config = f'checkpoint{max_chk}.chk'
        elif isinstance(cfg.LOAD_FROM_CHECKPOINT, int):
            chk_config = f'checkpoint{cfg.LOAD_FROM_CHECKPOINT}.chk'
        else:  # isinstance(cfg.LOAD_FROM_CHECKPOINT, str)
            chk_config = cfg.LOAD_FROM_CHECKPOINT
        if chk_config == 'checkpoint-1.chk':
            if cfg.PRINT_OUTPUT:
                print(f'ATTEMPTED TO LOAD CHECKPOINT, BUT NO CHECKPOINT DETECTED (INITIALIZING MODEL)...\n')

            chk_config = cfg.MODEL_CONFIG
        else:
            if cfg.PRINT_OUTPUT:
                print(f'INITIALIZING FROM CHECKPOINT \"{chk_config}\"...\n')

            assert chk_config in os.listdir(CHK_FP)
            chk_config = CHK_FP + chk_config

    batch_kwargs_main = {
        'batch_size': cfg.BATCH_SIZE // len(process_folders),
        'collect': cfg.COLLECT_BATCHES,
        'to_after_batch_kwargs': {
            'token_dtype': cfg.TOKEN_DTYPE,
            'target_dtype': cfg.TOKEN_DTYPE
        },
        'swat_input_kwargs': {
            'spec_tok_perturb_prob': cfg.SPEC_TOK_P_PROB,
            'perturb_prob': cfg.P_PROB,
            'mask_prob': cfg.M_PROB,
            'swap_prob': cfg.S_PROB
        }
    }

    t0 = datetime.now()

    with Manager() as m:
        all_files_main = m.list(process_files)
        mp.spawn(
            ddp_model,
            args=(
                all_files_main,
                len(process_folders),
                chk_config,
                META_BATCHES[0],
                META_BATCHES[1],
                batch_kwargs_main,
                CHK_FP,
                STATS_FP
            ),
            nprocs=len(process_folders),
            join=True
        )

    with open(cfg.FILEPATH + 'ddp_time.json', 'w') as f:
        json.dump({'': str(datetime.now() - t0)}, f)
