
import os
import gc
import json
import torch
import pickle
from tqdm import tqdm
from model.io import SWATForSCInput
from global_filepath import GLOBAL_FP
from model.model import SWATForSequenceClassification
from model.configs import SWATForSequenceClassificationConfig


TRAIN_FP = GLOBAL_FP + '/nli/data/train.json'
DEV_FP = GLOBAL_FP + '/nli/data/dev.json'
TEST_FP = GLOBAL_FP + '/nli/data/test.json'
CHK_FP = GLOBAL_FP + '/nli/runs/gfolds/checkpoints/'
OPTIM_FP = GLOBAL_FP + '/nli/runs/gfolds/optimizers/'
STAT_FP = GLOBAL_FP + '/nli/runs/gfolds/stats/'

# INIT_FROM_PT_CHECKPOINT = GLOBAL_FP + '/pretraining/run_2/checkpoints/ep3_mb199.chk'
INIT_FROM_PT_CHECKPOINT = None  # None => load from fine-tuning checkpoint
CHECKPOINT_MODEL = True
PRINT_RATE = 250
EVAL_RATE = 5000
BATCH_SIZE = 16
N_EPOCHS = 5
DEVICE = 0
SEED = 28

OPTIMIZER_LM_KWARGS = {
    'init_lr': 1e-5,  # if lr_vals is None, then lr = PEAK_LR
    'lr_vals': [2e-5, 3e-5, 1e-6, 1e-7],  # must be same len as LR_STEP
    'lr_step': [0.2, 0.6, 0.8, 1.0],
    'weight_decay': 1e-5
}
OPTIMIZER_HEAD_KWARGS = None
MODEL_CONFIG = SWATForSequenceClassificationConfig(
    n_classes=3,
    head_config__layer_norm=True,
    head_config__dropout_kwargs__p=0.1,
    pooling='mean',
    swat_config__d_model=1024,
    swat_config__n_encoder_layers=10,
    swat_config__embedding_config__d_swa=1024,
    swat_config__embedding_config__n_swa_layers=2
)


class DualOptimizer:
    def __init__(self, model_, optim_lm_kwargs, optim_head_kwargs=None, _ep=0):
        _init_optim_kwargs(optim_lm_kwargs)
        _init_optim_kwargs(optim_head_kwargs)

        if optim_head_kwargs is None or (set(optim_head_kwargs.items()) == set(optim_lm_kwargs.items())):  # single
            self.optim_lm, self.sched_lm, rop_lm = _init_optim(model_.parameters(), optim_lm_kwargs, _ep)
            self.optim_head, self.sched_head, rop_head = None, None, False

            def zg_fn():
                self.optim_lm.zero_grad()

            if self.sched_lm is None or rop_lm:
                def step_fn():
                    self.optim_lm.step()

                if rop_lm:
                    def step_rop(val):
                        self.sched_lm.step(val)
                else:
                    step_rop = lambda _: None
            else:
                step_rop = lambda _: None

                def step_fn():
                    self.optim_lm.step()
                    self.sched_lm.step()

            self._step_fn, self._zero_grad_fn, self._step_rop_fn = step_fn, zg_fn, step_rop
        else:
            self.optim_lm, self.sched_lm, rop_lm = _init_optim(model_.swat.parameters(), optim_lm_kwargs, _ep)
            self.optim_head, self.sched_head, rop_head = _init_optim(model_.head.parameters(), optim_head_kwargs, _ep)

            def zg_fn():
                self.optim_lm.zero_grad()
                self.optim_head.zero_grad()

            def step_fn0():
                self.optim_lm.step()
                self.optim_head.step()

            if self.sched_lm is None or rop_lm:
                step_fn1 = step_fn0

                if rop_lm:
                    def step_rop_fn0(val):
                        self.sched_lm.step(val)
                else:
                    step_rop_fn0 = lambda _: None
            else:
                step_rop_fn0 = lambda _: None

                def step_fn1():
                    step_fn0()
                    self.sched_lm.step()

            if self.sched_head is None or rop_head:
                step_fn2 = step_fn1

                if rop_head:
                    def step_rop_fn1(val):
                        step_rop_fn0(val)
                        self.sched_lm.step(val)
                else:
                    step_rop_fn1 = step_rop_fn0
            else:
                step_rop_fn1 = step_rop_fn0

                def step_fn2():
                    step_fn1()
                    self.sched_head.step()

            self._step_fn, self._zero_grad_fn, self._step_rop_fn = step_fn2, zg_fn, step_rop_fn1

    def step_rop(self, val):
        self._step_rop_fn(val)

    def step(self):
        self._step_fn()

    def zero_grad(self):
        self._zero_grad_fn()

    def save(self, filepath):
        if self.optim_head is None:
            state_dict = self.optim_lm.state_dict()
        else:
            state_dict = (self.optim_lm.state_dict(), self.optim_head.state_dict())

        with open(os.path.abspath(filepath), 'wb') as f_save:
            pickle.dump(state_dict, f_save)

    @classmethod
    def load(cls, filepath, model_, optim_lm_kwargs, **kwargs):
        with open(os.path.abspath(filepath), 'rb') as f_load:
            state_dict = pickle.load(f_load)

        cls_out = cls(model_, optim_lm_kwargs, **kwargs)

        if isinstance(state_dict, tuple):
            cls_out.optim_lm.load_state_dict(state_dict[0])
            cls_out.optim_head.load_state_dict(state_dict[1])
        else:
            assert cls_out.optim_head is None
            cls_out.optim_lm.load_state_dict(state_dict)

        return cls_out


def _init_optim(params, kwargs, ep_):
    assert 'lr' not in kwargs.keys()
    opt_type = torch.optim.AdamW if 'weight_decay' in kwargs.keys() else torch.optim.Adam
    init_lr, lr_vals, lr_step = kwargs.pop('init_lr'), kwargs.pop('lr_vals', None), kwargs.pop('lr_step', None)
    lr_vals, lr_step = map(lambda w: None if w is None or len(w) == 0 else w, (lr_vals, lr_step))

    if 'reduce_on_plateau' in kwargs.keys():
        assert lr_vals is None and lr_step is None
        rop_kwargs = kwargs.pop('reduce_on_plateau')
        opt = opt_type(params, lr=init_lr, **kwargs)

        return opt, torch.optim.lr_scheduler.ReduceLROnPlateau(opt, **rop_kwargs), True
    if lr_vals is None:  # no lr scheduling
        assert lr_step is None

        return opt_type(params, lr=init_lr, **kwargs), None, False

    assert all(0 < lr_step[j] < lr_step[j + 1] <= 1 for j in range(len(lr_step) - 1))
    assert len(lr_vals) == len(lr_step)
    assert all(x >= 0 for x in lr_vals)

    lr_vals = (init_lr,) + lr_vals + ((lr_vals[-1],) if lr_step[-1] < 1.0 else ())
    lr_step = (0.0,) + lr_step + ((1.0,) if lr_step[-1] < 1.0 else ())
    lr_frac = [(lr_vals[j + 1] - lr_vals[j]) / (lr_step[j + 1] - lr_step[j]) for j in range(len(lr_vals) - 1)]
    chk_ep_mb, max_ep_mb = ep_ * N_TRAIN_STEPS, N_EPOCHS * N_TRAIN_STEPS

    def lr_schedule_fn(ep_mb_):
        ep_mb_prop = (ep_mb_ + chk_ep_mb) / max_ep_mb  # chk_ep_mb > 0 => loaded from checkpoint
        val_i = next(j for j in range(len(lr_step) - 1) if lr_step[j + 1] >= ep_mb_prop)

        return lr_vals[val_i] + ((ep_mb_prop - lr_step[val_i]) * lr_frac[val_i])

    optim_out = opt_type(params, lr=1.0, **kwargs)
    scheduler = torch.optim.lr_scheduler.LambdaLR(optim_out, lr_lambda=lr_schedule_fn)

    return optim_out, scheduler, False


def _init_optim_kwargs(kwargs):
    if kwargs is not None:
        kw_keys = list(kwargs.keys())

        for k in kw_keys:
            if isinstance(kwargs[k], list):
                kwargs.update({k: tuple(kwargs[k])})


def generate_data(data_list):
    for x in data_list:
        yield SWATForSCInput.from_dir_graph(x['g'], targets=label_map[x['lbl']])


def record_loss(stats_dict_, cnt, loss_, ep_):
    loss_val = loss_ / cnt
    stats_dict_['loss'].append((loss_val, cnt))
    mean_loss_num, mean_loss_den = 0, 0

    for a, b in stats_dict['loss']:
        mean_loss_num += a * b
        mean_loss_den += b

    print()
    print(f'LOSS (EPOCH {ep_}): {loss_val} ({round(mean_loss_num / mean_loss_den, 3)})')
    print()


def evaluate(model_, eval_file, use_tqdm=True):
    eval_batches = SWATForSCInput.generate_batches(generate_data(eval_file), batch_size=BATCH_SIZE)
    tqdm_ = tqdm if use_tqdm else lambda z: z
    acc_total, acc_cnt = 0, 0
    model_.eval()

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

    model_.train()

    return acc_total * 100 / acc_cnt


torch.manual_seed(SEED)
torch.cuda.manual_seed(SEED)

with open(TEST_FP, 'r') as f:
    test_file = json.load(f)

with open(DEV_FP, 'r') as f:
    dev_file = json.load(f)

with open(TRAIN_FP, 'r') as f:
    train_file = json.load(f)

if MODEL_CONFIG.pooling == 'first':
    for data_file in (test_file, dev_file, train_file):
        for i in range(len(data_file)):
            cls_id = next(k for k in data_file[i]['g']['n'].keys() if k[0] == '-')
            data_file[i]['g']['n'][cls_id][0] = 2  # "[UNK] as [CLS]"

N_TRAIN_STEPS = -(-len(train_file) // BATCH_SIZE)

if INIT_FROM_PT_CHECKPOINT is None:  # load from fine-tuned checkpoint
    max_chk = max(int(x[2:-4]) for x in os.listdir(CHK_FP))

    print(f'INITIALIZING FROM FINE-TUNED CHECKPOINT ep{max_chk}.chk ...')

    model = SWATForSequenceClassification.from_pretrained(f'{CHK_FP}ep{max_chk}.chk')
    model.to(device=DEVICE)
    optimizer = DualOptimizer.load(
        f'{OPTIM_FP}ep{max_chk}.opt',
        model,
        OPTIMIZER_LM_KWARGS,
        optim_head_kwargs=OPTIMIZER_HEAD_KWARGS,
        _ep=(max_chk + 1)
    )
else:
    assert len(os.listdir(CHK_FP)) == 0
    assert len(os.listdir(OPTIM_FP)) == 0

    if INIT_FROM_PT_CHECKPOINT == 'reset':
        print('INITIALIZING MODEL FROM SCRATCH...')
        model, max_chk = SWATForSequenceClassification.from_config(MODEL_CONFIG), -1
    else:
        print(f'INITIALIZING FROM PRE-TRAINED CHECKPOINT {INIT_FROM_PT_CHECKPOINT.split("/")[-1]} ...')
        model, max_chk = SWATForSequenceClassification.from_swat_model(INIT_FROM_PT_CHECKPOINT, MODEL_CONFIG), -1

    model.to(device=DEVICE)
    optimizer = DualOptimizer(model, OPTIMIZER_LM_KWARGS, optim_head_kwargs=OPTIMIZER_HEAD_KWARGS)

optimizer.zero_grad()
label_map = {y: x for x, y in enumerate(('e', 'n', 'c'))}

for ep in range(max_chk + 1, N_EPOCHS):
    print(f'EPOCH {ep}:')

    model.train()
    optimizer.zero_grad()

    permutation = torch.randperm(len(train_file)).tolist()
    batch_perm = map(lambda i_: train_file[i_], permutation)

    train_batches = SWATForSCInput.generate_batches(generate_data(batch_perm), batch_size=BATCH_SIZE)
    print_cnt, eval_cnt, running_loss = 0, 0, 0
    stats_dict = {'loss': [], 'val': [], 'val_rate': EVAL_RATE}

    print('TRAINING LOOP:')
    print()
    print()

    for _ in tqdm(range(N_TRAIN_STEPS)):
        try:
            batch = next(train_batches)
            model_out = model(batch)
            loss = model_out.loss()

            loss.backward()
            optimizer.step()
            running_loss += loss.item()
            print_cnt += 1
            eval_cnt += 1

            optimizer.zero_grad()
            del batch, model_out, loss

            if print_cnt == PRINT_RATE:
                record_loss(stats_dict, print_cnt, running_loss, ep)
                print_cnt, running_loss = 0, 0.0
            if eval_cnt == EVAL_RATE:
                print()
                print('MID-EPOCH VALIDATION...')

                stats_dict['val'].append(evaluate(model, dev_file, use_tqdm=False))
                optimizer.step_rop(stats_dict['val'][-1])
                eval_cnt = 0

                print(f'VALIDATION ACC: {round(stats_dict["val"][-1], 5)}%')
                print()

        except StopIteration:
            break

    if print_cnt > 0:
        record_loss(stats_dict, print_cnt, running_loss, ep)

    print()
    print()
    print('VALIDATION STEP:')
    print()

    stats_dict['val'].append(evaluate(model, dev_file))

    print()
    print(f'VALIDATION ACC: {round(stats_dict["val"][-1], 5)}%')
    print()
    print()

    if CHECKPOINT_MODEL:
        model.save(f'{CHK_FP}ep{ep}.chk')
        optimizer.save(f'{OPTIM_FP}ep{ep}.opt')

        with open(f'{STAT_FP}stats_ep{ep}.json', 'w') as f:
            json.dump(stats_dict, f)

    gc.collect()
    torch.cuda.empty_cache()

print('TRAINING COMPLETE')
print()

test_acc = evaluate(model, test_file)

print()
print(f'TEST ACC: {round(test_acc, 5)}%')
