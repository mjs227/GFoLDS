
import os
import gc
import json
import torch
import pickle
from tqdm import tqdm
from collections import OrderedDict
from global_filepath import GLOBAL_FP
from transformers import AutoModelForMaskedLM, AutoTokenizer


TRAIN_FP = GLOBAL_FP + '/nli/data/train.json'
DEV_FP = GLOBAL_FP + '/nli/data/dev.json'
TEST_FP = GLOBAL_FP + '/nli/data/test.json'
CHK_FP = GLOBAL_FP + '/nli/runs/bert_large_comparison/checkpoints/'
OPTIM_FP = GLOBAL_FP + '/nli/runs/bert_large_comparison/optimizers/'
STAT_FP = GLOBAL_FP + '/nli/runs/bert_large_comparison/stats/'

BERT_MODEL = 'bert-large-uncased'
RESUME = True  # load from fine-tuned checkpoint
RESET_WGTS = False
CHECKPOINT_MODEL = True
PRINT_RATE = 250
EVAL_RATE = 5000
BATCH_SIZE = 16
N_EPOCHS = 5
DEVICE = 0
SEED = 21

OPTIMIZER_KWARGS = {
    'init_lr': 1e-6,  # if lr_vals is None, then lr = PEAK_LR
    'lr_vals': [2e-6, 3e-6, 1e-7, 1e-8],  # must be same len as LR_STEP
    'lr_step': [0.2, 0.6, 0.8, 1.0],
    'weight_decay': 1e-5
}
ACTIV_FN = torch.nn.GELU()
LAYER_NORM = True
DROPOUT_P = 0.1


def reset_all_weights(m):
    @torch.no_grad()
    def weight_reset(m_):
        reset_parameters = getattr(m_, 'reset_parameters', None)

        if callable(reset_parameters):
            m_.reset_parameters()

    m.apply(fn=weight_reset)


def pt_sd_to_mlm(in_sd, bert_mlm):
    mlm_sd_keys, in_sd_keys = map(set, (bert_mlm.state_dict().keys(), in_sd.keys()))
    assert mlm_sd_keys <= in_sd_keys

    if in_sd_keys == mlm_sd_keys:
        return in_sd

    new_sd = OrderedDict()

    for k_, v_ in in_sd.items():
        if k_ in mlm_sd_keys:
            new_sd.update({k_: v_})

    return new_sd


class BERTClassifier(torch.nn.Module):
    def __init__(self, bert_model, activ_fn=torch.nn.GELU(), layer_norm=True, dropout_p=0.1, reset_wgts=False):
        super(BERTClassifier, self).__init__()
        self.bert = AutoModelForMaskedLM.from_pretrained(bert_model).bert
        d_model = self.bert.embeddings.word_embeddings.weight.size(1)

        if reset_wgts:
            reset_all_weights(self.bert)

        self.dense = torch.nn.Linear(d_model, d_model)
        self.activ_fn = self._null_fn if activ_fn is None else activ_fn
        self.layer_norm = torch.nn.LayerNorm(d_model) if layer_norm else self._null_fn
        self.dropout = self._null_fn if dropout_p == 0.0 else torch.nn.Dropout(p=dropout_p)
        self.cls = torch.nn.Linear(d_model, 3)

        self.device = 'cpu'
        self._save_dict = {
            'bert_model': bert_model,
            'activ_fn': activ_fn,
            'layer_norm': layer_norm,
            'dropout_p': dropout_p
        }

    def to(self, *args, **kwargs):
        super(BERTClassifier, self).to(*args, **kwargs)
        device = kwargs.get('device', next((a for a in args if not isinstance(a, torch.dtype)), None))
        self.device = self.device if device is None else device

    def forward(self, x, attn_mask):
        x = self.bert(x, attention_mask=attn_mask).last_hidden_state
        x = torch.sum(x * attn_mask[..., None], dim=1) / torch.clamp(attn_mask.sum(dim=1)[..., None], min=1e-9)
        x = self.dense(x)
        x = self.activ_fn(x)
        x = self.layer_norm(x)
        x = self.dropout(x)

        return self.cls(x)

    def save(self, fp):
        device = self.device
        self.to(device='cpu')

        with open(fp, 'wb') as f_chk:
            pickle.dump({**self._save_dict, **{'state_dict': self.state_dict()}}, f_chk)

        self.to(device=device)

    def _null_fn(self, x):
        return x

    @classmethod
    def from_pretrained(cls, fp, bert_model=None, **kwargs):
        with open(fp, 'rb') as f_chk:
            bert_chk = pickle.load(f_chk)

        if 'state_dict' in bert_chk.keys():
            assert len(kwargs) == 0
            state_dict_chk = bert_chk.pop('state_dict')
            cls_out = cls(**bert_chk)
            cls_out.load_state_dict(state_dict_chk)
        else:
            assert bert_model is not None
            cls_out = cls(bert_model, **kwargs)
            bert = AutoModelForMaskedLM.from_pretrained(bert_model)
            bert.load_state_dict(pt_sd_to_mlm(bert_chk, bert))
            del_bert = cls_out.bert
            cls_out.bert = bert.bert
            del del_bert

        return cls_out


def generate_data(data_list):
    long_tensor = torch.LongTensor if DEVICE == 'cpu' else torch.cuda.LongTensor

    def template(x):
        s1, s2 = x['s']['s1'].strip(), x['s']['s2'].strip()

        return f'{s1}[SEP]{s2}'

    for k in range(-(-len(data_list) // BATCH_SIZE)):
        batch_data = data_list[k * BATCH_SIZE:(k + 1) * BATCH_SIZE]
        toks = tokenizer(list(map(template, batch_data)), padding=True)
        toks_ten = torch.tensor(toks['input_ids'], device=DEVICE)[:, 1:-1]
        mask_ten = torch.tensor(toks['attention_mask'], device=DEVICE)[:, 1:-1]
        trgt_ten = torch.tensor([label_map[x['lbl']] for x in batch_data], device=DEVICE).type(long_tensor)

        yield toks_ten, mask_ten, trgt_ten


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
    eval_batches = generate_data(eval_file)
    tqdm_ = tqdm if use_tqdm else lambda z: z
    acc_total, acc_cnt = 0, 0
    model_.eval()

    with torch.no_grad():
        for _ in tqdm_(range(-(-len(eval_file) // BATCH_SIZE))):
            try:
                eval_toks, eval_mask, eval_trgts = next(eval_batches)
                eval_out = model_(eval_toks, eval_mask)
                acc_total += torch.sum(torch.argmax(eval_out, dim=-1).flatten() == eval_trgts).item()
                acc_cnt += eval_toks.size(0)

                del eval_toks, eval_mask, eval_out
            except StopIteration:
                break

    model_.train()

    return acc_total * 100 / acc_cnt


assert 'lr' not in OPTIMIZER_KWARGS.keys()
assert not (RESET_WGTS and RESUME)
torch.manual_seed(SEED)
torch.cuda.manual_seed(SEED)

with open(TEST_FP, 'r') as f:
    test_file = json.load(f)

with open(DEV_FP, 'r') as f:
    dev_file = json.load(f)

with open(TRAIN_FP, 'r') as f:
    train_file = json.load(f)

N_TRAIN_STEPS = -(-len(train_file) // BATCH_SIZE)
tokenizer = AutoTokenizer.from_pretrained(BERT_MODEL)

if RESUME:  # load from fine-tuned checkpoint
    max_chk = max(int(x.split('.')[0][2:]) for x in os.listdir(CHK_FP))

    print(f'INITIALIZING FROM FINE-TUNED CHECKPOINT ep{max_chk}.chk ...')

    if max_chk > -1:
        model = BERTClassifier.from_pretrained(f'{CHK_FP}ep{max_chk}.chk')

        with open(f'{OPTIM_FP}ep{max_chk}.opt', 'rb') as f_opt:
            optim_sd = pickle.load(f_opt)
    else:
        model = BERTClassifier.from_pretrained(
            f'{CHK_FP}ep{max_chk}.chk',
            bert_model=BERT_MODEL,
            activ_fn=ACTIV_FN,
            layer_norm=LAYER_NORM,
            dropout_p=DROPOUT_P,
            reset_wgts=RESET_WGTS
        )
        optim_sd = None

    model.to(device=DEVICE)
else:
    assert len(os.listdir(CHK_FP)) == 0
    assert len(os.listdir(OPTIM_FP)) == 0

    print(f'INITIALIZING PRE-TRAINED BERT MODEL {BERT_MODEL} ...')

    model = BERTClassifier(
        BERT_MODEL,
        activ_fn=ACTIV_FN,
        layer_norm=LAYER_NORM,
        dropout_p=DROPOUT_P,
        reset_wgts=RESET_WGTS
    )
    model.to(device=DEVICE)
    optim_sd, max_chk = None, -1

max_chk += 1
optim_type = torch.optim.Adam if OPTIMIZER_KWARGS.get('weight_decay', 0.0) == 0.0 else torch.optim.AdamW
lr_step, lr_vals = OPTIMIZER_KWARGS.pop('lr_step', ()), OPTIMIZER_KWARGS.pop('lr_vals', ())
init_lr = OPTIMIZER_KWARGS.pop('init_lr')
assert len(lr_vals) == len(lr_step)

if len(lr_step) > 0:
    assert all(0 < lr_step[i] < lr_step[i + 1] <= 1 for i in range(len(lr_step) - 1))
    assert all(x >= 0 for x in lr_vals)
    LR_VALS = [init_lr] + lr_vals + ([lr_vals[-1]] if lr_step[-1] < 1.0 else [])
    LR_STEP = [0.0] + lr_step + ([1.0] if lr_step[-1] < 1.0 else [])
    LR_FRAC = [(LR_VALS[i + 1] - LR_VALS[i]) / (LR_STEP[i + 1] - LR_STEP[i]) for i in range(len(LR_VALS) - 1)]
    chk_ep_mb, max_ep_mb = max_chk * N_TRAIN_STEPS, N_EPOCHS * N_TRAIN_STEPS

    def lr_schedule_fn(ep_mb_):
        ep_mb_prop = (ep_mb_ + chk_ep_mb) / max_ep_mb  # chk_ep_mb > 0 => loaded from checkpoint
        val_i = next(i for i in range(len(LR_STEP) - 1) if LR_STEP[i + 1] >= ep_mb_prop)

        return LR_VALS[val_i] + ((ep_mb_prop - LR_STEP[val_i]) * LR_FRAC[val_i])

    optimizer = optim_type(model.parameters(), lr=1.0, **OPTIMIZER_KWARGS)
    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=lr_schedule_fn)

    def optimizer_step():
        optimizer.step()
        scheduler.step()
else:
    optimizer, scheduler = optim_type(model.parameters(), lr=init_lr, **OPTIMIZER_KWARGS), None

    def optimizer_step():
        optimizer.step()

if optim_sd is not None:
    optimizer.load_state_dict(optim_sd)

optimizer.zero_grad()
label_map = {y: x for x, y in enumerate(('e', 'n', 'c'))}
loss_fn = torch.nn.CrossEntropyLoss()

for ep in range(max_chk, N_EPOCHS):
    print(f'EPOCH {ep}:')

    model.train()
    optimizer.zero_grad()

    permutation = torch.randperm(len(train_file)).tolist()
    batch_perm = list(map(lambda i_: train_file[i_], permutation))

    train_batches = generate_data(batch_perm)
    print_cnt, eval_cnt, running_loss = 0, 0, 0
    stats_dict = {'loss': [], 'val': [], 'val_rate': EVAL_RATE}

    print('TRAINING LOOP:')
    print()
    print()

    for _ in tqdm(range(N_TRAIN_STEPS)):
        try:
            batch_toks, batch_mask, batch_trgt = next(train_batches)
            model_out = model(batch_toks, batch_mask)
            loss = loss_fn(model_out, batch_trgt)

            loss.backward()
            optimizer_step()
            running_loss += loss.item()
            print_cnt += 1
            eval_cnt += 1

            optimizer.zero_grad()
            del batch_toks, batch_mask, batch_trgt, model_out, loss

            if print_cnt == PRINT_RATE:
                record_loss(stats_dict, print_cnt, running_loss, ep)
                print_cnt, running_loss = 0, 0.0
            if eval_cnt == EVAL_RATE:
                print()
                print('MID-EPOCH VALIDATION...')

                stats_dict['val'].append(evaluate(model, dev_file, use_tqdm=False))
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

        with open(f'{OPTIM_FP}ep{ep}.opt', 'wb') as f:
            pickle.dump(optimizer.state_dict(), f)

        with open(f'{STAT_FP}stats_ep{ep}.json', 'w') as f:
            json.dump(stats_dict, f)

    gc.collect()
    torch.cuda.empty_cache()

print('TRAINING COMPLETE')
print()

test_acc = evaluate(model, test_file)

print()
print(f'TEST ACC: {round(test_acc, 5)}%')