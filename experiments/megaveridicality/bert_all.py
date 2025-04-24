
import gc
import os
import json
import torch
import pickle
from tqdm import tqdm
from collections import OrderedDict
from global_filepath import GLOBAL_FP
from transformers import AutoModelForMaskedLM, AutoTokenizer


DATA_FP = GLOBAL_FP + '/factuality/data/mv_data_bin_guc.json'
CHK_FP = GLOBAL_FP + '/bert/runs/run_lrg_2/checkpoints/'
BERT_MODEL = 'bert-large-uncased'
RESULTS_FP = f'{GLOBAL_FP}/factuality/results/{BERT_MODEL}.json'
PRINT_EP = True

PATIENCE = 5
LR = 1e-6
WEIGHT_DECAY = 1e-5
BATCH_SIZE = 8
DEVICE = 0
ACTIV_FN = torch.nn.GELU()
LAYER_NORM = True
DROPOUT_P = 0.1
SEED = 25


def pt_sd_to_mlm(in_sd, bert_mlm):
    mlm_sd_keys, in_sd_keys = map(set, (bert_mlm.state_dict().keys(), in_sd.keys()))
    assert mlm_sd_keys <= in_sd_keys

    if in_sd_keys == mlm_sd_keys:
        return in_sd

    new_sd = OrderedDict()

    for key_, v_ in in_sd.items():
        if key_ in mlm_sd_keys:
            new_sd.update({key_: v_})

    return new_sd


class BERTClassifier(torch.nn.Module):
    def __init__(self, bert_model, bert_chkpt, activ_fn=torch.nn.GELU(), layer_norm=True, dropout_p=0.1):
        super(BERTClassifier, self).__init__()
        bert_pt = AutoModelForMaskedLM.from_pretrained(bert_model)
        # bert_pt.load_state_dict(bert_chkpt)
        bert_pt.load_state_dict(pt_sd_to_mlm(bert_chkpt, bert_pt))
        self.bert = bert_pt.bert
        d_model = self.bert.embeddings.word_embeddings.weight.size(1)

        self.dense = torch.nn.Linear(d_model, d_model)
        self.activ_fn = self._null_fn if activ_fn is None else activ_fn
        self.layer_norm = torch.nn.LayerNorm(d_model) if layer_norm else self._null_fn
        self.dropout = self._null_fn if dropout_p == 0.0 else torch.nn.Dropout(p=dropout_p)
        self.cls = torch.nn.Linear(d_model, 1)

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
        x = torch.sum(x[..., 1:-1, :], dim=1) / (x.size(1) - 2)
        x = self.dense(x)
        x = self.activ_fn(x)
        x = self.layer_norm(x)
        x = self.dropout(x)

        return self.cls(x)

    def _null_fn(self, x):
        return x


def cont_check(cndn, check_str):
    in_str = '' if cndn else 'y'

    while in_str not in {'yes', 'y', 'no', 'n'}:
        in_str = input(check_str + ' Continue? (y[es]/n[o]): ').strip().lower()

    if in_str[0] == 'n':
        raise KeyboardInterrupt


def generate_data(data_list):
    long_tensor = torch.LongTensor if DEVICE == 'cpu' else torch.cuda.LongTensor

    for k in range(-(-len(data_list) // BATCH_SIZE)):
        batch_data = data_list[k * BATCH_SIZE:(k + 1) * BATCH_SIZE]
        toks = tokenizer([x['s'] for x in batch_data], padding=True)
        toks_ten = torch.tensor(toks['input_ids'], device=DEVICE)[:, 1:-1]
        mask_ten = torch.tensor(toks['attention_mask'], device=DEVICE)[:, 1:-1]
        trgt_ten = torch.tensor([x['l'] for x in batch_data], device=DEVICE).type(long_tensor)

        yield toks_ten, mask_ten, trgt_ten


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
        train_batches = generate_data(list(map(lambda i_: train_data[i_], permutation)))
        epoch_loss, loss_cnt = 0.0, 0

        for _ in tqdm_inner(range(-(-len(train_data) // BATCH_SIZE))):
            batch_ids, batch_mask, batch_trgts = next(train_batches)
            model_out = model(batch_ids, batch_mask)
            loss = loss_fn(model_out.flatten(), batch_trgts.float())

            loss.backward()
            optimizer.step()
            epoch_loss += loss.item()
            loss_cnt += batch_ids.size(0)
            loss_cnt += 1

            optimizer.zero_grad()
            del batch_ids, batch_mask, batch_trgts, model_out, loss

        if PRINT_EP:
            print()
            print(f'   LOSS: {epoch_loss / loss_cnt}')
            print()

        test_batches = generate_data(test_data)
        epoch_acc, acc_cnt = 0.0, 0
        model.eval()

        with torch.no_grad():
            for _ in tqdm(range(-(-len(test_data) // BATCH_SIZE))):
                test_ids, test_mask, test_trgts = next(test_batches)
                model_out = model(test_ids, test_mask)

                epoch_acc += torch.sum((model_out.flatten() >= 0.0) == test_trgts.flatten()).item()
                acc_cnt += test_ids.size(0)

                del test_ids, test_mask, test_trgts, model_out

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
tokenizer = AutoTokenizer.from_pretrained(BERT_MODEL)
loss_fn = torch.nn.BCEWithLogitsLoss()

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

    with open(f'{chk_fp}ep{ep}_mb{mb}.chk', 'rb') as f:
        chk_sd = pickle.load(f)

    chk_model = BERTClassifier(BERT_MODEL, chk_sd, activ_fn=ACTIV_FN, layer_norm=LAYER_NORM, dropout_p=DROPOUT_P)
    chk_model.to(device=DEVICE)
    chk_acc, chk_ep = train_eval_loop(chk_model)
    res_data.append({'acc': chk_acc, 'ep': chk_ep})

    del chk_model, chk_sd
    gc.collect()
    torch.cuda.empty_cache()

    if RESULTS_FP is None:
        if not PRINT_EP:
            print(f'   BEST ACC: {round(chk_acc * 100, 5)}%, NUM EPOCHS: {chk_ep}\n')
    else:
        with open(RESULTS_FP, 'w') as f:
            json.dump(res_data, f)
