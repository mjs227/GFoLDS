
import json
import torch
import pickle
from tqdm import tqdm
from collections import OrderedDict
from global_filepath import GLOBAL_FP
from transformers import AutoModelForMaskedLM, AutoTokenizer


DATA_FP = GLOBAL_FP + '/factuality/data/mv_data_bin_guc.json'
PT_FP = GLOBAL_FP + '/bert/runs/run_4/ep3_mb199.chk'
BERT_MODEL = 'bert-base-uncased'
RESET_WGTS = False
PATIENCE = 5
LR = 1e-6
WEIGHT_DECAY = 1e-5
BATCH_SIZE = 8
DEVICE = 0
ACTIV_FN = torch.nn.GELU()
LAYER_NORM = True
DROPOUT_P = 0.1
SEED = 25


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
        x = torch.sum(x * attn_mask[..., None], dim=1) / torch.clamp(attn_mask.sum(dim=1)[..., None], min=1e-9)
        x = self.dense(x)
        x = self.activ_fn(x)
        x = self.layer_norm(x)
        x = self.dropout(x)

        return self.cls(x)

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

    for k in range(-(-len(data_list) // BATCH_SIZE)):
        batch_data = data_list[k * BATCH_SIZE:(k + 1) * BATCH_SIZE]
        toks = tokenizer([x['s'] for x in batch_data], padding=True)
        toks_ten = torch.tensor(toks['input_ids'], device=DEVICE)[:, 1:-1]
        mask_ten = torch.tensor(toks['attention_mask'], device=DEVICE)[:, 1:-1]
        trgt_ten = torch.tensor([x['l'] for x in batch_data], device=DEVICE).type(long_tensor)

        yield toks_ten, mask_ten, trgt_ten


torch.manual_seed(SEED)
torch.cuda.manual_seed(SEED)

tokenizer = AutoTokenizer.from_pretrained(BERT_MODEL)

if PT_FP is None:
    model = BERTClassifier(
        BERT_MODEL,
        activ_fn=ACTIV_FN,
        reset_wgts=RESET_WGTS,
        layer_norm=LAYER_NORM,
        dropout_p=DROPOUT_P
    )
else:
    model = BERTClassifier.from_pretrained(
        PT_FP,
        bert_model=BERT_MODEL,
        activ_fn=ACTIV_FN,
        reset_wgts=RESET_WGTS,
        layer_norm=LAYER_NORM,
        dropout_p=DROPOUT_P
    )

model.to(device=DEVICE)
model.train()
optimizer = torch.optim.AdamW(model.parameters(), lr=LR, weight_decay=WEIGHT_DECAY)

with open(DATA_FP, 'r') as f:
    data_file = json.load(f)

train_data, test_data = data_file['train'], data_file['test']
ep_cnt, patience_cnt, best_acc = 0, 0, 0.0
loss_fn = torch.nn.BCEWithLogitsLoss()

while patience_cnt < PATIENCE:
    ep_cnt += 1
    print(f'EPOCH {ep_cnt}:')
    print()

    optimizer.zero_grad()
    permutation = torch.randperm(len(train_data)).tolist()
    train_batches = generate_data(list(map(lambda i_: train_data[i_], permutation)))
    epoch_loss, loss_cnt = 0.0, 0

    for _ in tqdm(range(-(-len(train_data) // BATCH_SIZE))):
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

    print()
    print(f'LOSS: {epoch_loss / loss_cnt}')
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

    print()
    print(f'ACC: {round(acc * 100, 5)}% (BEST={round(best_acc * 100, 5)}%; PATIENCE={patience_cnt})')
    print()
