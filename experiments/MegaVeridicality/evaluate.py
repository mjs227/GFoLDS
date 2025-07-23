
import json
import torch
import pickle
from tqdm import tqdm
from setup import GLOBAL_FP
from argparse import ArgumentParser
from collections import OrderedDict
from model.io import SWATForSCInput
from model.model import SWATForSequenceClassification
from transformers import AutoModelForMaskedLM, AutoTokenizer
from model.configs import SWATForSequenceClassificationConfig


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


def generate_data_bert(data_list):
    long_tensor = torch.LongTensor if DEVICE == 'cpu' else torch.cuda.LongTensor

    for k in range(-(-len(data_list) // BATCH_SIZE)):
        batch_data = data_list[k * BATCH_SIZE:(k + 1) * BATCH_SIZE]
        toks = tokenizer([x['s'] for x in batch_data], padding=True)
        toks_ten = torch.tensor(toks['input_ids'], device=DEVICE)[:, 1:-1]
        mask_ten = torch.tensor(toks['attention_mask'], device=DEVICE)[:, 1:-1]
        trgt_ten = torch.tensor([x['l'] for x in batch_data], device=DEVICE).type(long_tensor)

        yield toks_ten, mask_ten, trgt_ten


def generate_data_gfolds(data_list):
    for x in data_list:
        yield SWATForSCInput.from_dir_graph(x['g'], targets=x['l'])


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


bert_chkpt_help = 'Format: run_{n}/ep{k}_mb{i}.chk. If not specified, will use the original BERT '
bert_chkpt_help += 'model specified in --bert_model_type'
ap = ArgumentParser()
ap.add_argument('-g', '--gfolds_checkpoint', help='Format: run_{n}/ep{k}_mb{i}.chk', type=str, default=None)
ap.add_argument('-b', '--bert_checkpoint', help=bert_chkpt_help, type=str, default=None)
ap.add_argument('--bert_model_type', help='\'base\' or \'large\'', type=str, default='base')
ap_args = ap.parse_args()

DATA_FP = GLOBAL_FP + '/factuality/data/mv_data_bin_guc.json'
PATIENCE = 5
LR = 1e-6
WEIGHT_DECAY = 1e-5
BATCH_SIZE = 8
DEVICE = 0 if torch.cuda.is_available() else 'cpu'
SEED = 25

torch.manual_seed(SEED)
torch.cuda.manual_seed(SEED)

with open(DATA_FP, 'r') as f:
    data_file = json.load(f)

if ap_args.gfolds_checkpoint is None:
    if ap_args.bert_model_type is None:
        raise ValueError('Specify either \'--gfolds_checkpoint\' or \'--bert_model_type\' (but not both)')

    BERT_MODEL = f'bert-{ap_args.bert_model_type}-uncased'
    tokenizer = AutoTokenizer.from_pretrained(BERT_MODEL)

    if ap_args.bert_checkpoint is None:
        model = BERTClassifier(BERT_MODEL, activ_fn=torch.nn.GELU(), layer_norm=True, dropout_p=0.1)
    else:
        model = BERTClassifier.from_pretrained(
            '{}/bert/runs/{}/checkpoints/{}'.format(GLOBAL_FP, *ap_args.bert_checkpoint.split('/')),
            bert_model=BERT_MODEL
        )

    model.to(device=DEVICE)
    model.train()
    optimizer = torch.optim.AdamW(model.parameters(), lr=LR, weight_decay=WEIGHT_DECAY)

    train_data, test_data = data_file['train'], data_file['test']
    ep_cnt, patience_cnt, best_acc = 0, 0, 0.0
    loss_fn = torch.nn.BCEWithLogitsLoss()

    while patience_cnt < PATIENCE:
        ep_cnt += 1
        print(f'EPOCH {ep_cnt}:')
        print()

        optimizer.zero_grad()
        permutation = torch.randperm(len(train_data)).tolist()
        train_batches = generate_data_bert(list(map(lambda i_: train_data[i_], permutation)))
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

        test_batches = generate_data_bert(test_data)
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
elif ap_args.bert_model_type is None:
    torch.manual_seed(SEED)
    torch.cuda.manual_seed(SEED)

    model = SWATForSequenceClassification.from_swat_model(
        '{}/pretraining/{}/checkpoints/{}'.format(GLOBAL_FP, *ap_args.gfolds_checkpoint.split('/')),
        SWATForSequenceClassificationConfig(
            n_classes=2,
            head_config__layer_norm=True,
            head_config__dropout_kwargs__p=0.1,
            pooling='mean'
        )
    )
    model.to(device=DEVICE)
    model.train()
    optimizer = torch.optim.AdamW(model.parameters(), lr=LR, weight_decay=WEIGHT_DECAY)

    train_data, test_data = data_file['train'], data_file['test']
    ep_cnt, patience_cnt, best_acc = 0, 0, 0.0

    while patience_cnt < PATIENCE:
        ep_cnt += 1
        print(f'EPOCH {ep_cnt}:')
        print()

        optimizer.zero_grad()
        permutation = torch.randperm(len(train_data)).tolist()
        batch_perm = map(lambda i_: train_data[i_], permutation)
        train_batches = SWATForSCInput.generate_batches(generate_data_gfolds(batch_perm), batch_size=BATCH_SIZE)
        epoch_loss, loss_cnt = 0.0, 0

        for _ in tqdm(range(-(-len(train_data) // BATCH_SIZE))):
            batch = next(train_batches)
            model_out = model(batch)
            loss = model_out.loss()

            loss.backward()
            optimizer.step()
            epoch_loss += loss.item()
            loss_cnt += batch.batch_size

            optimizer.zero_grad()
            del batch, model_out, loss

        print()
        print(f'LOSS: {epoch_loss / loss_cnt}')
        print()

        test_batches = SWATForSCInput.generate_batches(generate_data_gfolds(test_data), batch_size=BATCH_SIZE)
        epoch_acc, acc_cnt = 0.0, 0
        model.eval()

        with torch.no_grad():
            for _ in tqdm(range(-(-len(test_data) // BATCH_SIZE))):
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

        print()
        print(f'ACC: {round(acc * 100, 5)}% (BEST={round(best_acc * 100, 5)}%; PATIENCE={patience_cnt})')
        print()
else:
    raise ValueError('Specify either \'--gfolds_checkpoint\' or \'--bert_checkpoint\' (but not both)')