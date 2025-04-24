
import gc
import os
import json
import torch
import pickle
from tqdm import tqdm
from collections import OrderedDict
from global_filepath import GLOBAL_FP
from transformers import AutoModelForMaskedLM, AutoTokenizer


DATA_FP = f'{GLOBAL_FP}/relpron/relpron_exs_bert_test.json'
CHK_FP = f'{GLOBAL_FP}/bert/runs/run_4/checkpoints/'
BERT_MODEL = 'bert-base-uncased'
DEVICE = 0


def avg_prec(rel_x, pred_scores_list):
    pred_scores = sorted(list(enumerate(pred_scores_list)), key=lambda z: -z[1])
    prec_cnt, out_score = 0, 0

    for k_, (idx, _) in enumerate(pred_scores):
        prec_cnt += idx in rel_x
        out_score += (prec_cnt / (k_ + 1)) * (idx in rel_x)

    return out_score / len(rel_x)


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


with open(DATA_FP, 'r') as f:
    # bert_exs = [{**{'sent': x['sent']}, **x['bert']} for x in json.load(f)]
    bert_exs = [x['bert'] for x in json.load(f)]

tokenizer = AutoTokenizer.from_pretrained(BERT_MODEL)
base_model = AutoModelForMaskedLM.from_pretrained(BERT_MODEL)
chk_fp, chk_map = (os.path.abspath(CHK_FP) + '/').replace('//', '/'), {}
checkpoint_ep, checkpoints = {}, []

for chk_fn in os.listdir(chk_fp):
    ep, mb = map(int, chk_fn[2:-4].split('_mb'))

    if ep in checkpoint_ep.keys():
        checkpoint_ep[ep].append(mb)
    else:
        checkpoint_ep.update({ep: [mb]})

for ep in sorted(checkpoint_ep.keys()):
    checkpoints.extend((ep, mb) for mb in sorted(checkpoint_ep[ep]))

for ep, mb in checkpoints:
    print(f'CHECKPOINT EP={ep}, MB={mb}:')
    print()

    base_model.to(device='cpu')

    with open(f'{chk_fp}ep{ep}_mb{mb}.chk', 'rb') as f:
        chk_sd = pt_sd_to_mlm(pickle.load(f), base_model)

    base_model.load_state_dict(chk_sd)
    base_model.to(device=DEVICE)
    base_model.eval()
    term_dict = {}

    for b in bert_exs:
        if b['trgt_tok'][0] not in term_dict.keys():
            term_dict.update({tuple(b['trgt_tok']): {'probs': [], 'rel': set()}})

    for i, b in enumerate(tqdm(bert_exs)):
        with torch.no_grad():
            bert_input = torch.tensor([b['s']], device=DEVICE)
            bert_output = torch.nn.functional.softmax(base_model(bert_input).logits, dim=-1).squeeze(0)
            term_dict[tuple(b['trgt_tok'])]['rel'].add(i)

            for k in term_dict.keys():
                if len(k) == len(b['trgt_idx']):
                    k_prob = 1.0

                    for k_pos, k_tok in zip(b['trgt_idx'], k):
                        k_prob *= bert_output[k_pos, k_tok].item()
                else:
                    k_prob = 0.0

                term_dict[k]['probs'].append(k_prob)

    chk_map.update({(ep, mb): sum(avg_prec(x['rel'], x['probs']) for x in term_dict.values()) / len(term_dict.keys())})

    print()
    print(f'MAP: {chk_map[(ep, mb)]}')
    print()
    print()

    del chk_sd
    gc.collect()
    torch.cuda.empty_cache()

print('\n\n')

abs_max_map, max_map, curr_ep, max_str = max(chk_map.values()), 0.0, -1, ''

for ep, mb in checkpoints:
    if not ep == curr_ep:
        max_map, curr_ep = max(v for (e, _), v in chk_map.items() if e == ep), ep
        max_str = '*' * (3 if max_map == abs_max_map else 1)
        print()

    print(f'EP={ep}, MB={mb}: {chk_map[(ep, mb)]}{max_str if chk_map[(ep, mb)] == max_map else ""}')
