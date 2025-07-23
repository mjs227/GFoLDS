
import json
import torch
import pickle
from tqdm import tqdm
from setup import GLOBAL_FP
from model import GRPH_IDX_IDX
from argparse import ArgumentParser
from collections import OrderedDict
from model.io import SWATForMLMInput
from model.model import SWATForMaskedLM
from transformers import AutoModelForMaskedLM, AutoTokenizer


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


bert_chkpt_help = 'Format: run_{n}/ep{k}_mb{i}.chk. If not specified, will use the original BERT '
bert_chkpt_help += 'model specified in --bert_model_type'
ap = ArgumentParser()
ap.add_argument('--split', help='\'test\' or \'dev\'', type=str, default='test')
ap.add_argument('-g', '--gfolds_checkpoint', help='Format: run_{n}/ep{k}_mb{i}.chk', type=str, default=None)
ap.add_argument('-b', '--bert_checkpoint', help=bert_chkpt_help, type=str, default=None)
ap.add_argument('--bert_model_type', help='\'base\' or \'large\'', type=str, default=None)
ap_args = ap.parse_args()

DEVICE = 0 if torch.cuda.is_available() else 'cpu'

with open(f'{GLOBAL_FP}/relpron/relpron_exs_{ap_args.split}.json', 'r') as f:
    split_data = json.load(f)

if ap_args.gfolds_checkpoint is None:
    if ap_args.bert_model_type is None:
        raise ValueError('Specify either \'--gfolds_checkpoint\' or \'--bert_model_type\' (but not both)')

    BERT_MODEL = f'bert-{ap_args.bert_model_type}-uncased'
    model = AutoModelForMaskedLM.from_pretrained(BERT_MODEL)
    tokenizer = AutoTokenizer.from_pretrained(BERT_MODEL)

    if ap_args.bert_checkpoint is not None:
        with open('{}/bert/runs/{}/checkpoints/{}'.format(GLOBAL_FP, *ap_args.bert_checkpoint.split('/')), 'rb') as f:
            model.load_state_dict(pt_sd_to_mlm(pickle.load(f), model))

    model.to(device=DEVICE)
    model.eval()
    bert_exs, term_dict = [{**{'sent': x['sent']}, **x['bert']} for x in split_data], {}

    for b in bert_exs:
        if b['trgt_tok'][0] not in term_dict.keys():
            term_dict.update({tuple(b['trgt_tok']): {'probs': [], 'rel': set()}})

    for i, b in enumerate(tqdm(bert_exs)):
        with torch.no_grad():
            bert_input = torch.tensor([b['s']], device=DEVICE)
            bert_output = torch.nn.functional.softmax(model(bert_input).logits, dim=-1).squeeze(0)
            term_dict[tuple(b['trgt_tok'])]['rel'].add(i)

            for k in term_dict.keys():
                if len(k) == len(b['trgt_idx']):
                    k_prob = 1.0

                    for k_pos, k_tok in zip(b['trgt_idx'], k):
                        k_prob *= bert_output[k_pos, k_tok].item()
                else:
                    k_prob = 0.0

                term_dict[k]['probs'].append(k_prob)
elif ap_args.bert_checkpoint is None:
    gfolds_model = SWATForMaskedLM.from_pretrained(
        '{}/pretraining/{}/checkpoints/{}'.format(GLOBAL_FP, *ap_args.gfolds_checkpoint.split('/'))
    )
    gfolds_model.to(device=0, dtype=torch.float32)
    gfolds_model.eval()
    gfolds_exs, term_dict = [{**{'s': x['sent']}, **x['gfolds']} for x in split_data], {}

    for g in gfolds_exs:
        if g['trgt_tok'] not in term_dict.keys():
            term_dict.update({g['trgt_tok']: {'probs': [], 'rel': set()}})

    for i, g in enumerate(tqdm(gfolds_exs)):
        trgt_idx = g['g']['n'][g['trgt_idx']][GRPH_IDX_IDX]

        with torch.no_grad():
            gfolds_input = next(SWATForMLMInput.generate_batches(
                [g['g']],
                batch_size=1,
                swat_input_kwargs={'perturb_prob': 0.0}
            ))
            gfolds_input.mask(0, trgt_idx)
            gfolds_output = gfolds_model(gfolds_input).softmax_logits
            gfolds_trgt = gfolds_output[0, trgt_idx].flatten()
            term_dict[g['trgt_tok']]['rel'].add(i)

            for k in term_dict.keys():
                term_dict[k]['probs'].append(gfolds_trgt[k].item())
else:
    raise ValueError('Specify either \'--gfolds_checkpoint\' or \'--bert_model_type\' (but not both)')

print()
print(f'MAP: {sum(avg_prec(x["rel"], x["probs"]) for x in term_dict.values()) / len(term_dict.keys())}')
