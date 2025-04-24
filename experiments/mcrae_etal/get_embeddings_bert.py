
import json
import torch
import pickle
from collections import OrderedDict
from global_filepath import GLOBAL_FP
from transformers import AutoModelForMaskedLM, AutoTokenizer


def pt_sd_to_mlm(in_sd, bert_mlm):
    mlm_sd_keys, in_sd_keys = map(set, (bert_mlm.state_dict().keys(), in_sd.keys()))

    if in_sd_keys == mlm_sd_keys:
        return in_sd

    if mlm_sd_keys <= in_sd_keys:
        pfx = 0
    else:
        assert mlm_sd_keys <= {k[5:] for k in in_sd_keys}
        pfx = 5

    new_sd = OrderedDict()

    for k_, v_ in in_sd.items():
        if k_[pfx:] in mlm_sd_keys:
            new_sd.update({k_[pfx:]: v_})

    return new_sd


PT_FP = GLOBAL_FP + '/bert/runs/run_4/checkpoints/ep3_mb199.chk'
FILE_FP = GLOBAL_FP + '/prop_inf/mcrae_concept_sents_all'
FEAT_FP = GLOBAL_FP + '/prop_inf/mcrae_features.json'
OUT_FP = GLOBAL_FP + '/prop_inf/mcrae_data_bert_comp_base.pkl'
BERT_MODEL = 'bert-base-uncased'
DEVICE = 0

model = AutoModelForMaskedLM.from_pretrained(BERT_MODEL).bert
tokenizer = AutoTokenizer.from_pretrained(BERT_MODEL)

if PT_FP is not None:
    with open(PT_FP, 'rb') as f:
        pt_sd = pt_sd_to_mlm(pickle.load(f), model)

    model.load_state_dict(pt_sd)

model.to(device=DEVICE)
model.eval()
mask_id, out_file = tokenizer('[MASK]')['input_ids'][1], []

with open(FEAT_FP, 'r') as f:
    feat_file = json.load(f)

with open(FILE_FP, 'r') as f:
    for line in map(lambda x: x.strip(), f):
        term, sent = map(lambda x: x.strip(), line.split(':'))
        sent = sent + ('' if sent[-1] == '.' else '.')

        if '|' in term:
            term, current_term = map(lambda z: z.strip(), term.split('|'))
        elif '_' in term:
            current_term = term.split('_')[0]
        else:
            current_term = term

        with torch.no_grad():
            toks = tokenizer(sent.replace(current_term, '[MASK]'))['input_ids']
            toks1 = toks[:next(i for i in range(len(toks)) if toks[i] == mask_id)]
            idx1 = len(toks1)
            toks2 = toks[idx1 + 1:]
            trgt_toks = tokenizer(current_term)['input_ids'][1:-1]
            idx2 = idx1 + len(trgt_toks)

            model_output = model(torch.tensor([toks1 + trgt_toks + toks2], device=DEVICE)).last_hidden_state
            trgt_output = model_output[:, idx1:idx2, :].squeeze(0)

            out_file.append({
                'term': term,
                'features': {int(k): v for k, v in feat_file[term].items()},
                'embedding': (torch.sum(trgt_output, dim=0).flatten() / trgt_output.size(0)).to(device='cpu')
            })

with open(OUT_FP, 'wb') as f_out:
    pickle.dump(out_file, f_out)