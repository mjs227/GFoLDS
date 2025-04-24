
import os
import re
import gc
import os
import json
import torch
from tqdm import tqdm
from model import GRPH_IDX_IDX
from model.io import SWATForMLMInput
from global_filepath import GLOBAL_FP
from model.model import SWATForMaskedLM
from model.graphtools import DMRSGraphParser


DATA_FP = f'{GLOBAL_FP}/relpron/relpron_exs_test.json'
CHK_FP = f'{GLOBAL_FP}/pretraining/run_2/checkpoints/'
PARSER_FP = GLOBAL_FP + '/tokenizer_config.json'


def avg_prec(rel_x, pred_scores_x):
    prec_cnt, out_score = 0, 0

    for k_, (idx, _) in enumerate(sorted(enumerate(pred_scores_x), key=lambda z: -z[1]), start=1):
        prec_cnt += idx in rel_x
        out_score += (prec_cnt / k_) * (idx in rel_x)

    return out_score / len(rel_x)


with open(DATA_FP, 'r') as f:
    gfolds_exs = [{**{'s': x['sent']}, **x['gfolds']} for x in json.load(f)]

graph_parser = DMRSGraphParser.from_pretrained(PARSER_FP)
chk_map, checkpoint_ep, checkpoints = {}, {}, []

for fn in os.listdir(CHK_FP):
    ep, mb = map(lambda x: int(re.sub(r'\D', '', x)), fn.split('_')[-2:])

    if ep in checkpoint_ep.keys():
        checkpoint_ep[ep].append(mb)
    else:
        checkpoint_ep.update({ep: [mb]})

for ep in sorted(checkpoint_ep.keys()):
    checkpoints.extend((ep, mb) for mb in sorted(checkpoint_ep[ep]))

for ep, mb in checkpoints:
    if (ep, mb) in chk_map.keys():
        continue

    print(f'CHECKPOINT EP={ep}, MB={mb}:')
    print()

    gfolds_model = SWATForMaskedLM.from_pretrained(f'{CHK_FP}ep{ep}_mb{mb}.chk')
    gfolds_model.to(device=0, dtype=torch.float32)
    gfolds_model.eval()
    term_dict = {}

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

            del gfolds_input, gfolds_output, gfolds_trgt

    chk_map.update({(ep, mb): sum(avg_prec(x['rel'], x['probs']) for x in term_dict.values()) / len(term_dict.keys())})

    print()
    print(f'MAP: {chk_map[(ep, mb)]}')
    print()
    print()

    del gfolds_model
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
