
import gc
import os
import json
import nltk
import torch
import pickle
from tqdm import tqdm
from collections import OrderedDict
from model.io import SWATForMLMInput
from global_filepath import GLOBAL_FP
from model.model import SWATForMaskedLM
from nltk.tokenize import word_tokenize, sent_tokenize
from transformers import AutoTokenizer, AutoModelForMaskedLM
from pos_gen_data import TOK_FP as TOK_FP_P, BERT_TKNZR as BERT_TKNZR_P, TRGT_FP as TRGT_FP_P, POS_MAP, get_pos
from quantifier_gen_data import (
    TOK_FP as TOK_FP_Q,
    BERT_TKNZR as BERT_TKNZR_Q,
    TRGT_FP as TRGT_FP_Q,
    SINGULAR,
    PLURAL,
    BOTH
)


PROBE = 'pos'
# GFOLDS_CHK_FP = GLOBAL_FP + '/pretraining/run_2/checkpoints/'
GFOLDS_CHK_FP = None
BERT_CHK_FP = GLOBAL_FP + '/bert/runs/run_4/checkpoints/'
# BERT_CHK_FP = None
BERT_MODEL = 'bert-base-uncased'
RESULTS_FP = GLOBAL_FP + '/elem_eval/stats/bert-base-uncased/pos.json'
# RESULTS_FP = None
PRINT_TOP = False
PREC_CNT = 10  # POS only
DEVICE = 0


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


def print_prec_stats_pos(res_list, ex_list):
    prec_val, prec_cnt = {t: 0.0 for t in POS_MAP.keys()}, {t: 0 for t in POS_MAP.keys()}

    for top_preds, ex in zip(res_list, ex_list):
        prec_val[ex['pos']] += sum(p[2] for p in top_preds) / len(top_preds)  # NOT "PREC_CNT" (fn. can be imported)
        prec_cnt[ex['pos']] += 1

    print('\n\n   PREC:')
    print(f'      ALL: {sum(prec_val.values()) / sum(prec_cnt.values())} ({sum(prec_cnt.values())})')

    for tag in POS_MAP.keys():
        prec_t = 'NaN' if prec_cnt[tag] == 0.0 else (prec_val[tag] / prec_cnt[tag])
        print(f'      {tag.upper()}:   {prec_t} ({prec_cnt[tag]})')

    print('\n\n')


def print_prec_stats_quant(res_list, ex_list):
    prec_val, prec_cnt = {'s': 0.0, 'p': 0.0}, {'s': 0, 'p': 0}

    for top_preds, ex in zip(res_list, ex_list):
        prec_val[ex['type']] += sum(r for _, _, r in top_preds) / len(top_preds)
        prec_cnt[ex['type']] += 1

    print('\n\nPREC:')
    print(f'   ALL: {sum(prec_val.values()) / sum(prec_cnt.values())} ({sum(prec_cnt.values())})')
    print(f'   SG:  {prec_val["s"] / prec_cnt["s"]} ({prec_cnt["s"]})')
    print(f'   PL:  {prec_val["p"] / prec_cnt["p"]} ({prec_cnt["p"]})\n\n')


def get_prec_print_fn_pos(model_type):
    global bert_tokenizer_p

    if bert_tokenizer_p is None:
        bert_tokenizer_p = AutoTokenizer.from_pretrained(BERT_TKNZR_P)

    if model_type == 'gfolds':
        global gfolds_toks_rev_p

        if gfolds_toks_rev_p is None:
            gfolds_toks_rev_p = _get_gfolds_toks_rev(TOK_FP_P)

        def pred_data_fn(pred_data):
            return gfolds_toks_rev_p(pred_data[0]), pred_data[1], pred_data[2], ''
    else:
        def pred_data_fn(pred_data):
            return bert_tokenizer_p.decode(pred_data[0]), pred_data[1], pred_data[2], f'; {pred_data[3].upper()}'

    def prec_print_fn(top_preds, ex):
        prec = sum(p[2] for p in top_preds) / len(top_preds)
        print(f'\n\n   {bert_tokenizer_p.decode(ex["bert"]["s"][0])}\n   POS={ex["pos"].upper()}, P={prec}:')

        for i, (t, p, r, pr) in enumerate(map(pred_data_fn, top_preds), start=1):
            print(f'      {i}: {t} ({p}{pr}){"*" if r else ""}')

    return prec_print_fn


def get_prec_print_fn_quant(model_type):
    global bert_tokenizer_q

    if bert_tokenizer_q is None:
        bert_tokenizer_q = AutoTokenizer.from_pretrained(BERT_TKNZR_Q)

    if model_type == 'gfolds':
        global gfolds_toks_rev_q

        if gfolds_toks_rev_q is None:
            gfolds_toks_rev_q = _get_gfolds_toks_rev(TOK_FP_Q)

        decoder = gfolds_toks_rev_q
    else:
        decoder = bert_tokenizer_q.decode

    def prec_print_fn(top_preds, ex):
        prec = sum(r for _, _, r in top_preds) / len(top_preds)
        ex_type = 'SG' if ex['type'] == 's' else 'PL'
        print(f'\n\n   {bert_tokenizer_q.decode(ex["bert"]["s"][0])}\n   NUM={ex_type}, P={prec}:')

        for i, (t, p, r) in enumerate(top_preds, start=1):
            print(f'      {i}: {decoder(t)} ({p}){"*" if r else ""}')

    return prec_print_fn


def inf_gfolds(gfolds_model, in_ex):
    ex_input = SWATForMLMInput.from_dir_graph(in_ex['gfolds']['g'], perturb_prob=0.0)
    ex_trgt = in_ex['gfolds']['idx']
    ex_input.mask((0, ex_trgt))

    ex_output = gfolds_model(ex_input).softmax_logits
    ex_logits = ex_output[0, ex_trgt].flatten()
    del ex_input, ex_output

    return ex_logits


def inf_bert(bert_model, in_ex):
    ex_input, ex_trgt = torch.tensor(in_ex['bert']['s'], device=DEVICE), in_ex['bert']['idx']
    ex_output = bert_model(ex_input)

    ex_logits = torch.nn.functional.softmax(ex_output.logits[0, ex_trgt].flatten(), dim=0)
    del ex_input, ex_output

    return ex_logits


def cont_check(cndn, check_str):
    in_str = '' if cndn else 'y'

    while in_str not in {'yes', 'y', 'no', 'n'}:
        in_str = input(check_str + ' Continue? (y[es]/n[o]): ').strip().lower()

    if in_str[0] == 'n':
        raise KeyboardInterrupt


def init_gfolds(pos):
    model = SWATForMaskedLM.from_pretrained(f'{chk_fp}ep{checkpoints[0][0]}_mb{checkpoints[0][1]}.chk')
    model.to(device=DEVICE)
    gfolds_toks = _get_gfolds_toks_rev(TOK_FP_P if pos else TOK_FP_Q, incl_toks=(not pos))
    bert_tokenizer = AutoTokenizer.from_pretrained(BERT_TKNZR_Q) if PRINT_TOP else None

    def load_model(chk_str):
        model.load_checkpoint(chk_str)

    return 'GFoLDS', inf_gfolds, model, load_model, gfolds_toks, bert_tokenizer


def init_bert(pos):
    model = AutoModelForMaskedLM.from_pretrained(BERT_MODEL)
    bert_tokenizer = AutoTokenizer.from_pretrained(BERT_TKNZR_P if pos else BERT_TKNZR_Q)

    def load_model(chk_str):
        with open(chk_str, 'rb') as f_chk:
            chk_sd = pt_sd_to_mlm(pickle.load(f_chk), model)

        model.to(device='cpu')
        model.load_state_dict(chk_sd)
        model.to(device=DEVICE)

    return BERT_MODEL.upper(), inf_bert, model, load_model, bert_tokenizer


def init_pos():
    global bert_tokenizer_p

    with open(TRGT_FP_P, 'r') as f_data:
        data_file = json.load(f_data)

    if BERT_CHK_FP is None:
        global gfolds_toks_rev_p
        mdl_str, inf_fn, model, load_model, gfolds_toks_rev_p, bert_tokenizer_p = init_gfolds(True)
    else:
        global bert_pos_map
        mdl_str, inf_fn, model, load_model, bert_tokenizer_p = init_bert(True)
        bert_pos_map = {}

        for k, v in POS_MAP.items():
            bert_pos_map.update({v_: k for v_ in v})

    calc_prec_fn = _get_pos_calc_prec_fn(mdl_str.lower())

    return 'POS', mdl_str, load_model, data_file, inf_fn, model, calc_prec_fn


def init_quant():
    global bert_tokenizer_q

    with open(TRGT_FP_Q, 'r') as f_data:
        data_file = json.load(f_data)

    if BERT_CHK_FP is None:
        global gfolds_toks_rev_q
        mdl_str, inf_fn, model, load_model, (gfolds_toks_rev_q, gfolds_toks), bert_tokenizer_q = init_gfolds(False)
        trgt_toks = {
            's': set(SINGULAR.keys()).union(set(BOTH.keys())),
            'p': set(PLURAL.keys()).union(set(BOTH.keys()))
        }

        for k, v in trgt_toks.items():
            trgt_toks[k] = set(map(lambda x: gfolds_toks[x], v))
    else:
        mdl_str, inf_fn, model, load_model, bert_tokenizer_q = init_bert(False)
        trgt_toks = {
            's': set(SINGULAR.values()).union(set(BOTH.values())).union({'an'}),
            'p': set(PLURAL.values()).union(set(BOTH.values()))
        }

        for k, v in trgt_toks.items():
            trgt_toks[k] = set(map(lambda x: bert_tokenizer_q(x)['input_ids'][1], v))

    calc_prec_fn = _get_quant_calc_prec_fn(mdl_str.lower(), trgt_toks)

    return 'QUANT', mdl_str, load_model, data_file, inf_fn, model, calc_prec_fn


def run_probe(probe_type, mdl_str, load_model, data_file, inf_fn, model, calc_prec_fn):
    if is_printing:
        final_print_fn = print_prec_stats_pos if probe_type == 'POS' else print_prec_stats_quant
    else:
        final_print_fn = lambda *_: None

    print(f'EVALUATING {mdl_str} ON {probe_type} PROBE...')
    print()
    print()

    for chk_i, (ep, mb) in enumerate(tqdm_outer(checkpoints)):
        if is_printing:
            print(f'CHECKPOINT EP={ep}, MB={mb} ({chk_i + 1}/{len(checkpoints)}):\n\n')

        load_model(f'{chk_fp}ep{ep}_mb{mb}.chk')
        gc.collect()
        torch.cuda.empty_cache()
        model.eval()
        chk_res = []

        with torch.no_grad():
            for d in tqdm_inner(data_file):
                d_logits = inf_fn(model, d)
                chk_res.append(calc_prec_fn(d_logits, d))
                del d_logits

        final_print_fn(chk_res, data_file)

        res_file['res'].append(chk_res)


def _get_gfolds_toks_rev(tok_fp, incl_toks=False):
    with open(tok_fp, 'r') as f_tok:
        gfolds_toks = json.load(f_tok)['tokenizer_dict']['v']

    gfolds_toks_rev = {v: k for k, v in gfolds_toks.items()}

    if incl_toks:
        return (lambda z: gfolds_toks_rev[z]), gfolds_toks

    del gfolds_toks

    return lambda z: gfolds_toks_rev[z]


def _prec_bert_metadata(ex):
    return bert_tokenizer_p.decode(ex['bert']['s'][0][1:-1])


def _pred_prec_bert(max_pred, logits, ex, s_decoded):
    def find_trgt_pos(tagged_s, in_s):
        out_s, len_out, len_in = '', 0, len(in_s)

        for w, tag in tagged_s:
            if in_s[len_out:].strip().startswith('[MASK]'):  # w is target
                return tag

            n_space = 0
            len_out += len(w)

            while not out_s + (' ' * n_space) + w == in_s[:len_out + n_space]:
                n_space += 1

                if n_space > len_in:
                    raise StopIteration

            out_s = out_s + (' ' * n_space) + w
            len_out += n_space

    pred_w = bert_tokenizer_p.decode(max_pred)

    if '#' in pred_w:
        pred_pos = 'x'
    else:
        pred_pos = find_trgt_pos(
            nltk.pos_tag(word_tokenize(sent_tokenize(s_decoded.replace('[MASK]', pred_w))[0])),
            s_decoded
        )
        pred_pos = bert_pos_map.get(pred_pos.lower(), 'x')

    return max_pred, logits[max_pred].item(), pred_pos == ex['pos'], pred_pos


def _get_pos_calc_prec_fn(model_type):
    if model_type == 'gfolds':
        prec_md_fn = lambda *_: None

        def pred_prec_fn(max_pred, logits, ex, *_):
            return max_pred, logits[max_pred].item(), (get_pos(gfolds_toks_rev_p(max_pred)) == ex['pos'])
    else:
        assert model_type in {'bert-base-uncased', 'bert-large-uncased'}
        pred_prec_fn, prec_md_fn = _pred_prec_bert, _prec_bert_metadata

    print_fn = get_prec_print_fn_pos(model_type) if PRINT_TOP else lambda *_: None

    def calc_prec_fn(logits, ex):
        top_preds, ex_md = [], prec_md_fn(ex)

        for _ in range(PREC_CNT):
            max_pred = torch.argmax(logits).item()
            top_preds.append(pred_prec_fn(max_pred, logits, ex, ex_md))
            logits[max_pred] = 0.0

        print_fn(top_preds, ex)

        return top_preds

    return calc_prec_fn


def _get_quant_calc_prec_fn(model_type, trgt_toks):
    print_fn = get_prec_print_fn_quant(model_type) if PRINT_TOP else lambda *_: None

    def calc_prec_fn(logits, ex):
        top_preds, rel = [], trgt_toks[ex['type']]

        for _ in range(len(rel)):
            max_pred = torch.argmax(logits).item()
            top_preds.append((max_pred, logits[max_pred].item(), max_pred in rel))
            logits[max_pred] = 0.0

        print_fn(top_preds, ex)

        return top_preds

    return calc_prec_fn


gfolds_toks_rev_p, bert_tokenizer_p, bert_pos_map = None, None, None
gfolds_toks_rev_q, bert_tokenizer_q = None, None

if __name__ == '__main__':
    assert not ((GFOLDS_CHK_FP is None) == (BERT_CHK_FP is None))
    assert PROBE.strip().lower() in {'pos', 'quantifier'}

    cont_check(RESULTS_FP is None, 'RESULTS_FP is None (results will not be recorded...)!')
    cont_check(PRINT_TOP, 'PRINT_TOP=True (this might get messy...).')

    is_printing = RESULTS_FP is None or PRINT_TOP
    tqdm_outer = (lambda x: x) if is_printing else tqdm
    tqdm_inner = tqdm if RESULTS_FP is None and not PRINT_TOP else lambda x: x

    res_file = {'type': PROBE.strip().lower(), 'model': 'gfolds' if BERT_CHK_FP is None else BERT_MODEL, 'res': []}
    chk_fp = (os.path.abspath(GFOLDS_CHK_FP if BERT_CHK_FP is None else BERT_CHK_FP) + '/').replace('//', '/')
    checkpoints = []

    if os.path.isdir(chk_fp):
        ep_mbs = {}

        for chk_fn in os.listdir(chk_fp):
            ep_, mb_ = map(int, chk_fn[2:-4].split('_mb'))

            if ep_ in ep_mbs.keys():
                ep_mbs[ep_].append(mb_)
            else:
                ep_mbs.update({ep_: [mb_]})

        for k_ in sorted(ep_mbs.keys()):
            checkpoints.extend((k_, w) for w in sorted(ep_mbs[k_]))
    else:
        assert chk_fp.endswith('.chk/')
        chk_fp, chk_fn = chk_fp[:-1].rsplit('/', 1)
        chk_fp = chk_fp + '/'
        checkpoints.append(tuple(map(int, chk_fn[2:-4].split('_mb'))))

    run_probe(*(init_pos() if res_file['type'] == 'pos' else init_quant()))

    if RESULTS_FP is not None:
        with open(os.path.abspath(RESULTS_FP), 'w') as f_res:
            json.dump(res_file, f_res)
