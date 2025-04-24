
import json
import torch
import random
from tqdm import tqdm
from copy import deepcopy
from global_filepath import GLOBAL_FP
from model.graphtools import SWADirGraph


CONFIG = [
    {'NUM_NEG': 6, 'MULTI_NEG': True, 'CREATE_ADV': True, 'CREATE_TRAIN': True},
]
DATA_IN_FP = GLOBAL_FP + '/nli/'
DATA_OUT_FP = GLOBAL_FP + '/double_negation/data/'
TOK_FP = GLOBAL_FP + 'tokenizer_config.json'
SEED = 1


_LBL_DICT = {
    0: {
        'c': 'c',
        'n': 'n',
        'e': 'e'
    },
    1: {
        'c': 'e',
        'n': 'n',
        'e': 'c'
    }
}

with open(DATA_IN_FP + 'dev', 'r') as f:
    len_dev = len(json.load(f))

with open(DATA_IN_FP + 'train', 'r') as f:
    train_file = json.load(f)
    len_train = len(train_file)

with open(TOK_FP, 'r') as f:
    tok_dict = json.load(f)['tokenizer_dict']

SF, PROG, PERF = tok_dict['f']['[SF:prop]'], tok_dict['f']['[PROG:-]'], tok_dict['f']['[PERF:-]']
UNTENSED, PRES = tok_dict['f']['[TENSE:untensed]'], tok_dict['f']['[TENSE:pres]']
MOOD = tok_dict['f']['[MOOD:indicative]']
NEG, TRUE = tok_dict['v']['neg'], tok_dict['v']['true_a_of']
ARG1 = tok_dict['e']['ARG1']

torch.manual_seed(SEED)
random.seed(SEED)


def get_neg_fn():
    def neg_fn(grph_, curr_trgt_):
        neg_id = grph_.add_node(NEG)
        true_id = grph_.add_node(TRUE)
        grph_.add_edge(true_id, curr_trgt_, ARG1)
        grph_.add_edge(neg_id, true_id, ARG1)

        for f_id in (SF, PROG, PERF, MOOD, UNTENSED):
            grph_.add_feature(neg_id, f_id)

        return neg_id, true_id, (SF, PROG, PERF, MOOD, PRES)

    def neg_(g, n=1):
        grph = SWADirGraph.from_dict(g)
        if_then_id = min(grph.nodes.keys())
        curr_trgt = next(x for x, y in grph.adj_out[if_then_id] if y == ARG1)  # top id of S2
        grph.remove_edge(if_then_id, curr_trgt, ARG1)

        for _ in range(n):
            curr_trgt, adj_id, feats = neg_fn(grph, curr_trgt)

            for f_id in feats:
                grph.add_feature(adj_id, f_id)

        grph.add_edge(if_then_id, curr_trgt, ARG1)

        return grph.save()

    return neg_


for config in tqdm(CONFIG):
    assert set(config.keys()) == {'NUM_NEG', 'MULTI_NEG', 'CREATE_ADV', 'CREATE_TRAIN'}
    assert isinstance(config['NUM_NEG'], int) and config['NUM_NEG'] > 0
    assert config['CREATE_ADV'] or config['CREATE_TRAIN']

    neg = get_neg_fn()
    train_file_out = {k: [] for k in _LBL_DICT[0].keys()}
    adv_file_out = {k: [] for k in _LBL_DICT[0].keys()}

    train_file_indices = torch.randperm(len_train).tolist()
    lbl_cnt_train = {k: 0 for k in train_file_out.keys()}
    lbl_cnt_adv = {k: 0 for k in adv_file_out.keys()}
    # lbl_lim_train = (len_dev // 3) if config['CREATE_TRAIN'] else 0
    # lbl_lim_adv = (len_dev // 6) if config['CREATE_ADV'] else 0
    lbl_lim_train = 3333 if config['CREATE_TRAIN'] else 0  # TODO
    lbl_lim_adv = 1666 if config['CREATE_ADV'] else 0
    max_size, size_cnt, i = (lbl_lim_train + lbl_lim_adv) * 3, 0, -1

    while size_cnt < max_size:
        i += 1
        item = train_file[train_file_indices[i]]
        label = item['lbl']

        if label in lbl_cnt_train.keys() and item['grm'] and '?' not in item['s']['s2']:
            if lbl_cnt_train[label] < lbl_lim_train:
                lbl_cnt_train[label] += 1
                out_file = train_file_out
            elif lbl_cnt_adv[label] < lbl_lim_adv:
                lbl_cnt_adv[label] += 1
                out_file = adv_file_out
            else:
                continue

            item_ = deepcopy(item)
            item_.update({'index': train_file_indices[i]})
            out_file[label].append(item_)
            size_cnt += 1

    for out_file in (train_file_out, adv_file_out):
        len_file = len(out_file['n'])
        assert all(len(x) == len_file for _, x in out_file.items())

        if len_file > 0:  # i.e. if CREATE_[SPLIT] = True
            for lbl, lbl_list in out_file.items():
                random.shuffle(lbl_list)

                if config['MULTI_NEG']:
                    split_size = int(len_file // config['NUM_NEG'])
                    total_coverage = split_size * config['NUM_NEG']
                    remainder = len_file - total_coverage
                    num_neg_fn = lambda z: z + 1
                else:
                    total_coverage, remainder, split_size = 0, 0, len_file
                    num_neg_fn = lambda z: config['NUM_NEG']

                for i in range(config['NUM_NEG'] if config['MULTI_NEG'] else 1):
                    num_neg = num_neg_fn(i)
                    lbl_dict = _LBL_DICT[num_neg % 2]

                    for j in range(i * split_size, (i + 1) * split_size):
                        lbl_list[j].update({
                            'g': neg(lbl_list[j]['g'], n=num_neg),
                            'lbl': lbl_dict[lbl]
                        })

                if remainder > 0:
                    for i in range(remainder):
                        num_neg = i % config['NUM_NEG']
                        lbl_list[total_coverage + i].update({
                            'g': neg(lbl_list[total_coverage + i]['g'], n=num_neg),
                            'lbl': _LBL_DICT[num_neg % 2][lbl]
                        })

    train_file_out = sum((x for _, x in train_file_out.items()), [])
    adv_file_out = sum((x for _, x in adv_file_out.items()), [])
    random.shuffle(train_file_out)
    random.shuffle(adv_file_out)
    multi = '_MULTI' if config['MULTI_NEG'] else ''

    for split, out_list in (('adv', adv_file_out), ('train', train_file_out)):
        if len(out_list) > 0:
            with open(f'{DATA_OUT_FP}{split}_{config["NUM_NEG"]}{multi}.json', 'w') as f:
                json.dump(out_list, f)
