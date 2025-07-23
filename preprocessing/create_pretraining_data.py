
import os
import json
import warnings
import traceback
from setup import GLOBAL_FP
from datetime import datetime
from argparse import ArgumentParser
from transformers import AutoTokenizer
from multiprocessing import Pool, Manager
from model.graphtools import DMRSGraphParser
from model import GRPH_LABEL_IDX, DEFAULT_TOKENIZER_DICT


def worker_manager_mlm(comm_list, status_list, file_list, stats_dict):
    total_parsed, total_sents, len_file_list = 0, 0, len(file_list)
    curr_len, prev_len = (len(status_list),) * 2

    for _, (np, nt) in stats_dict.items():
        total_parsed += np
        total_sents += nt

    while sum(x is None for x in status_list) < len(status_list):
        curr_len = round(sum(1 if z in {None, -1} else z for z in status_list), 2)

        if not curr_len == prev_len:
            print('; '.join(f'{i_}:{round(c, 3)}' for i_, c in enumerate(status_list) if c not in {None, -1}))

        new_files, prev_len = False, curr_len

        for i_ in range(len(status_list)):
            if status_list[i_] == -1:
                if isinstance(comm_list[i_], tuple):  # worker output
                    new_files = True
                    file_name, num_parsed, num_total = comm_list[i_]
                    stats_dict.update({file_name.split('.')[0]: (num_parsed, num_total)})
                    total_parsed += num_parsed
                    total_sents += num_total
                    file_perc = round((num_parsed / num_total) * 100, 2)
                    total_perc = round((total_parsed / total_sents) * 100, 2)

                    print_to_log(f'STATS FILE {file_name} ({len(stats_dict)}/{len_file_list} FILES PARSED):')
                    print_to_log(f'    FILE:      {num_parsed}/{num_total} PARSED ({file_perc}%)')
                    print_to_log(f'    TOTAL:     {total_parsed}/{total_sents} PARSED ({total_perc}%)')
                    print_to_log(f'    DATE/TIME: {datetime.now()}')

                if len(file_list) > 0:
                    comm_list[i_] = file_list.pop(0)
                    status_list[i_] = 0
                else:
                    status_list[i_] = None  # kill worker

        if new_files:
            with open(f'{OUTPUT_FP}misc/parse_stats.json', 'w') as f_stats:
                json.dump(dict(stats_dict), f_stats)


def tokenizer_worker_mlm(self_idx, comm_list, status_list):
    tokenizer = AutoTokenizer.from_pretrained('bert-base-uncased')

    while True:
        while status_list[self_idx] == -1:  # waiting for assignment
            pass
        if status_list[self_idx] is None:  # order 66
            break

        curr_fn, out_file = comm_list[self_idx], []

        with open(INPUT_FP + curr_fn, 'r') as text_f:
            text_file = json.load(text_f)

        for i_, sentence in enumerate(text_file):
            try:
                out_file.append(tokenizer(sentence)['input_ids'])
            except Exception as e:
                if curr_fn in os.listdir(OUTPUT_FP + 'exceptions/'):
                    open_type, init_str = 'a', '\n\n\n--------------------------------------------\n\n\n'
                else:
                    open_type, init_str = 'w', ''

                with open(f'{OUTPUT_FP}exceptions/{curr_fn}', open_type) as f_err:
                    f_err.write(init_str + str(type(e)) + ' (S=' + str(i_) + '):\n\n\n')
                    f_err.write('\n'.join(traceback.format_tb(e.__traceback__)))

            status_list[self_idx] = (i_ + 1) / len(text_file)

        if len(out_file) > 0:
            with open(f'{OUTPUT_FP}parsed_files/{curr_fn}.json', 'w') as f_fn:
                json.dump(out_file, f_fn)

        file_name = comm_list[self_idx]
        comm_list[self_idx] = (file_name, len(out_file), len(text_file))
        status_list[self_idx] = -1


def tokenizer_wrapper_mlm(a):
    return tokenizer_worker_mlm(*a) if isinstance(a[0], int) else worker_manager_mlm(*a)


def tokenizer_worker_nsp(self_files):
    tokenizer = AutoTokenizer.from_pretrained('bert-base-uncased')

    for self_fn in self_files:
        out_file = []

        with open(INPUT_FP + self_fn, 'r') as f:
            try:
                while True:
                    line1, line2 = next(f).strip(), next(f).strip()

                    if len(line1) > 0 and len(line2) > 0:
                        line1_toks, line2_toks = map(lambda x: tokenizer(x)['input_ids'][1:-1], (line1, line2))

                        if sum(map(len, (line1_toks, line2_toks))) <= 509:
                            out_file.append((line1, line2))
            except StopIteration:
                pass

        with open(OUTPUT_FP + self_fn, 'w') as f:
            json.dump(out_file, f)

        print(f'FILE {self_fn} DONE')


def tokenizer_manager_gfolds(comm_list, status_list, file_list, stats_dict):
    total_parsed, total_sents, len_file_list = 0, 0, len(file_list)
    curr_len, prev_len = (len(status_list),) * 2

    for _, (np, nt) in stats_dict.items():
        total_parsed += np
        total_sents += nt

    while sum(x is None for x in status_list) < len(status_list):
        curr_len = round(sum(1 if z in {None, -1} else z for z in status_list), 4)

        if not curr_len == prev_len:
            print('; '.join(f'{i_}:{round(c, 4)}' for i_, c in enumerate(status_list) if c not in {None, -1}))

        new_files, prev_len = False, curr_len

        for i_ in range(len(status_list)):
            if status_list[i_] == -1:
                if isinstance(comm_list[i_], tuple):  # worker output
                    new_files = True
                    file_name, num_parsed, num_total = comm_list[i_]
                    stats_dict.update({file_name.split('.')[0]: (num_parsed, num_total)})
                    total_parsed += num_parsed
                    total_sents += num_total
                    file_perc = round((num_parsed / num_total) * 100, 2)
                    total_perc = round((total_parsed / total_sents) * 100, 2)

                    print_to_log(f'STATS FILE {file_name} ({len(stats_dict)}/{len_file_list} FILES PARSED):')
                    print_to_log(f'    FILE:      {num_parsed}/{num_total} PARSED ({file_perc}%)')
                    print_to_log(f'    TOTAL:     {total_parsed}/{total_sents} PARSED ({total_perc}%)')
                    print_to_log(f'    DATE/TIME: {datetime.now()}')

                if len(file_list) > 0:
                    comm_list[i_] = file_list.pop(0)
                    status_list[i_] = 0
                else:
                    status_list[i_] = None  # kill worker

        if new_files:
            with open(f'{OUTPUT_FP}misc/parse_stats.json', 'w') as f_stats:
                json.dump(dict(stats_dict), f_stats)


def tokenizer_worker_gfolds(self_idx, init_tok_dict, comm_list, status_list):
    graph_parser = DMRSGraphParser.init_for_pretraining(tokenizer_dict=init_tok_dict, **MRS_GRAPH_PARSER_KWARGS)

    while True:
        while status_list[self_idx] == -1:  # waiting for assignment
            pass
        if status_list[self_idx] is None:  # order 66
            break

        curr_fn, out_file, len_text_f = comm_list[self_idx], [], 0

        with open(INPUT_FP + curr_fn, 'r') as text_f:
            text_file = text_f.readlines()

        for i_, sentence in enumerate(map(lambda z: z.strip(), text_file)):
            if len(sentence) > 0:
                len_text_f += 1

                try:
                    grph = graph_parser(sentence, save=True)

                    if grph is not None:
                        out_file.append(grph)
                except Exception as e:
                    if curr_fn in os.listdir(OUTPUT_FP + 'exceptions/'):
                        open_type, init_str = 'a', '\n\n\n--------------------------------------------\n\n\n'
                    else:
                        open_type, init_str = 'w', ''

                    with open(f'{OUTPUT_FP}exceptions/{curr_fn}', open_type) as f_err:
                        f_err.write(init_str + str(type(e)) + ' (S=' + str(i_) + '):\n\n\n')
                        f_err.write('\n'.join(traceback.format_tb(e.__traceback__)))

                status_list[self_idx] = (i_ + 1) / len(text_file)

        if len(out_file) > 0:
            with open(f'{OUTPUT_FP}parsed_files/{self_idx}_{curr_fn}', 'w') as f_fn:
                json.dump(out_file, f_fn)

        file_name = comm_list[self_idx]
        comm_list[self_idx] = (file_name, len(out_file), len_text_f)

        with open(f'{OUTPUT_FP}misc/{self_idx}.json', 'w') as f_dict:
            json.dump(graph_parser.tokenizer_dict, f_dict)

        status_list[self_idx] = -1


def tokenizer_wrapper_gfolds(a):
    return tokenizer_worker_gfolds(*a) if isinstance(a[0], int) else tokenizer_manager_gfolds(*a)


def translation_manager(comm_list, all_file_lens):
    total_file_lens, cnt = sum(all_file_lens), 0
    curr_file_lens = [0] * len(all_file_lens)

    while cnt < total_file_lens:
        if len(comm_list) > cnt:
            w_idx, fn_ = comm_list[cnt]
            curr_file_lens[w_idx] += 1
            cnt += 1

            print_str = f'FILE \"{fn_}\" TRANSLATED (WORKER {w_idx}: {curr_file_lens[w_idx]}/{all_file_lens[w_idx]}'
            print_to_both(f'        {print_str} | TOTAL: {cnt}/{total_file_lens})')


def translation_worker(self_idx, comm_list, self_translate, self_files):
    self_files.sort(key=lambda n: int(n.replace('.json', '')))

    for fn_ in self_files:
        with open(f'{OUTPUT_FP}parsed_files/{self_idx}_{fn_}', 'r') as f_fn:
            file_fn = json.load(f_fn)

        for i_ in range(len(file_fn)):
            for k_ in file_fn[i_]['n'].keys():
                file_fn[i_]['n'][k_][GRPH_LABEL_IDX] = self_translate['v'][file_fn[i_]['n'][k_][GRPH_LABEL_IDX]]

            for k_ in file_fn[i_]['f'].keys():
                for j_ in range(len(file_fn[i_]['f'][k_])):
                    file_fn[i_]['f'][k_][j_] = self_translate['f'][file_fn[i_]['f'][k_][j_]]

            for k_ in file_fn[i_]['i'].keys():
                for j_ in range(len(file_fn[i_]['i'][k_])):
                    file_fn[i_]['i'][k_][j_][1] = self_translate['e'][file_fn[i_]['i'][k_][j_][1]]

            # for k_ in file_fn[i_]['g']['o'].keys():
            #     for j_ in range(len(file_fn[i_]['g']['o'][k_])):
            #         file_fn[i_]['g']['o'][k_][j_][1] = self_translate['e'][file_fn[i_]['g']['o'][k_][j_][1]]

        with open(f'{OUTPUT_FP}parsed_files/{fn_}', 'w') as f_fn:
            json.dump(file_fn, f_fn)

        comm_list.append((self_idx, fn_))


def translation_wrapper(a):
    return translation_worker(*a) if isinstance(a[0], int) else translation_manager(*a)


def print_to_log(print_str):
    with open(f'{OUTPUT_FP}misc/log.txt', 'a') as f_log:
        f_log.write('\n' + print_str)


def print_to_both(print_str):
    print_to_log(print_str)
    print(print_str)


ap = ArgumentParser()
ap.add_argument('-n', '--num_workers', help='Number of parsing threads', type=int, required=True)
ap.add_argument('-b', '--bert', help='Generate BERT pretraining data', action='store_true')
ap.add_argument('--nsp', help='Create NSP data (BERT only)', action='store_true')
args = ap.parse_args()
assert not (args.nsp and not args.bert)

NUM_WORKERS = args.num_workers

if args.bert:
    os.environ['TOKENIZERS_PARALLELISM'] = 'true'

    if args.nsp:
        INPUT_FP = GLOBAL_FP + '/raw_data/en_wiki_sents/all_sents/'
        OUTPUT_FP = GLOBAL_FP + '/bert/data/nsp/preproc_data/'
        input_file_names, worker_files = set(os.listdir(INPUT_FP)), [[] for _ in range(NUM_WORKERS)]

        for i, fn in enumerate(input_file_names):
            worker_files[i % NUM_WORKERS].append(fn)

        with Pool() as pool:
            pool.map(tokenizer_worker_nsp, worker_files)

        print('\n\nDONE!')
    else:
        INPUT_FP = GLOBAL_FP + '/raw_data/en_wiki_sents/chosen_sents/'
        OUTPUT_FP = GLOBAL_FP + '/bert/data/mlm/preproc_data/'

        output_fp_folders, input_file_names = set(os.listdir(OUTPUT_FP)), os.listdir(INPUT_FP)
        file_names_done = {x.split('_')[1] for x in os.listdir(OUTPUT_FP + 'parsed_files')}

        assert output_fp_folders == {'parsed_files', 'exceptions', 'misc'}
        assert len(file_names_done) < len(input_file_names)

        if len(file_names_done) == 0:  # starting from scratch
            stats_dict_master = {}

            for folder in ('exceptions', 'misc'):
                if len(os.listdir(OUTPUT_FP + folder)) > 0:
                    warnings.warn(f'{folder} is not empty---clearing folder...', UserWarning)

                    for fn in os.listdir(OUTPUT_FP + folder):
                        os.remove(f'{OUTPUT_FP}{folder}/{fn}')

            with open(f'{OUTPUT_FP}misc/log.txt', 'w') as f:
                f.write('')
        else:  # recovery (in case of error)
            with open(f'{OUTPUT_FP}misc/parse_stats.json', 'r') as f:
                stats_dict_master = json.load(f)

            assert {x.split('.')[0] for x in file_names_done} == set(stats_dict_master.keys())
            input_file_names = [x for x in input_file_names if x not in file_names_done]

        input_file_names.sort(key=lambda x: int(x.split('.')[0]))
        NUM_WORKERS = min(NUM_WORKERS, len(input_file_names))

        with Manager() as m:
            mngr_comm_list, mngr_status_list = m.list([None] * NUM_WORKERS), m.list([-1] * NUM_WORKERS)
            mngr_stats_dict = m.dict(stats_dict_master)

            worker_arg_temp = (mngr_comm_list, mngr_status_list)
            worker_mngr_args = worker_arg_temp + (input_file_names, mngr_stats_dict)
            worker_args = [(x,) + worker_arg_temp for x in range(NUM_WORKERS)]

            with Pool() as pool:
                pool.map(tokenizer_wrapper_mlm, [worker_mngr_args] + worker_args)

        print_to_both('\n\nDONE!')
else:
    INPUT_FP = GLOBAL_FP + '/raw_data/en_wiki_sents/chosen_sents/'
    OUTPUT_FP = GLOBAL_FP + 'pretraining/data/parsed_graphs/'
    MRS_GRAPH_PARSER_KWARGS = {
        'grammar': GLOBAL_FP + '/ace-0.9.34/erg-1214-x86-64-0.9.34.dat',
        'unk_as_mask': True
    }

    output_fp_folders, input_file_names = set(os.listdir(OUTPUT_FP)), os.listdir(INPUT_FP)
    file_names_done = {x.split('_')[1] for x in os.listdir(OUTPUT_FP + 'parsed_files')}
    init_tok_dicts = [None] * NUM_WORKERS

    assert 'tokenizer_dict' not in MRS_GRAPH_PARSER_KWARGS.keys()
    assert output_fp_folders == {'parsed_files', 'exceptions', 'misc'}
    assert len(file_names_done) < len(input_file_names)

    if len(file_names_done) == 0:  # starting from scratch
        stats_dict_master = {}

        for folder in ('exceptions', 'misc'):
            if len(os.listdir(OUTPUT_FP + folder)) > 0:
                warnings.warn(f'{folder} is not empty---clearing folder...', UserWarning)

                for fn in os.listdir(OUTPUT_FP + folder):
                    os.remove(f'{OUTPUT_FP}{folder}/{fn}')

        with open(f'{OUTPUT_FP}misc/log.txt', 'w') as f:
            f.write('')
    else:  # recovery (in case of error)
        with open(f'{OUTPUT_FP}misc/parse_stats.json', 'r') as f:
            stats_dict_master = json.load(f)

        for fn in os.listdir(OUTPUT_FP + 'misc'):
            if fn not in {'log.txt', 'parse_stats.json'}:
                with open(f'{OUTPUT_FP}misc/{fn}', 'r') as f:
                    init_tok_dicts[int(fn.split('.')[0])] = json.load(f)

        assert {x.split('.')[0] for x in file_names_done} == set(stats_dict_master.keys())
        input_file_names = [x for x in input_file_names if x not in file_names_done]

    input_file_names.sort(key=lambda x: int(x.split('.')[0]))
    NUM_WORKERS = min(NUM_WORKERS, len(input_file_names))

    with Manager() as m:
        mngr_comm_list, mngr_status_list = m.list([None] * NUM_WORKERS), m.list([-1] * NUM_WORKERS)
        mngr_stats_dict = m.dict(stats_dict_master)

        worker_arg_temp = (mngr_comm_list, mngr_status_list)
        worker_mngr_args = worker_arg_temp + (input_file_names, mngr_stats_dict)
        worker_args = [(x, init_tok_dicts[x]) + worker_arg_temp for x in range(NUM_WORKERS)]

        with Pool() as pool:
            pool.map(tokenizer_wrapper_gfolds, [worker_mngr_args] + worker_args)

    print_to_both('\n\nPARSING COMPLETE!\nMERGING TOKENIZER DICTIONARIES...')
    print_to_both('    BUILDING TRANSLATION LISTS:')

    master_td = {k: {a: b for a, b in v.items()} for k, v in DEFAULT_TOKENIZER_DICT.items()}
    translation_lists, worker_files = [None] * NUM_WORKERS, [[] for _ in range(NUM_WORKERS)]
    td_types = ('v', 'e', 'f')

    for fn in os.listdir(OUTPUT_FP + 'parsed_files/'):
        worker_idx, file_idx = fn.split('_')
        worker_files[int(worker_idx)].append(file_idx)

    for worker_idx in range(NUM_WORKERS):
        with open(f'{OUTPUT_FP}misc/{worker_idx}.json', 'r') as f:
            worker_td = json.load(f)

        print_to_both(f'        WORKER {worker_idx}/{NUM_WORKERS - 1}...')
        print()

        translate = {k: [0] * len(worker_td[k].keys()) for k in td_types}

        for td_type in td_types:
            for k, v in worker_td[td_type].items():
                if k in master_td[td_type].keys():
                    translate[td_type][v] = master_td[td_type][k]
                else:
                    translate[td_type][v] = len(master_td[td_type])
                    master_td[td_type].update({k: translate[td_type][v]})

        translation_lists[worker_idx] = translate

    print()
    print_to_both('    TRANSLATING WORKER FILES:')

    with Manager() as m:
        mngr_comm_list = m.list([])
        worker_args = [(i, mngr_comm_list, translation_lists[i], worker_files[i]) for i in range(NUM_WORKERS)]

        with Pool() as pool:
            pool.map(translation_wrapper, [(mngr_comm_list, list(map(len, worker_files)))] + worker_args)

    MRS_GRAPH_PARSER_KWARGS.update({'tokenizer_dict': master_td})

    with open(f'{GLOBAL_FP}/tokenizer_config.json', 'w') as f:
        json.dump(MRS_GRAPH_PARSER_KWARGS, f)

    print_to_both('\n\nMERGING COMPLETE!\nREMOVING UNMERGED FILES...')

    for fn in os.listdir(OUTPUT_FP + 'parsed_files'):  # untranslated files
        if '_' in fn:
            os.remove(f'{OUTPUT_FP}parsed_files/{fn}')

    for fn in os.listdir(OUTPUT_FP + 'misc'):  # individual worker tokenizer dicts
        if fn not in {'log.txt', 'parse_stats.json'}:
            os.remove(f'{OUTPUT_FP}misc/{fn}')

    print_to_both('DONE!')
