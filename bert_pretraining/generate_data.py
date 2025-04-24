
import os
import json
import warnings
import traceback
from datetime import datetime
from global_filepath import GLOBAL_FP
from transformers import AutoTokenizer
from multiprocessing import Pool, Manager


def worker_manager(comm_list, status_list, file_list, stats_dict):
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


def tokenizer_worker(self_idx, comm_list, status_list):
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


def wrapper(args):
    return tokenizer_worker(*args) if isinstance(args[0], int) else worker_manager(*args)


def print_to_log(print_str):
    with open(f'{OUTPUT_FP}misc/log.txt', 'a') as f_log:
        f_log.write('\n' + print_str)


def print_to_both(print_str):
    print_to_log(print_str)
    print(print_str)


NUM_WORKERS = 28
INPUT_FP = GLOBAL_FP + '/raw_data/en_wiki_sents/all_sents/'
OUTPUT_FP = GLOBAL_FP + '/bert/data/mlm/'

os.environ['TOKENIZERS_PARALLELISM'] = 'true'

if __name__ == '__main__':
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
            pool.map(wrapper, [worker_mngr_args] + worker_args)

    print_to_both('\n\nDONE!')
