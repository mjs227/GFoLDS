
import re
import os
import gc
import json
import spacy
import random
import traceback
from tqdm import tqdm
from itertools import chain
from setup import GLOBAL_FP
from datasets import load_dataset
from argparse import ArgumentParser
from multiprocessing import Pool, Manager


def _sent_check(in_sent):
    if len(in_sent) == 0:
        return (None,)

    re_match = re.search(r'[^A-Z](\.|\?|\!)[A-Z]', in_sent)  # n>1 sents conjoined

    if re_match is None:  # catching bad strings... (goal: high prec., so-so recall)
        if not (lambda z: z.isalnum() and (z.isupper() or z in DIGITS))(in_sent[0]):  # non-alphanumeric (capital) start char
            return (None,)
        if re.match(r'^[^?!]+(\.|\?|\!)(")?$', in_sent) is None:  # EOS is not punct (optional closing ")
            return (None,)
        if sum(1 for char in in_sent if char == '"') % 2 == 1:  # unclosed quotes
            return (None,)

        return (in_sent,)

    splt = re_match.span()[0] + 2

    return chain.from_iterable(_sent_check(x) for x in (in_sent[:splt], in_sent[splt:]))


def sent_check(in_sent):
    for y in _sent_check(in_sent):
        if y is not None:
            yield y


def worker(self_idx, comm_list):
    with open(f'{RAW_FP}{self_idx}', 'r') as f_self:
        self_articles = f_self.readlines()

    comm_list[self_idx] = sum(1 for x in self_articles if x.strip() == ARTICLE_DELIM)

    while not comm_list[self_idx] == 0:
        pass

    senter = spacy.load('en_core_web_sm', exclude=['parser'])
    senter.enable_pipe('senter')
    save_cnt, batch_txt, skip = 0, '', True

    for line in map(lambda z: z.strip(), self_articles):
        if skip:
            skip = not line == ARTICLE_DELIM
        elif line.lower() in END_SECTIONS:
            skip = True
            save_cnt += 1
        elif not all(x not in line for x in S_PUNCT):  # should catch empty lines + section headers
            try:
                para_sents = senter(line)
                batch_txt += '\n'.join(chain.from_iterable(sent_check(*sent.text.strip()) for sent in para_sents.sents))
            except Exception as e:
                with open(f'{ALL_FP}exceptions/{self_idx}', 'a') as f_err:
                    f_err.write('\n\n\n--------------------------------------------\n\n')
                    f_err.write(f'{type(e)}' + '\n'.join(traceback.format_tb(e.__traceback__)))

        if save_cnt % SAVE_INTERVAL == 0:
            with open(f'{ALL_FP}/{self_idx}', 'a') as f_self:
                f_self.write(batch_txt)

            comm_list[self_idx], batch_txt = save_cnt, ''

    with open(f'{ALL_FP}/{self_idx}', 'a') as f_self:
        f_self.write(batch_txt + '\n')

    comm_list[self_idx], batch_txt = None, ''


def manager(comm_list, stats_file):
    run = True
    worker_lens = [0 for _ in range(len(comm_list))]

    while not all(not x == -1 for x in comm_list):
        pass

    print('\n\nBEGINNING MAIN LOOP\n\n')

    for i_ in range(len(comm_list)):
        worker_lens[i_] = comm_list[i_]
        comm_list[i_] = 0

    while run:
        iter_updates = {}

        for i_, w_l in enumerate(worker_lens):
            comm_i = (lambda w: w_l if w is None else w)(comm_list[i_])
            run = run and w_l == comm_i

            if not comm_i == stats_file[i_][0]:
                iter_updates.update({i_: [comm_i, round(comm_i / w_l, 3)]})

        if len(iter_updates) > 0:
            stats_file.update(iter_updates)

            with open(ALL_FP + 'worker_stats.json', 'w') as f_stat:
                json.dump(stats_file, f_stat)

            print('; '.join(f'{i_}:{stats_file[i_][1]}' for i_ in range(len(worker_lens))))

        run = not run

    return None


def wrapper(a):
    return worker(*a) if isinstance(a[0], int) else manager(*a)


def print_tqdm(print_str):
    print()
    print(print_str)
    print()


ap = ArgumentParser()
ap.add_argument('-n', '--num_workers', help='Number of sentence-splitting threads', type=int, required=True)
ap.add_argument('-f', '--num_files', help='Number of output files to split sentence data into', type=int, default=200)
ap.add_argument(
    '-k', '--k_sent', help='Number of sentences to keep (for GFoLDS & BERT MLM pretraining)', type=int, default=17460320
)
args = ap.parse_args()
N_OUTPUT_FILES = args.num_files
N_WORKERS = args.num_workers
N_CHOSEN = args.k_sent

RAW_FP = f'{GLOBAL_FP}/raw_data/en_wiki_raw/'
ALL_FP = f'{GLOBAL_FP}/raw_data/en_wiki_sents/all_sents/'
CHOSEN_FP = f'{GLOBAL_FP}/raw_data/en_wiki_sents/chosen_sents/'
assert len(os.listdir(CHOSEN_FP)) == 0

ARTICLE_DELIM = '\n[<_*AT*_>]:==\n'
END_SECTIONS = {'see also', 'references', 'sources', 'bibliography', 'external links', 'further reading', 'notes'}
S_PUNCT = ('?', '!', '.')
SAVE_INTERVAL = 1000  # workers save every SAVE_INTERVAL articles
AD_LEN = len(ARTICLE_DELIM)
DIGITS = set(map(str, range(10)))

wiki_ds = load_dataset('wikimedia/wikipedia', '20231101.en')['train']
worker_files, i = [[] for _ in range(N_WORKERS)], 0

for i, art in tqdm(enumerate(wiki_ds)):
    worker_files[i % N_WORKERS].append(art['text'])

for i, files in tqdm(enumerate(worker_files)):
    with open(f'{RAW_FP}{i}', 'w') as f:
        for art in files:
            f.write(ARTICLE_DELIM)
            f.write(art)

worker_files.clear()
del wiki_ds
gc.collect()
worker_stats_file = {i: [0, 0.0] for i in range(N_WORKERS)}

for i in range(N_WORKERS):
    with open(ALL_FP + str(i), 'w') as f:
        f.write('')

    with open(f'{ALL_FP}exceptions/{i}', 'w') as f:
        f.write('')

with Manager() as m:
    main_comm_list = m.list([-1] * N_WORKERS)
    worker_args = [
        (idx, main_comm_list) for idx in range(N_WORKERS)
    ]
    manager_args = (main_comm_list, worker_stats_file)

    with Pool() as pool:
        pool.map(wrapper, [manager_args] + worker_args)

gc.collect()

sent_idxs, file_sents, src_files = [], [], os.listdir(ALL_FP)
file_idxs = [set() for _ in range(len(src_files))]
print_tqdm('COLLECTING SENTENCE INDICES')

for fn in tqdm(src_files):
    f_idx = int(fn)

    with open(ALL_FP + fn, 'r') as f:
        sent_idxs.extend((f_idx, s_idx) for s_idx, line in enumerate(map(lambda x: x.strip(), f)) if len(line) > 0)

print_tqdm(f'CHOOSING {N_CHOSEN} / {len(sent_idxs)} SENTENCES ({round((N_CHOSEN / len(sent_idxs)) * 100, 3)}%)')
random.shuffle(sent_idxs)

for f_idx, s_idx in tqdm(sent_idxs[:N_CHOSEN]):
    file_idxs[f_idx].add(s_idx)

print_tqdm('COLLECTING CHOSEN SENTENCES')

for fn in tqdm(src_files):
    fn_idxs = file_idxs[int(fn)]

    with open(ALL_FP + fn, 'r') as f:
        file_sents.extend(sent for i, sent in enumerate(map(lambda x: x.strip(), f)) if i in fn_idxs)

print_tqdm('WRITING CHOSEN SENTENCES TO ' + CHOSEN_FP)
random.shuffle(file_sents)
batch_cnt, batch_sizes = 0, [len(file_sents) // N_OUTPUT_FILES] * N_OUTPUT_FILES

for i in range(len(file_sents) - sum(batch_sizes)):  # remainder
    batch_sizes[i] += 1

for batch_i in tqdm(range(N_OUTPUT_FILES)):
    with open(f'{CHOSEN_FP}{batch_i}', 'w') as f:
        for sent in file_sents[batch_cnt:batch_cnt + batch_sizes[batch_i]]:
            f.write(sent + '\n')

        batch_cnt += batch_sizes[batch_i]
