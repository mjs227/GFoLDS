
import os
import json
from multiprocessing import Pool
from global_filepath import GLOBAL_FP
from transformers import AutoTokenizer


def tokenizer_worker(self_files):
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


NUM_WORKERS = 28
INPUT_FP = GLOBAL_FP + '/raw_data/en_wiki_sents/all_sents/'
OUTPUT_FP = GLOBAL_FP + '/bert/data/nsp/'
os.environ['TOKENIZERS_PARALLELISM'] = 'true'

if __name__ == '__main__':
    input_file_names, worker_files = set(os.listdir(INPUT_FP)), [[] for _ in range(NUM_WORKERS)]

    for i, fn in enumerate(input_file_names):
        worker_files[i % NUM_WORKERS].append(fn)

    with Pool() as pool:
        pool.map(tokenizer_worker, worker_files)

    print('\n\nDONE!')
