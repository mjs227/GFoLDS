
import json
from global_filepath import GLOBAL_FP
from transformers import AutoTokenizer


def inflect_term(term):
    word, pos = term.split('_')

    if pos == 'N':
        return ('an' if word[0] in VOWELS else 'a') + ' ' + word

    return word + 's'


INCL_MULTI_TOK = True
VOWELS = {'a', 'i', 'o', 'u'}
tokenizer = AutoTokenizer.from_pretrained('bert-base-uncased')

for split in ('dev', 'test'):
    with open(f'{GLOBAL_FP}/relpron/relpron.{split}', 'r') as f:
        split_lines = f.readlines()

    out_file = []

    for line in map(lambda z: z.strip(), split_lines):
        if len(line) > 0:
            _, trgt, hyper, _, w1, w2 = map(lambda x: x.strip(), line.split(' '))
            trgt_art, trgt_w = inflect_term(trgt.replace(':', '')).split(' ')
            trgt_tok = tokenizer(trgt_w)['input_ids'][1:-1]

            mask_str = '[MASK]' * len(trgt_tok)
            sent = f'{inflect_term(hyper)} that {inflect_term(w1)} {inflect_term(w2)} is {trgt_art} {mask_str}.'
            toks = tokenizer(sent)['input_ids']

            if INCL_MULTI_TOK or len(trgt_tok) == 1:
                out_file.append({
                    'bert': {
                        'sent': sent,
                        'trgt_tok': trgt_tok,
                        'trgt_idx': list(range(len(toks) - (2 + len(trgt_tok)), len(toks) - 2)),
                        's': toks
                    }
                })

    with open(f'{GLOBAL_FP}/relpron/relpron_exs_bert_{split}.json', 'w') as f:
        json.dump(out_file, f)
