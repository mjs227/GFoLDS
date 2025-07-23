
import os
from argparse import ArgumentParser


GLOBAL_FP = ''
_local_fp = __file__.rsplit("/", 1)[0]


if __name__ == '__main__':
    ap = ArgumentParser()
    ap.add_argument('-g', '--global_filepath', help='Main directory', type=str, required=True)
    args = ap.parse_args()

    GLOBAL_FP = os.path.abspath(args.global_filepath).rstrip('/')
    assert not (os.path.isdir(GLOBAL_FP) or 'global_filepath' in os.listdir(f'{_local_fp}'))

    os.mkdir(GLOBAL_FP)
    os.mkdir(f'{GLOBAL_FP}/prop_inf')
    os.mkdir(f'{GLOBAL_FP}/relpron')

    os.mkdir(f'{GLOBAL_FP}/raw_data')
    os.mkdir(f'{GLOBAL_FP}/raw_data/en_wiki_raw')
    os.mkdir(f'{GLOBAL_FP}/raw_data/en_wiki_sents')
    os.mkdir(f'{GLOBAL_FP}/raw_data/en_wiki_sents/all_sents')
    os.mkdir(f'{GLOBAL_FP}/raw_data/en_wiki_sents/all_sents/exceptions')
    os.mkdir(f'{GLOBAL_FP}/raw_data/en_wiki_sents/chosen_sents')

    os.mkdir(f'{GLOBAL_FP}/bert')
    os.mkdir(f'{GLOBAL_FP}/bert/run_0')
    os.mkdir(f'{GLOBAL_FP}/bert/run_0/stats')
    os.mkdir(f'{GLOBAL_FP}/bert/run_0/optimizers')
    os.mkdir(f'{GLOBAL_FP}/bert/run_0/checkpoints')
    os.mkdir(f'{GLOBAL_FP}/bert/data')

    for pt_type in ('mlm', 'nsp'):
        os.mkdir(f'{GLOBAL_FP}/bert/data/{pt_type}')
        os.mkdir(f'{GLOBAL_FP}/bert/data/{pt_type}/batch_data')
        os.mkdir(f'{GLOBAL_FP}/bert/data/{pt_type}/preproc_data')

    os.mkdir(f'{GLOBAL_FP}/bert/data/mlm/preproc_data/misc')
    os.mkdir(f'{GLOBAL_FP}/bert/data/mlm/preproc_data/exceptions')
    os.mkdir(f'{GLOBAL_FP}/bert/data/mlm/preproc_data/parsed_files')

    os.mkdir(f'{GLOBAL_FP}/elem_eval')
    os.mkdir(f'{GLOBAL_FP}/elem_eval/stats')
    os.mkdir(f'{GLOBAL_FP}/elem_eval/stats/gfolds')
    os.mkdir(f'{GLOBAL_FP}/elem_eval/stats/bert-base-uncased')
    os.mkdir(f'{GLOBAL_FP}/elem_eval/stats/bert-large-uncased')

    os.mkdir(f'{GLOBAL_FP}/nli')
    os.mkdir(f'{GLOBAL_FP}/nli/runs')
    os.mkdir(f'{GLOBAL_FP}/nli/data')
    os.mkdir(f'{GLOBAL_FP}/nli/data/dev')
    os.mkdir(f'{GLOBAL_FP}/nli/data/test')
    os.mkdir(f'{GLOBAL_FP}/nli/data/train')

    for model in ('gfolds', 'bert_large_comparison', 'bert_base_comparison', 'bert_large', 'bert_base'):
        os.mkdir(f'{GLOBAL_FP}/nli/runs/{model}')
        os.mkdir(f'{GLOBAL_FP}/nli/runs/{model}/stats')
        os.mkdir(f'{GLOBAL_FP}/nli/runs/{model}/optimizers')
        os.mkdir(f'{GLOBAL_FP}/nli/runs/{model}/checkpoints')

    os.mkdir(f'{GLOBAL_FP}/double_negation')
    os.mkdir(f'{GLOBAL_FP}/double_negation/data')
    os.mkdir(f'{GLOBAL_FP}/double_negation/stats')
    os.mkdir(f'{GLOBAL_FP}/double_negation/checkpoints')

    os.mkdir(f'{GLOBAL_FP}/factuality')
    os.mkdir(f'{GLOBAL_FP}/factuality/data')
    os.mkdir(f'{GLOBAL_FP}/factuality/results')

    os.mkdir(f'{GLOBAL_FP}/pretraining')
    os.mkdir(f'{GLOBAL_FP}/pretraining/data')
    os.mkdir(f'{GLOBAL_FP}/pretraining/data/batch_data')
    os.mkdir(f'{GLOBAL_FP}/pretraining/data/parsed_graphs')
    os.mkdir(f'{GLOBAL_FP}/pretraining/data/parsed_graphs/misc')
    os.mkdir(f'{GLOBAL_FP}/pretraining/data/parsed_graphs/exceptions')
    os.mkdir(f'{GLOBAL_FP}/pretraining/data/parsed_graphs/parsed_files')

    os.mkdir(f'{GLOBAL_FP}/pretraining/run_0')
    os.mkdir(f'{GLOBAL_FP}/pretraining/run_0/stats')
    os.mkdir(f'{GLOBAL_FP}/pretraining/run_0/optimizers')
    os.mkdir(f'{GLOBAL_FP}/pretraining/run_0/checkpoints')

    os.rename(f'{_local_fp}/data/qp_raw', f'{GLOBAL_FP}/elem_eval/qp_raw')
    os.rename(f'{_local_fp}/data/relpron_clean', f'{GLOBAL_FP}/relpron/relpron_clean')
    os.rename(f'{_local_fp}/data/mcrae_fold_words', f'{GLOBAL_FP}/prop_inf/mcrae_fold_words')
    os.rename(f'{_local_fp}/data/mcrae_features.json', f'{GLOBAL_FP}/prop_inf/mcrae_features.json')
    os.rename(f'{_local_fp}/data/mcrae_concept_sents_all', f'{GLOBAL_FP}/prop_inf/mcrae_concept_sents_all')
    os.rmdir(f'{_local_fp}/data')

    with open(f'{_local_fp}/global_filepath', 'w') as f:
        f.write(GLOBAL_FP)
else:
    with open(f'{_local_fp}/global_filepath', 'r') as f:
        GLOBAL_FP = (''.join(f.readlines())).strip()
