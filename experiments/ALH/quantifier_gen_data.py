
import os
import re
import json
import random
from subprocess import DEVNULL
from global_filepath import GLOBAL_FP
from transformers import AutoTokenizer
from model.graphtools import GraphParser, SWADirGraph
from model import PROHIBITED_TOKENS, LINK_ROLE_MAP, GRPH_IDX_IDX, GRPH_LABEL_IDX


SRC_FP = GLOBAL_FP + '/elem_eval/qp_raw'
TRGT_FP = GLOBAL_FP + '/elem_eval/quantifier_probe_data.json'
TOK_FP = GLOBAL_FP + '/tokenizer_config.json'
BERT_TKNZR = 'bert-base-uncased'

SINGULAR = {
    'another_q': 'another',
    'either_q': 'either',
    'neither_q': 'neither',
    'that_q_dem': 'that',
    'this_q_dem': 'this',
    'every_q': 'every',
    'a_q': 'a',
    'each_q': 'each',
    'some_q_indiv': 'some'
}
PLURAL = {
    'these_q_dem': 'these',
    'certain_q': 'certain',
    'most_q': 'most',
    'those_q_dem': 'those',
    'all_q': 'all',
    'some_q': 'some',
    'such_q': 'such',
    'both_q': 'both'
}
BOTH = {
    'the_q': 'the',
    'any_q': 'any',
    'enough_q': 'enough',
    'no_q': 'no',
    'which_q': 'which'
}
ALL_QUANTS = {**SINGULAR, **PLURAL, **BOTH}


class QPGraphParser(GraphParser):
    def __init__(self, grammar, tokenizer_dict, unk_as_mask=True, _fpt=False):
        super(QPGraphParser, self).__init__(_fpt=_fpt)
        from delphin import ace, dmrs

        self.grammar, self.tokenizer_dict = os.path.abspath(grammar), tokenizer_dict
        ace_fp = self.grammar[:-len(self.grammar.split('/')[-1])]

        if re.search(re.compile(f'(^|:){re.escape(ace_fp)}($|:)'), os.environ['PATH'].strip()) is None:
            os.environ['PATH'] += ('' if len(os.environ['PATH'].strip()) == 0 else ':') + ace_fp

        self.mrs_parser, self._to_dmrs = ace, dmrs.from_mrs
        self._unk_toks, self.unk_as_mask = {}, unk_as_mask
        assert set(tokenizer_dict.keys()) == {'v', 'e', 'f'}

        for k in ('v', 'e', 'f'):
            if unk_as_mask or '[UNK]' not in self.tokenizer_dict[k].keys():
                self._unk_toks.update({k: self.tokenizer_dict[k].get('[MASK]', -1)})
            else:
                self._unk_toks.update({k: self.tokenizer_dict[k]['[UNK]']})

        self._num_sg = self.tokenizer_dict['f']['[NUM:sg]']
        self._num_pl = self.tokenizer_dict['f']['[NUM:pl]']

    def save(self, *args, **kwargs):
        raise NotImplementedError

    def _call(
            self,
            input_sentences: str,
            print_failure: bool = False,
            include_unk: bool = False,
            include_carg: bool = False,
            bert_tokenizer=None
    ):
        assert bert_tokenizer is not None
        bert_mask_id = bert_tokenizer('[MASK]')['input_ids'][1]

        for line in map(lambda s: s.strip(), input_sentences):
            if len(line) > 0:
                yield self._parse_template(
                    line,
                    print_failure,
                    include_unk,
                    include_carg,
                    bert_tokenizer,
                    bert_mask_id
                )

    def _parse_template(
            self,
            in_sent,
            print_failure,
            include_unk,
            include_carg,
            bert_tokenizer,
            bert_mask_id
    ):
        response = next(self.mrs_parser.parse_from_iterable(self.grammar, (in_sent,), cmdargs=['-1'], stderr=DEVNULL))

        if len(response['results']) == 0:
            return self._record_failure(in_sent, 'DMRS parse failure', print_failure)

        dmrs = self._to_dmrs(response.result(0).mrs())
        graph, pred_id_dict = SWADirGraph(), {}
        potential_quants, both_quants = [], {}

        for pred in dmrs.predications:
            if pred.carg is not None and not include_carg:
                return self._record_failure(in_sent, 'carg detected', print_failure)

            pred_name = pred.predicate.lower()
            pred_name = pred_name[1:] if pred_name[0] == '_' else pred_name

            if pred_name not in PROHIBITED_TOKENS:  # {'focus_d', 'parg_d'}
                if '/' in pred_name:  # OOV
                    pred_id = graph.add_node(self._unk_toks['v'])
                else:
                    pred_id = graph.add_node(self._tokenize_fn(pred_name, 'v'))

                for feat, val in pred.properties.items():
                    graph.add_feature(pred_id, self._tokenize_fn(f'[{feat}:{val}]', 'f'))

                pred_id_dict.update({pred.id: pred_id})

                if graph.nodes[pred_id][GRPH_LABEL_IDX] == self._unk_toks['v'] and not include_unk:
                    return self._record_failure(in_sent, 'UNK tok detected', print_failure)

                if pred_name in ALL_QUANTS.keys():
                    if pred_name in SINGULAR.keys():
                        potential_quants.append((pred, 's'))
                    elif pred_name in PLURAL.keys():
                        potential_quants.append((pred, 'p'))
                    else:  # pred_name in BOTH.keys():
                        both_quants.update({pred.id: pred})

        for link in dmrs.links:
            if link.start in pred_id_dict.keys() and link.end in pred_id_dict.keys():
                graph.add_edge(
                    pred_id_dict[link.start],
                    pred_id_dict[link.end],
                    self._tokenize_fn(LINK_ROLE_MAP.get(link.role, link.role), 'e')
                )

                if link.role == 'RSTR' and link.start in both_quants.keys():
                    link_trgt, bq_pred = pred_id_dict[link.end], both_quants.pop(link.start)

                    if self._num_sg in graph.features[link_trgt]:
                        potential_quants.append((bq_pred, 's'))
                    elif self._num_pl in graph.features[link_trgt]:
                        potential_quants.append((bq_pred, 'p'))

        if len(potential_quants) == 0:
            return self._record_failure(in_sent, 'quantifiers not identified in DMRS', print_failure)

        random.shuffle(potential_quants)
        graph._assign_idxs()
        q_pred, q_type = None, None

        for q_pred_, q_type_ in potential_quants:
            q_pred_name = q_pred_.predicate.lower()
            q_pred_name = q_pred_name[1:] if q_pred_name[0] == '_' else q_pred_name

            if (q_pred_name == 'a_q' and in_sent[q_pred_.cfrom:q_pred_.cto].lower() in {'a', 'an'}) \
                    or in_sent[q_pred_.cfrom:q_pred_.cto].lower() == ALL_QUANTS[q_pred_name]:
                q_pred, q_type = q_pred_, q_type_
                break

        if q_pred is None:
            return self._record_failure(in_sent, 'quantifier(s) not identified in BERT input', print_failure)

        bert_toks = bert_tokenizer([in_sent[:q_pred.cfrom] + '[MASK]' + in_sent[q_pred.cto:]])['input_ids']

        return {
            'sent': in_sent,
            'type': q_type,
            'gfolds': {
                'idx': graph.nodes[pred_id_dict[q_pred.id]][GRPH_IDX_IDX],
                'g': graph.save()
            },
            'bert': {
                'idx': next(i for i, x in enumerate(bert_toks[0]) if x == bert_mask_id),
                's': bert_toks
            }
        }

    def _record_failure(self, mrs_template, fail_type, print_failure):
        if print_failure:
            print(f'{fail_type}: {mrs_template}')

        return {'failure': {'sent': mrs_template, 'type': fail_type}}

    def _tokenize_fn(self, s, kind):
        return self.tokenizer_dict[kind].get(s, self._unk_toks[kind])

    @classmethod
    def init_for_pretraining(cls, *args, **kwargs) -> None:
        raise NotImplementedError

    @classmethod
    def from_pretrained(cls, filepath, **kwargs) -> "QPGraphParser":
        with open(os.path.abspath(filepath), 'r') as f:
            fp_kwargs = {**json.load(f), **kwargs, **{'_fpt': True}}

        return cls(fp_kwargs.pop('grammar'), fp_kwargs.pop('tokenizer_dict'), **fp_kwargs)


if __name__ == '__main__':
    with open(SRC_FP, 'r') as f_src:
        src_file = f_src.readlines()

    parser = QPGraphParser.from_pretrained(TOK_FP)
    failures, valid_exs = {}, []
    parsed = parser(
        src_file,
        print_failure=True,
        include_unk=True,
        include_carg=True,
        bert_tokenizer=AutoTokenizer.from_pretrained(BERT_TKNZR)
    )

    for grph in parsed:
        if 'failure' in grph.keys():
            failure_type = grph['failure']['type']

            if failure_type in failures.keys():
                failures[failure_type] += 1
            else:
                failures.update({failure_type: 1})
        else:
            valid_exs.append(grph)

    with open(TRGT_FP, 'w') as f_out:
        json.dump(valid_exs, f_out)

    total_f, total_v = sum(failures.values()), len(valid_exs)
    total_exs = total_f + total_v

    print(f'\n\n\nPARSED: {total_v} / {total_exs} ({round(total_v * 100 / total_exs, 3)}%)')
    print(f'FAILED: {total_f} / {total_exs} ({round(total_f * 100 / total_exs, 3)}%)')

    for f_str, f_cnt in sorted(list(failures.items()), key=lambda z: z[1]):
        print(f'   {f_str}: {f_cnt}')
