
import os
import re
import json
from setup import GLOBAL_FP
from subprocess import DEVNULL
from transformers import AutoTokenizer
from typing import Iterable, Generator, Optional, Union
from model.graphtools import GraphParser, SWADirGraph, dmrs_to_swa_graph


class RELPRONGraphParser(GraphParser):
    def __init__(self, grammar, tokenizer_dict, unk_as_mask=True, _fpt=False):
        super(RELPRONGraphParser, self).__init__(_fpt=_fpt)
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

    def save(self, *args, **kwargs):
        raise NotImplementedError

    def _call(
            self,
            input_sentences: Iterable[str],
            print_failure: bool = False,
            include_unk: bool = False,
            include_carg: bool = False,
            include_multi_bert_toks: bool = False,
            bert_sep_tok: int = 102,
            bert_mask_tok: int = 103,
            bert_tokenizer=None
    ) -> Generator[Optional[Union[SWADirGraph, dict]], None, None]:
        assert include_multi_bert_toks or bert_tokenizer is not None
        current_term, hyper, term_article = '', '', ''

        for line in map(lambda s: s.strip().lower(), input_sentences):
            if len(line) == 0:
                continue
            if ':' in line:
                current_term, hyper = line.split(':')
                hyper = hyper.strip()

                if '(' in current_term:
                    current_term = current_term.replace('(', '').replace(')', '')
                    term_article, current_term = current_term.split(' ')
                    term_article = term_article + ' '
                else:
                    term_article = ''

                continue

            mrs_template = f'{hyper} that {line[1:].strip()} is {term_article}{current_term}.'

            yield self._parse_template(
                mrs_template[0].upper() + mrs_template[1:],
                print_failure,
                current_term,
                include_unk,
                include_carg,
                include_multi_bert_toks,
                bert_sep_tok,
                bert_mask_tok,
                bert_tokenizer
            )

    def _parse_template(
            self,
            mrs_template,
            print_failure,
            current_term,
            include_unk,
            include_carg,
            include_multi_bert_toks,
            bert_sep_tok,
            bert_mask_tok,
            bert_tokenizer
    ):
        response = next(self.mrs_parser.parse_from_iterable(
            self.grammar,
            (mrs_template,),
            cmdargs=['-1'],
            stderr=DEVNULL
        ))

        if len(response['results']) == 0:
            return self._record_failure(mrs_template, 'DMRS parse failure', print_failure)

        dmrs = self._to_dmrs(response.result(0).mrs())
        trgt_tok, trgt_lbl = None, ''

        for rel in dmrs.predications:
            if rel.carg is not None and not include_carg:
                return self._record_failure(mrs_template, 'carg detected', print_failure)

            if (mrs_template[rel.cfrom:rel.cto].replace('.', '').lower() == current_term and
                    re.match(r'^(_)?[a-zA-Z]+_[a-zA-Z\d]+(_)?[a-zA-Z\d]*', rel.predicate) and
                    len(trgt_lbl) == 0 and not (lambda p: '_q_' in p or p.endswith('_q'))(rel.predicate)):
                trgt_lbl = rel.predicate

                if trgt_lbl[0] == '_':  # MRSGraphParser does this when tokenizing...
                    trgt_lbl = trgt_lbl[1:]

                trgt_tok = self.tokenizer_dict['v'].get(trgt_lbl, None)

        if trgt_tok is None:
            return self._record_failure(mrs_template, 'target node search failure', print_failure)

        dmrs_graph, trgt_idx = dmrs_to_swa_graph(self, dmrs).save(), -1

        for idx, (node_label, _, _) in dmrs_graph['n'].items():
            if node_label == self._unk_toks['v'] and not include_unk:
                return self._record_failure(mrs_template, '(non-target) UNK tok detected', print_failure)
            elif node_label == trgt_tok:
                if trgt_idx == -1:
                    trgt_idx = idx
                else:  # trgt occurs more than once
                    return self._record_failure(mrs_template, 'unknown error', print_failure)

        if trgt_idx == -1:
            return self._record_failure(mrs_template, 'target idx search failure', print_failure)

        out_dict = {
            'sent': mrs_template,
            'gfolds': {
                'trgt_lbl': trgt_lbl,
                'trgt_tok': trgt_tok,
                'trgt_idx': trgt_idx,
                'g': dmrs_graph
            }
        }

        if bert_tokenizer is None:
            return out_dict

        bert_toks = bert_tokenizer(mrs_template.replace(current_term, f'[SEP]{current_term}[SEP]'))['input_ids']
        bert_span = tuple(i for i in range(1, len(bert_toks) - 1) if bert_toks[i] == bert_sep_tok)

        if len(bert_span) == 2:
            start, end = bert_span
        else:
            return self._record_failure(mrs_template, 'BERT span detection failure', print_failure)

        bert_start, bert_end = bert_toks[:start], bert_toks[end + 1:]
        bert_trgt = bert_toks[start + 1:end]

        if not (len(bert_trgt) == 1 or include_multi_bert_toks):
            return self._record_failure(mrs_template, 'BERT toks > 1', print_failure)

        out_dict.update({
            'bert': {
                'trgt_tok': bert_trgt,
                'trgt_idx': list(range(start + 1, end)),
                's': bert_start + ([bert_mask_tok] * len(bert_trgt)) + bert_end,
            }
        })

        return out_dict

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
    def from_pretrained(cls, filepath, **kwargs) -> "RELPRONGraphParser":
        with open(os.path.abspath(filepath), 'r') as f:
            fp_kwargs = {**json.load(f), **kwargs, **{'_fpt': True}}

        return cls(fp_kwargs.pop('grammar'), fp_kwargs.pop('tokenizer_dict'), **fp_kwargs)


parser = RELPRONGraphParser.from_pretrained(f'{GLOBAL_FP}/tokenizer_config.json')
tokenizer = AutoTokenizer.from_pretrained('bert-large-uncased')

with open(f'{GLOBAL_FP}/relpron/relpron_clean', 'r') as f_relpron:
    relpron_lines = f_relpron.readlines()

relpron_dev = relpron_lines[:647]
relpron_test = relpron_lines[647:]

for (relpron_split, split_name) in ((relpron_dev, 'dev'), (relpron_test, 'test')):
    failures, valid_exs = {}, []

    for grph in parser(relpron_split, print_failure=True, bert_tokenizer=tokenizer):
        if 'failure' in grph.keys():
            failure_type = grph['failure']['type']

            if failure_type in failures.keys():
                failures[failure_type] += 1
            else:
                failures.update({failure_type: 1})
        else:
            valid_exs.append(grph)

    with open(f'{GLOBAL_FP}/relpron/relpron_exs_{split_name}.json', 'w') as f_out:
        json.dump(valid_exs, f_out)

    total_f, total_v = sum(failures.values()), len(valid_exs)
    total_exs = total_f + total_v

    print(f'\n\n\nSPLIT: {split_name.upper()}')
    print(f'PARSED: {total_v} / {total_exs} ({round(total_v * 100 / total_exs, 3)}%)')
    print(f'FAILED: {total_f} / {total_exs} ({round(total_f * 100 / total_exs, 3)}%)')

    for f_str, f_cnt in sorted(list(failures.items()), key=lambda z: z[1]):
        print(f'   {f_str}: {f_cnt}')

    print('\n\n\n')
