
import os
import re
import json
import torch
import pickle
from model.io import SWATInput
from subprocess import DEVNULL
from global_filepath import GLOBAL_FP
from model import GRPH_IDX_IDX as IDX
from model.model import SWATransformer
from typing import Iterable, Generator, Optional, Union
from model.graphtools import GraphParser, SWADirGraph, dmrs_to_swa_graph


TOK_FP = GLOBAL_FP + '/tokenizer_config.json'
CHK_FP = GLOBAL_FP + '/pretraining/run_2/checkpoints/ep3_mb199.chk'
FILE_FP = GLOBAL_FP + '/prop_inf/mcrae_concept_sents_all'
FEAT_FP = GLOBAL_FP + '/prop_inf/mcrae_features.json'
OUT_FP = GLOBAL_FP + '/prop_inf/mcrae_data_parsable.pkl'


class PropInfGraphParser(GraphParser):
    def __init__(self, grammar, tokenizer_dict, unk_as_mask=True, _fpt=False):
        super(PropInfGraphParser, self).__init__(_fpt=_fpt)
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
            include_carg: bool = False
    ) -> Generator[Optional[Union[SWADirGraph, dict]], None, None]:
        for line in map(lambda s: s.strip().lower(), input_sentences):
            if len(line) == 0:
                continue

            term, sent = map(lambda x: x.strip(), line.split(':'))
            sent = sent[0].upper() + sent[1:]
            sent = sent[:-1] if sent[-1] == '.' else sent

            if '|' in term:
                term, current_term = map(lambda z: z.strip(), term.split('|'))
            elif '_' in term:
                current_term = term.split('_')[0]
            else:
                current_term = term

            yield {
                **self._parse_template(
                    sent,
                    print_failure,
                    current_term,
                    include_unk,
                    include_carg
                ),
                **{'term': term}
            }

    def _parse_template(
            self,
            mrs_template,
            print_failure,
            current_term,
            include_unk,
            include_carg
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

        return {'sent': mrs_template, 'trgt_idx': trgt_idx, 'g': dmrs_graph}

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
    def from_pretrained(cls, filepath, **kwargs) -> "PropInfGraphParser":
        with open(os.path.abspath(filepath), 'r') as f:
            fp_kwargs = {**json.load(f), **kwargs, **{'_fpt': True}}

        return cls(fp_kwargs.pop('grammar'), fp_kwargs.pop('tokenizer_dict'), **fp_kwargs)


with open(FEAT_FP, 'r') as f:
    feat_file = json.load(f)

model = SWATransformer.from_swat_module(CHK_FP)
model.to(device=0)
model.eval()

parser = PropInfGraphParser.from_pretrained(TOK_FP)
out_file, failures = [], {}

with open(FILE_FP, 'r') as f_in:
    mcrae_lines = f_in.readlines()

for grph in parser(mcrae_lines, print_failure=True):
    if 'failure' in grph.keys():
        failure_type = grph['failure']['type']

        if failure_type in failures.keys():
            failures[failure_type] += 1
        else:
            failures.update({failure_type: 1})
    else:
        with torch.no_grad():
            model_input = SWATInput.from_dir_graph(grph['g'])
            model_output = model(model_input).embeddings
            out_file.append({
                'term': grph['term'],
                'features': {int(k): v for k, v in feat_file[grph['term']].items()},
                'embedding': model_output[..., grph['g']['n'][grph['trgt_idx']][IDX], :].flatten().to(device='cpu')
            })

            del model_input, model_output

with open(OUT_FP, 'wb') as f_out:
    pickle.dump(out_file, f_out)

total_f, total_v = sum(failures.values()), len(out_file)
total_exs = total_f + total_v

print(f'PARSED: {total_v} / {total_exs} ({round(total_v * 100 / total_exs, 3)}%)')
print(f'FAILED: {total_f} / {total_exs} ({round(total_f * 100 / total_exs, 3)}%)')

for f_str, f_cnt in sorted(list(failures.items()), key=lambda z: z[1]):
    print(f'   {f_str}: {f_cnt}')
