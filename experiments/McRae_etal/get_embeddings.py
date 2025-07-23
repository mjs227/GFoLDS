
import re
import os
import json
import torch
import pickle
from setup import GLOBAL_FP
from model.io import SWATInput
from subprocess import DEVNULL
from argparse import ArgumentParser
from collections import OrderedDict
from model import GRPH_IDX_IDX as IDX
from model.model import SWATransformer
from typing import Iterable, Generator, Optional, Union
from transformers import AutoModelForMaskedLM, AutoTokenizer
from model.graphtools import GraphParser, SWADirGraph, dmrs_to_swa_graph


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
        with open(os.path.abspath(filepath), 'r') as f_tok:
            fp_kwargs = {**json.load(f_tok), **kwargs, **{'_fpt': True}}

        return cls(fp_kwargs.pop('grammar'), fp_kwargs.pop('tokenizer_dict'), **fp_kwargs)



def pt_sd_to_mlm(in_sd, bert_mlm):
    mlm_sd_keys, in_sd_keys = map(set, (bert_mlm.state_dict().keys(), in_sd.keys()))

    if in_sd_keys == mlm_sd_keys:
        return in_sd

    if mlm_sd_keys <= in_sd_keys:
        pfx = 0
    else:
        assert mlm_sd_keys <= {k[5:] for k in in_sd_keys}
        pfx = 5

    new_sd = OrderedDict()

    for k_, v_ in in_sd.items():
        if k_[pfx:] in mlm_sd_keys:
            new_sd.update({k_[pfx:]: v_})

    return new_sd


bert_chkpt_help = 'Format: run_{n}/ep{k}_mb{i}.chk. If not specified, will create embeddings for the original BERT '
bert_chkpt_help += 'model specified in --bert_model_type'

ap = ArgumentParser()
ap.add_argument('-g', '--gfolds_checkpoint', help='Format: run_{n}/ep{k}_mb{i}.chk', type=str, default=None)
ap.add_argument('-b', '--bert_checkpoint', help=bert_chkpt_help, type=str, default=None)
ap.add_argument('--bert_model_type', help='\'base\' or \'large\'', type=str, default='base')
ap_args = ap.parse_args()

DEVICE = 0 if torch.cuda.is_available() else 'cpu'
FILE_FP = GLOBAL_FP + '/prop_inf/mcrae_concept_sents_all'

with open(GLOBAL_FP + '/prop_inf/mcrae_features.json', 'r') as f:
    feat_file = json.load(f)

if ap_args.gfolds_checkpoint is None:
    if ap_args.bert_model_type is None:
        raise ValueError('Specify either \'--gfolds_checkpoint\' or \'--bert_model_type\' (but not both)')

    BERT_MODEL = f'bert-{ap_args.bert_model_type}-uncased'
    model = AutoModelForMaskedLM.from_pretrained(BERT_MODEL).bert
    tokenizer = AutoTokenizer.from_pretrained(BERT_MODEL)

    if ap_args.bert_checkpoint is not None:
        run, chkpt = ap_args.bert_checkpoint.split('/')
        OUT_FP = f'{GLOBAL_FP}/prop_inf/mcrae_data_bert_comp_{ap_args.bert_model_type}.pkl'

        with open(f'{GLOBAL_FP}/bert/runs/{run}/checkpoints/{chkpt}', 'rb') as f:
            pt_sd = pt_sd_to_mlm(pickle.load(f), model)

        model.load_state_dict(pt_sd)
    else:
        OUT_FP = f'{GLOBAL_FP}/prop_inf/mcrae_data_bert_{ap_args.bert_model_type}.pkl'

    model.to(device=DEVICE)
    model.eval()
    mask_id, out_file = tokenizer('[MASK]')['input_ids'][1], []

    with open(FILE_FP, 'r') as f:
        for line in map(lambda x: x.strip(), f):
            term, sent = map(lambda x: x.strip(), line.split(':'))
            sent = sent + ('' if sent[-1] == '.' else '.')

            if '|' in term:
                term, current_term = map(lambda z: z.strip(), term.split('|'))
            elif '_' in term:
                current_term = term.split('_')[0]
            else:
                current_term = term

            with torch.no_grad():
                toks = tokenizer(sent.replace(current_term, '[MASK]'))['input_ids']
                toks1 = toks[:next(i for i in range(len(toks)) if toks[i] == mask_id)]
                idx1 = len(toks1)
                toks2 = toks[idx1 + 1:]
                trgt_toks = tokenizer(current_term)['input_ids'][1:-1]
                idx2 = idx1 + len(trgt_toks)

                model_output = model(torch.tensor([toks1 + trgt_toks + toks2], device=DEVICE)).last_hidden_state
                trgt_output = model_output[:, idx1:idx2, :].squeeze(0)

                out_file.append({
                    'term': term,
                    'features': {int(k): v for k, v in feat_file[term].items()},
                    'embedding': (torch.sum(trgt_output, dim=0).flatten() / trgt_output.size(0)).to(device='cpu')
                })

    with open(OUT_FP, 'wb') as f_out:
        pickle.dump(out_file, f_out)
elif ap_args.bert_model_type is None:
    run, chkpt = ap_args.gfolds_checkpoint.split('/')
    model = SWATransformer.from_swat_module(f'{GLOBAL_FP}/pretraining/{run}/checkpoints/{chkpt}')
    model.to(device=0)
    model.eval()

    parser = PropInfGraphParser.from_pretrained(GLOBAL_FP + '/tokenizer_config.json')
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

    with open(GLOBAL_FP + '/prop_inf/mcrae_data_gfolds.pkl', 'wb') as f_out:
        pickle.dump(out_file, f_out)

    total_f, total_v = sum(failures.values()), len(out_file)
    total_exs = total_f + total_v

    print(f'PARSED: {total_v} / {total_exs} ({round(total_v * 100 / total_exs, 3)}%)')
    print(f'FAILED: {total_f} / {total_exs} ({round(total_f * 100 / total_exs, 3)}%)')

    for f_str, f_cnt in sorted(list(failures.items()), key=lambda z: z[1]):
        print(f'   {f_str}: {f_cnt}')
else:
    raise ValueError('Specify either \'--gfolds_checkpoint\' or \'--bert_model_type\' (but not both)')
