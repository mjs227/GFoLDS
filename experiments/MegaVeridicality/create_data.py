
import os
import re
import json
import random
from tqdm import tqdm
from setup import GLOBAL_FP
from subprocess import DEVNULL
from model.graphtools import GraphParser, SWADirGraph
from model import PROHIBITED_TOKENS, LINK_ROLE_MAP, GRPH_LABEL_IDX


DATA_FP = GLOBAL_FP + '/factuality/data/mega-veridicality-v2.1.tsv'
OUT_FP = GLOBAL_FP + '/factuality/data/mv_data_bin_guc.json'
PARSER_FP = GLOBAL_FP + '/tokenizer_config.json'
BIN = True
GUC = True


class MVGraphParser(GraphParser):
    def __init__(self, grammar, tokenizer_dict, unk_as_mask=True, _fpt=False):
        super(MVGraphParser, self).__init__(_fpt=_fpt)
        from delphin import ace, dmrs

        self.grammar, self.tokenizer_dict = os.path.abspath(grammar), tokenizer_dict
        ace_fp = self.grammar[:-len(self.grammar.split('/')[-1])]

        if re.search(re.compile(f'(^|:){re.escape(ace_fp)}($|:)'), os.environ['PATH'].strip()) is None:
            os.environ['PATH'] += ('' if len(os.environ['PATH'].strip()) == 0 else ':') + ace_fp

        self.mrs_parser, self._to_dmrs = ace, dmrs.from_mrs
        self._unk_toks, self.unk_as_mask = {}, unk_as_mask
        assert set(tokenizer_dict.keys()) == {'v', 'e', 'f'}

        for k_ in ('v', 'e', 'f'):
            if unk_as_mask or '[UNK]' not in self.tokenizer_dict[k_].keys():
                self._unk_toks.update({k_: self.tokenizer_dict[k_].get('[MASK]', -1)})
            else:
                self._unk_toks.update({k_: self.tokenizer_dict[k_]['[UNK]']})

        self._num_sg = self.tokenizer_dict['f']['[NUM:sg]']
        self._num_pl = self.tokenizer_dict['f']['[NUM:pl]']

    def save(self, *args, **kwargs):
        raise NotImplementedError

    def _call(self, input_sentences: str, **_):
        for line_ in map(lambda s: s.strip(), input_sentences):
            if len(line_) > 0:
                yield self._parse_template(line_)

    def _parse_template(self, in_sent):
        response = next(self.mrs_parser.parse_from_iterable(self.grammar, (in_sent,), cmdargs=['-1'], stderr=DEVNULL))

        if len(response['results']) == 0:
            return {'g': None}

        dmrs = self._to_dmrs(response.result(0).mrs())
        graph, pred_id_dict, no_unk, no_carg = SWADirGraph(), {}, True, True

        for pred in dmrs.predications:
            no_carg = no_carg and pred.carg is None
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
                no_unk = no_unk and not graph.nodes[pred_id][GRPH_LABEL_IDX] == self._unk_toks['v']

        for link in dmrs.links:
            if link.start in pred_id_dict.keys() and link.end in pred_id_dict.keys():
                graph.add_edge(
                    pred_id_dict[link.start],
                    pred_id_dict[link.end],
                    self._tokenize_fn(LINK_ROLE_MAP.get(link.role, link.role), 'e')
                )

        return {'g': graph.save(), 's': in_sent, 'no_unk': no_unk, 'no_carg': no_carg}

    def _tokenize_fn(self, s, kind):
        return self.tokenizer_dict[kind].get(s, self._unk_toks[kind])

    @classmethod
    def init_for_pretraining(cls, *args, **kwargs) -> None:
        raise NotImplementedError

    @classmethod
    def from_pretrained(cls, filepath, **kwargs) -> "MVGraphParser":
        with open(os.path.abspath(filepath), 'r') as f_pt:
            fp_kwargs = {**json.load(f_pt), **kwargs, **{'_fpt': True}}

        return cls(fp_kwargs.pop('grammar'), fp_kwargs.pop('tokenizer_dict'), **fp_kwargs)


parser = MVGraphParser.from_pretrained(PARSER_FP)
label_map = {'yes': 1, 'maybe': 0, 'no': -1}
data_dict, out_data = {}, []

with open(DATA_FP, 'r') as f:
    for line in map(lambda z: z.strip(), f):
        if len(line) > 0 and not line.startswith('participant'):
            line_items = line.split('\t')
            label = line_items[8].strip()

            if len(label) > 0:
                sent = line_items[7].strip().replace('\\\'', '\'')

                if sent in data_dict.keys():
                    data_dict[sent].append(label_map[label])
                else:
                    data_dict.update({sent: [label_map[label]]})

for k, v in tqdm(data_dict.items()):
    grph = parser(k, save=True, include_text=True)

    if (grph['g'] is not None) and ((grph['no_unk'] and grph['no_carg']) or not GUC):
        grph.update({'l': (lambda w: (1 if w > 0 else 0) if BIN else w)(sum(v) / len(v))})
        out_data.append(grph)

random.shuffle(out_data)
len_train = len(out_data) * 4 // 5
out_file = {'train': out_data[:len_train], 'test': out_data[len_train:]}

with open(OUT_FP, 'w') as f:
    json.dump(out_file, f)

print()
print(f'PARSE RATE: {len(out_data) * 100 / len(data_dict.keys())}%')
print(f'TRAIN: {len(out_file["train"])}')
print(f'TEST:  {len(out_file["test"])}')
