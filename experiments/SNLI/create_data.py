
import gc
import os
import re
import json
import traceback
from tqdm import tqdm
from setup import GLOBAL_FP
from subprocess import DEVNULL
from multiprocessing import Pool
from argparse import ArgumentParser
from model.graphtools import GraphParser, SWADirGraph
from typing import Generator, Optional, Iterable, Union, Tuple
from model import PROHIBITED_TOKENS, GRPH_LABEL_IDX, LINK_ROLE_MAP


class NLIGraphParser(GraphParser):
    def __init__(self, grammar, tokenizer_dict, unk_as_mask=True, _fpt=False):
        super(NLIGraphParser, self).__init__(_fpt=_fpt)
        from delphin import ace, dmrs

        self.grammar, self.tokenizer_dict = os.path.abspath(grammar), tokenizer_dict
        ace_fp = self.grammar[:-len(self.grammar.split('/')[-1])]

        if re.search(re.compile(f'(^|:){re.escape(ace_fp)}($|:)'), os.environ['PATH'].strip()) is None:
            os.environ['PATH'] += ('' if len(os.environ['PATH'].strip()) == 0 else ':') + ace_fp

        self.mrs_parser, self._to_dmrs = ace, dmrs.from_mrs
        self.unk_as_mask, self._unk_toks, self._if_then = unk_as_mask, {}, tokenizer_dict['v']['if_x_then']
        self._arg1, self._arg2 = tokenizer_dict['e']['ARG1'], tokenizer_dict['e']['ARG2']
        self._if_then_feats = tuple(
            map(lambda feat: self.tokenizer_dict['f'][f'[{feat}]'],
                ('SF:prop', 'PROG:-', 'PERF:-', 'MOOD:indicative', 'TENSE:untensed'))
        )
        assert set(self.tokenizer_dict.keys()) == {'v', 'e', 'f'}

        for k in ('v', 'e', 'f'):
            if unk_as_mask or '[UNK]' not in self.tokenizer_dict[k].keys():
                self._unk_toks.update({k: self.tokenizer_dict[k].get('[MASK]', -1)})
            else:
                self._unk_toks.update({k: self.tokenizer_dict[k]['[UNK]']})

    def save(self, *filepath: str, **kwargs) -> Optional[dict]:
        pass

    def _call(
            self,
            input_sentences: Iterable[str],
            s1_idx: int = 5,
            s2_idx: int = 6,
            s1_tree_idx: Optional[int] = 3,
            s2_tree_idx: Optional[int] = 4,
            label_idx: int = 0,
            save: Union[bool, dict] = False,  # : False => rtn graph obj g, dict => rtn g.save(**save) (True => {})
            include_text: bool = True  # no effect if save = False
    ) -> Generator[Optional[
        Union[dict, Tuple[SWADirGraph, str, bool, bool], Tuple[SWADirGraph, str, bool, bool, bool]]], None, None]:
        if s1_tree_idx is None:
            assert len({s1_idx, s2_idx, label_idx}) == 3
            assert s2_tree_idx is None
            check_sents = False
        else:
            assert len({s1_idx, s2_idx, label_idx, s1_tree_idx, s2_tree_idx}) == 5
            assert s2_tree_idx is not None
            check_sents = True

        if save is False:
            include_text = False
        elif save is True:
            save = {}

        for input_s in input_sentences:
            input_items, grm = input_s.split('\t'), False
            s1, s2 = input_items[s1_idx].strip(), input_items[s2_idx].strip()
            label = input_items[label_idx].strip()[0]

            if label not in {'e', 'n', 'c'}:
                out_obj, unk, carg = None, False, False
            else:
                s1_graph, s1_unk, s1_carg, s1_top = self._parse_to_graph(s1)

                if s1_graph is None:
                    out_obj, unk, carg = None, False, False
                else:
                    out_obj, unk, carg, _ = self._parse_to_graph(s2, prev_graph=s1_graph, prev_top=s1_top)
                    unk, carg = unk or s1_unk, carg or s1_carg

            if check_sents:
                grm = all(input_items[x].strip()[:9] == '(ROOT (S ' for x in (s2_tree_idx, s1_tree_idx))

            if save is False:
                yield (out_obj, unk, carg, label, grm) if check_sents else (out_obj, unk, carg, label)
            else:
                out_dict = {
                    'lbl': label,
                    'unk': unk,
                    'carg': carg,
                    'g': None if out_obj is None else out_obj.save(**save)
                }

                if include_text:
                    out_dict.update({'s': {'s1': s1, 's2': s2}})
                if check_sents:
                    out_dict.update({'grm': grm})

                yield out_dict

    def _parse_to_graph(self, sent, prev_graph=None, prev_top=None):
        response = next(self.mrs_parser.parse_from_iterable(self.grammar, (sent,), cmdargs=['-1'], stderr=DEVNULL))

        if len(response['results']) == 0:
            return None, False, False, None

        dmrs = self._to_dmrs(response.result(0).mrs())
        dmrs_top, dmrs_top_id, pred_id_dict = dmrs.top, None, {}
        graph = SWADirGraph() if prev_graph is None else prev_graph
        unk, carg = False, False

        for pred in dmrs.predications:
            pred_name = pred.predicate.lower()

            if pred_name not in PROHIBITED_TOKENS:  # {'focus_d', 'parg_d'}
                if '/' in pred_name:  # OOV
                    pred_id = graph.add_node(self._unk_toks['v'])
                else:
                    pred_id = graph.add_node(self._tokenize_fn(pred_name[(1 if pred_name[0] == '_' else 0):], 'v'))

                for feat, val in pred.properties.items():
                    graph.add_feature(pred_id, self._tokenize_fn(f'[{feat}:{val}]', 'f'))

                unk = unk or (graph.nodes[pred_id][GRPH_LABEL_IDX] == self._unk_toks['v'])
                carg = carg or (pred.carg is not None)
                pred_id_dict.update({pred.id: pred_id})

                if pred.id == dmrs_top:
                    if dmrs_top_id is not None:  # multiple tops???
                        return None, unk, carg, None

                    dmrs_top_id = pred.id

        if dmrs_top_id is None:  # no top
            return None, unk, carg, None

        for link in dmrs.links:
            if link.start in pred_id_dict.keys() and link.end in pred_id_dict.keys():
                graph.add_edge(
                    pred_id_dict[link.start],
                    pred_id_dict[link.end],
                    self._tokenize_fn(LINK_ROLE_MAP.get(link.role, link.role), 'e')
                )

        if prev_graph is None:
            return graph, unk, carg, pred_id_dict[dmrs_top_id]

        if_then_id = graph.add_node(self._if_then, prepend=True)
        graph.add_edge(if_then_id, prev_top, self._arg2)
        graph.add_edge(if_then_id, pred_id_dict[dmrs_top_id], self._arg1)

        for ft in self._if_then_feats:
            graph.add_feature(if_then_id, ft)

        return graph, unk, carg, None

    def _tokenize_fn(self, s, kind):
        return self.tokenizer_dict[kind].get(s, self._unk_toks[kind])

    @classmethod
    def from_pretrained(cls, filepath: str, **kwargs) -> "NLIGraphParser":
        with open(os.path.abspath(filepath), 'r') as f:
            fp_kwargs = {**json.load(f), **kwargs, **{'_fpt': True}}

        return cls(fp_kwargs.pop('grammar'), fp_kwargs.pop('tokenizer_dict'), **fp_kwargs)

    @classmethod
    def init_for_pretraining(cls, *args, **kwargs) -> None:
        raise NotImplementedError


def worker(self_idx_split):
    graph_parser = NLIGraphParser.from_pretrained(f'{GLOBAL_FP}tokenizer_config.json', **GRAPH_PARSER_KWARGS)
    (self_idx, split), out_file = self_idx_split, []
    stats_dict = {'success': 0, 'failure': 0, 'exception': 0}

    with open(f'{FP}{split}/exceptions/{self_idx}', 'w') as f:
        f.write('')

    with open(f'{FP}{split}/text/{self_idx}', 'r') as f:
        for i_, line in enumerate(map(lambda x: x.strip(), f)):
            if len(line) > 0:
                try:
                    grph = graph_parser(line, save=True)

                    if grph['g'] is None:
                        stats_dict['failure'] += 1
                    else:
                        stats_dict['success'] += 1
                        out_file.append(grph)
                except Exception as e:
                    stats_dict['exception'] += 1

                    with open(f'{FP}{split}/exceptions/{self_idx}', 'a') as f_err:
                        f_err.write('\n\n\n--------------------------------------------\n\n\n')
                        f_err.write(str(type(e)) + ' (S=' + str(i_) + '):\n\n\n')
                        f_err.write('\n'.join(traceback.format_tb(e.__traceback__)))

    with open(f'{FP}{split}/stats/{self_idx}_stats', 'w') as f:
        json.dump(stats_dict, f)

    if len(out_file) > 0:
        with open(f'{FP}{split}/parsed_files/{self_idx}', 'w') as f:
            json.dump(out_file, f)


ap = ArgumentParser()
ap.add_argument('-n', '--num_workers', help='Number of parsing threads', type=int, required=True)
args = ap.parse_args()
N_WORKERS = args.num_workers

GRAPH_PARSER_KWARGS = {'unk_as_mask': True}
FP = f'{GLOBAL_FP}/nli/data/'

for split_name in ('dev', 'test', 'train'):
    with open(f'{FP}snli_1.0/snli_1.0_{split_name}.txt', 'r') as f:
        readlines = [x for x in f.readlines()[1:] if len(x.strip()) > 0]

    len_rl = len(readlines)
    split_size = -(-len_rl // N_WORKERS)
    os.mkdir(f'{FP}{split_name}/text')

    for i in tqdm(range(N_WORKERS)):
        with open(f'{FP}{split_name}/text/{i}', 'w') as f:
            for line in readlines[i * split_size: (i + 1) * split_size]:
                f.write(line.strip() + '\n')

    del readlines
    gc.collect()

    for file_name in ('stats', 'parsed_files', 'exceptions'):
        if file_name not in os.listdir(f'{FP}{split_name}'):
            os.mkdir(f'{FP}{split_name}/{file_name}')
        else:
            for fn in os.listdir(f'{FP}{split_name}/{file_name}'):
                os.remove(f'{FP}{split_name}/{file_name}/{fn}')

    with Pool() as pool:
        pool.map(worker, [(i, split_name) for i in range(N_WORKERS)])

    stats_dict_main = {'success': 0, 'failure': 0, 'exception': 0}

    for fn in os.listdir(f'{FP}{split_name}/stats'):
        with open(f'{FP}{split_name}/stats/{fn}', 'r') as f_fn:
            stats_fn = json.load(f_fn)

        os.remove(f'{FP}{split_name}/stats/{fn}')

        for k_stat in stats_dict_main.keys():
            stats_dict_main[k_stat] += stats_fn[k_stat]

    os.rmdir(f'{FP}{split_name}/stats')
    total_lines = sum(stats_dict_main.values())
    s_rate = round(stats_dict_main['success'] * 100 / total_lines, 3)
    f_rate = round(stats_dict_main['failure'] * 100 / total_lines, 3)
    e_rate = round(stats_dict_main['exception'] * 100 / total_lines, 3)

    with open(f'{FP}{split_name}/stats', 'w') as f_stat:
        f_stat.write(f'PARSED:     {stats_dict_main["success"]} ({s_rate}%)\n')
        f_stat.write(f'FAILED:     {stats_dict_main["failure"]} ({f_rate}%)\n')
        f_stat.write(f'EXCEPTIONS: {stats_dict_main["exception"]} ({e_rate}%)\n')

    split_data = []

    for fn in os.listdir(f'{FP}{split_name}/parsed_files'):
        with open(f'{FP}{split_name}/parsed_files/{fn}', 'r') as f_fn:
            split_data.extend(json.load(f_fn))

        os.remove(f'{FP}{split_name}/parsed_files/{fn}')

    os.rmdir(f'{FP}{split_name}/parsed_files')

    with open(f'{FP}{split_name}.json', 'w') as f_split:
        json.dump(split_data, f_split)
