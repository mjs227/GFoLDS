
import re
import os
import json
import torch
import _constants as cons
from copy import deepcopy
from subprocess import DEVNULL
from abc import ABC, abstractmethod
from typing import Union, Iterable, Generator, Optional, List, Any, TypeVar, Type


GraphParserSubclass = TypeVar('GraphParserSubclass', bound="GraphParser")


class SWADirGraph:
    def __init__(self):
        self._adj_in, self._adj_out, self.nodes, self.features = {}, {}, {}, {}
        self.is_transposed, self._idxs_assigned, self._count = False, False, 0
        self.adj_out, self.adj_in = self._adj_out, self._adj_in

    def add_node(self, label: Union[int, str], prepend: bool = False) -> int:  # can be str for debug/decode
        self._idxs_assigned = self._idxs_assigned and not prepend
        cnt = self._count * (-1 if prepend else 1)
        self.nodes.update({cnt: [label, len(self.nodes), cnt]})  # lbl, idx, id
        self.adj_in.update({cnt: set()})
        self.adj_out.update({cnt: set()})
        self.features.update({cnt: set()})
        self._count += 1

        return cnt  # node id

    def remove_node(self, i: int) -> None:
        for j, lbl in self.adj_in[i]:
            self.adj_out[j].remove((i, lbl))

        for j, lbl in self.adj_out[i]:
            self.adj_in[j].remove((i, lbl))

        _ = self.adj_in.pop(i), self.adj_out.pop(i), self.nodes.pop(i), self.features.pop(i)
        self._idxs_assigned = self._idxs_assigned and abs(i) == len(self.nodes) - 1

    def add_feature(self, node: int, feature: Union[int, str]) -> None:  # can be str for debug/decode
        self.features[node].add(feature)

    def remove_feature(self, node: int, feature: Union[int, str]) -> None:
        self.features[node].remove(feature)

    def add_edge(self, src: int, trgt: int, lbl: Union[int, str]) -> None:  # can be str for debug/decode
        self.adj_out[src].add((trgt, lbl))
        self.adj_in[trgt].add((src, lbl))

    def remove_edge(self, i: int, j: int, lbl: Union[int, str]) -> None:
        self.adj_out[i].remove((j, lbl))
        self.adj_in[j].remove((i, lbl))

    def transpose(self, copy: bool = False) -> Optional["SWADirGraph"]:
        if copy:
            out_dg = deepcopy(self)
            out_dg.transpose()

            return out_dg

        self.is_transposed = not self.is_transposed

        if self.is_transposed:
            self.adj_in, self.adj_out = self._adj_out, self._adj_in
        else:
            self.adj_in, self.adj_out = self._adj_in, self._adj_out

    def save(
            self,
            *filepath: str,  # No args => return dict, str => save dict as json to filepath
            sparse: bool = True,  # compressed representation (recommended)
            feature_counts: bool = True,  # slight speedup for batching (no effect if sparse is false)
            assign_indices: bool = True  # slight speedup for batching (no effect if sparse is True)
    ) -> Optional[dict]:
        save_dict = {'n': {str(k): v for k, v in self.nodes.items()}}  # keys => str to align with json format

        if sparse:
            if feature_counts:
                save_dict.update({'l': max(map(len, self.features.values()), default=0)})

            self._assign_idxs()
            save_dict.update({
                'f': {str(k): list(v) for k, v in self.features.items() if len(v) > 0},
                'i': {str(k): list(v) for k, v in self._adj_in.items() if len(v) > 0}
            })
        else:
            if assign_indices:
                self._assign_idxs()

            save_dict.update({
                'f': {str(k): list(v) for k, v in self.features.items()},
                'i': {str(k): list(v) for k, v in self._adj_in.items()},
                'o': {str(k): list(v) for k, v in self._adj_out.items()},
                'x': self._idxs_assigned,
                'c': self._count,
                't': self.is_transposed
            })

        if len(filepath) == 0:
            return save_dict
        if len(filepath) == 1:
            with open(os.path.abspath(filepath[0]), 'w') as f:
                json.dump(save_dict, f)
        else:
            raise TypeError(f'SWADirGraph.save() takes <= 1 arguments ({len(filepath)} given).')

    def visualize(
            self,
            name: str,
            include_features: bool = True,
            **pyvis_kwargs
    ) -> None:
        from pyvis.network import Network

        self._assign_idxs()
        g = Network(**{**pyvis_kwargs, **{'directed': True}})
        g.add_nodes(list(range(len(self.nodes))))

        for node_id, node in self.nodes.items():
            g.nodes[node[cons.IDX]]['label'] = str(node[cons.LABEL])

            for trgt_id, lbl in self.adj_out[node_id]:
                g.add_edge(node[cons.IDX], self.nodes[trgt_id][cons.IDX], label=str(lbl))

        if include_features:
            node_cnt = len(self.nodes)

            for node_id, features in self.features.items():
                node_idx = self.nodes[node_id][cons.IDX]

                for feature in features:
                    g.add_node(n_id=node_cnt, label=str(feature))
                    g.add_edge(node_cnt, node_idx, label='FEAT')
                    node_cnt += 1

        g.save_graph(name)

    def get_nodes(self, as_sorted: bool = False, as_is: bool = False) -> List[List[Union[str, int]]]:
        assert as_sorted or not as_is

        if as_sorted:
            if as_is:
                return self._get_nodes_sorted_asis()

            return self._get_nodes_sorted()

        return self._get_nodes()

    def _get_nodes(self):
        return [v for _, v in self.nodes.items()]

    def _get_nodes_sorted_asis(self):  # sorted (as-is)
        out_list = self._get_nodes()
        out_list.sort(key=lambda x: x[cons.IDX])

        return out_list

    def _get_nodes_sorted(self):  # sorted (update)
        if self._idxs_assigned:
            return self._get_nodes_sorted_asis()

        return self._assign_idxs(rtn=True)

    def _assign_idxs(self, rtn=False):
        if not self._idxs_assigned:
            self._idxs_assigned = True
            node_list = self.get_nodes()
            node_list.sort(key=lambda x: x[cons.ID])

            for i in range(len(node_list)):
                node_list[i][cons.IDX] = i

            if rtn:
                return node_list

    @classmethod
    def from_dict(cls, graph_dictionary: Union[dict, str]) -> "SWADirGraph":
        if isinstance(graph_dictionary, str):
            with open(os.path.abspath(graph_dictionary), 'r') as f:
                in_dict = json.load(f)
        else:
            in_dict = graph_dictionary

        out_graph = cls()
        out_graph._idxs_assigned = in_dict.get('x', True)  # no 'x' for sparse (assign idxs before save)
        out_graph.nodes.update({int(k): v for k, v in in_dict['n'].items()})

        if 'o' in in_dict.keys():  # not sparse
            out_graph._count = in_dict['c']
            out_graph._adj_in.update({int(k): set(map(tuple, v)) for k, v in in_dict['i'].items()})
            out_graph._adj_out.update({int(k): set(map(tuple, v)) for k, v in in_dict['o'].items()})
            out_graph.features.update({int(k): set(v) for k, v in in_dict['f'].items()})

            if in_dict['t']:
                out_graph.transpose()
        else:  # sparse
            out_graph._count = max(abs(v[cons.ID]) for _, v in out_graph.nodes.items()) + 1
            out_graph.features.update({k: set(in_dict['f'].get(str(k), ())) for k in out_graph.nodes.keys()})
            out_graph._adj_in.update({k: set(map(tuple, in_dict['i'].get(str(k), ()))) for k in out_graph.nodes.keys()})
            out_graph._adj_out.update({k: set() for k in out_graph.nodes.keys()})

            for trgt, adj in out_graph._adj_in.items():
                for src, lbl in adj:
                    out_graph._adj_out[src].add((trgt, lbl))

        return out_graph


class GraphParser(ABC):
    def __init__(self, _fpt=False):
        if not _fpt:
            self_type = cons.type_str(self)
            raise EnvironmentError(
                f'Instantiate using the {self_type}.from_pretrained() or {self_type}.init_for_pretraining() methods'
            )

    def __call__(
            self,
            input_sentences: Union[str, Iterable[str]],
            **kwargs
    ) -> Union[Optional[Union[SWADirGraph, dict]], Generator[Optional[Union[SWADirGraph, dict]], None, None]]:
        if isinstance(input_sentences, str):
            return next(self._call((input_sentences,), **kwargs))

        return self._call(input_sentences, **kwargs)

    @abstractmethod
    def save(self, *filepath: str, **kwargs) -> Optional[dict]:
        pass

    @abstractmethod
    def _call(
            self,
            input_sentences: Iterable[str],
            **kwargs
    ) -> Generator[Optional[Union[SWADirGraph, dict]], None, None]:
        pass

    @classmethod
    @abstractmethod
    def from_pretrained(cls: Type[GraphParserSubclass], *args, **kwargs) -> GraphParserSubclass:
        pass

    @classmethod
    @abstractmethod
    def init_for_pretraining(cls: Type[GraphParserSubclass], *args, **kwargs) -> GraphParserSubclass:
        pass


class DMRSGraphParser(GraphParser):
    def __init__(self, grammar: str, tokenizer_dict: dict, unk_as_mask: bool = False, _debug=False, _fpt=False):
        super(DMRSGraphParser, self).__init__(_fpt=(_fpt or _debug))
        from delphin import ace, dmrs

        self.grammar, self.tokenizer_dict = os.path.abspath(grammar), tokenizer_dict
        ace_fp, _ = self.grammar.rsplit('/', 1)

        if re.search(re.compile(f'(^|:){re.escape(ace_fp)}($|:)'), os.environ['PATH'].strip()) is None:
            os.environ['PATH'] += ('' if len(os.environ['PATH'].strip()) == 0 else ':') + ace_fp

        self.mrs_parser, self._to_dmrs, self.unk_as_mask = ace, dmrs.from_mrs, unk_as_mask
        self._td_decode, self._decode_fn = None, self._decode_err

        if _debug:
            self._unk_toks, self._tokenize_fn = {k: '[UNK]' for k in ('v', 'e', 'f')}, (lambda z, _: z)
        else:
            assert set(tokenizer_dict.keys()) == {'v', 'e', 'f'}
            self._unk_toks = {}

            for k in ('v', 'e', 'f'):
                if unk_as_mask or '[UNK]' not in self.tokenizer_dict[k].keys():
                    self._unk_toks.update({k: self.tokenizer_dict[k].get('[MASK]', -1)})
                else:
                    self._unk_toks.update({k: self.tokenizer_dict[k]['[UNK]']})

            self._tokenize_fn = self._tokenize

    def save(self, *filepath: str) -> Optional[dict]:
        if len(filepath) == 0:
            return {'grammar': self.grammar, 'unk_as_mask': self.unk_as_mask, 'tokenizer_dict': self.tokenizer_dict}
        if len(filepath) == 1:
            with open(os.path.abspath(filepath[0]), 'w') as f:
                json.dump(self.save(), f)
        else:
            raise TypeError(f'DRMSGraphParser.save() takes <= 1 arguments ({len(filepath)} given).')

    def parse_dmrs(  # basically just a wrapper for self._parse_dmrs()
            self,
            input_sentences: Union[str, Iterable[str]],
            as_generator: bool = True  # no effect if input_sentences is str (i.e. not iterable)
    ) -> Union[Any, Generator[Any, None, None], List[Any]]:  # "Any" = delphin.DMRS (can't declare w/o importing)
        if isinstance(input_sentences, str):
            return next(self._parse_dmrs((input_sentences,)))[0]

        return (lambda x: x if as_generator else list(x))(y for y, _ in self._parse_dmrs(input_sentences))

    def decode(  # just a wrapper for self._decode_fn()
            self,
            graph,  # Union[dict, SWADirGraph, "SWATForMLMOutput"] (can't declare bc of circular imports)
            copy: bool = False,
            k: int = 1,  # top-k for mask fill (k = 0 => do not fill); for SWATOutput only
            mask_idx: int = cons.MASK_TOKEN_ID,  # for SWATOutput only
            ignore_idx: int = cons.IGNORE_ID,  # for SWATOutput only
            token_pad_idx: int = cons.PAD_TOKEN_ID,  # for SWATOutput only
            feature_pad_idx: Optional[int] = None  # for SWATOutput only
    ) -> Union[dict, SWADirGraph]:
        return self._decode_fn(graph, copy, k, mask_idx, ignore_idx, token_pad_idx, feature_pad_idx)

    def _parse_dmrs(self, in_sents):
        responses = self.mrs_parser.parse_from_iterable(self.grammar, in_sents, cmdargs=['-1'], stderr=DEVNULL)

        for response, in_sent in zip(responses, in_sents):
            if len(response['results']) == 0:
                yield None, None  # for parse_dmrs() above
            else:
                yield self._to_dmrs(response.result(0).mrs()), in_sent

    def _call(
            self,
            input_sentences: Iterable[str],
            save: Union[bool, dict] = False,  # : False => rtn graph obj g, dict => rtn g.save(**save) (True => {})
            include_text: bool = False  # no effect if save = False
    ) -> Generator[Optional[Union[SWADirGraph, dict]], None, None]:
        if save is False:
            include_text = False
        elif save is True:
            save = {}

        for dmrs, in_sent in self._parse_dmrs(input_sentences):
            if dmrs is None:
                yield {'s': in_sent, 'g': None} if include_text else None
                continue

            graph = dmrs_to_swa_graph(self, dmrs)

            if save is False:
                yield graph
            else:
                yield (lambda g: {'s': in_sent, 'g': g} if include_text else g)(graph.save(**save))

    def _tokenize_pt(self, s, kind):
        td_kind = self.tokenizer_dict[kind]

        if s in td_kind.keys():
            return td_kind[s]

        n_toks = len(td_kind)
        td_kind.update({s: n_toks})

        return n_toks

    def _tokenize(self, s, kind):
        return self.tokenizer_dict[kind].get(s, self._unk_toks[kind])

    def _decode(self, graph, copy, k, mask_idx, ignore_idx, token_pad_idx, feature_pad_idx):
        if isinstance(graph, dict):
            return self.decode(SWADirGraph.from_dict(graph)).save(
                sparse=('o' not in graph.keys()),
                feature_counts=('l' in graph.keys()),
                assign_indices=graph.get('x', True)
            )
        if isinstance(graph, SWADirGraph):
            if copy:
                return self.decode(deepcopy(graph))

            for node in graph.nodes.values():
                node[cons.LABEL] = self._td_decode['v'][node[cons.LABEL]]

            for k, k_feat in graph.features.items():
                graph.features[k] = set(self._td_decode['f'][x] for x in k_feat)

            for src, adj_src in graph._adj_out.items():
                graph._adj_out[src] = set((x, self._td_decode['e'][y]) for x, y in adj_src)

            for trgt, adj_trgt in graph._adj_in.items():
                graph._adj_in[trgt] = set((x, self._td_decode['e'][y]) for x, y in adj_trgt)

            return graph

        # if isinstance(graph, SWATForMLMOutput):
        if k < 0:
            raise ValueError(f'k (={k}) must be > 0')

        out_list, f_pad = [], (token_pad_idx if feature_pad_idx is None else feature_pad_idx)

        def fill_mask(tok_pos, logits):
            tok_logits = logits[0, tok_pos]

            for j_ in range(k):
                top_idx = torch.argmax(tok_logits, dim=0).item()
                top_val = round(tok_logits[top_idx].item(), 5)
                tok_logits[top_idx] = 0.0

                yield f'{self._td_decode["v"][top_idx]} (P={top_val})'

        for i, batch_elem in enumerate(graph.debatch()):
            if batch_elem._softmax_logits is None:
                batch_logits = batch_elem.softmax_logits
                batch_elem._softmax_logits = None
            else:
                batch_logits = batch_elem.softmax_logits.clone()

            out_graph, nodes = SWADirGraph(), batch_elem.tokens[0].tolist()

            if graph.targets is None:
                targets = [ignore_idx] * len(nodes)
            else:
                targets = batch_elem.targets[0].tolist()

            for j, node in enumerate(nodes):
                if node == mask_idx:
                    node_str = f"TARGET={'NONE' if targets == ignore_idx else self._td_decode['v'][targets[j]]}:\n"
                    out_graph.add_node(node_str + '\n'.join(fill_mask(j, batch_logits)))
                else:
                    out_graph.add_node(self._td_decode['v'][node])

            for lbl_i in range(batch_elem.edge_mask.shape(1)):
                lbl = self._td_decode['v'][lbl_i]

                for trgt_i in range(len(nodes)):
                    for src_i in range(len(nodes)):
                        if batch_elem.edge_mask[0][lbl_i][trgt_i][src_i].item():
                            out_graph.add_edge(src_i, trgt_i, lbl)

            for i_, feats in enumerate(batch_elem.features[0].tolist()):
                for f in set(feats) - {f_pad}:
                    out_graph.add_feature(i_, self._td_decode['f'][f])

            out_list.append(out_graph)

            del batch_logits

        return out_list if len(out_list) > 1 else out_list[0]

    def _decode_err(self, *_):
        raise EnvironmentError(
            'DMRSGraphParser.decode() can only be called on a DMRSGraphParser object that has been instantiated '
            'via DMRSGraphParser.from_pretrained()'
        )

    @classmethod
    def from_pretrained(cls, filepath: str, **kwargs) -> "DMRSGraphParser":
        with open(os.path.abspath(filepath), 'r') as f:
            fp_kwargs = {**json.load(f), **kwargs, **{'_fpt': True}}

        out_obj = cls(
            fp_kwargs.pop('grammar'),
            cons.dict_kwargs(fp_kwargs.pop('tokenizer_dict', None), default=cons.DEFAULT_TOKENIZER_DICT),
            **fp_kwargs
        )
        out_obj._td_decode, out_obj._decode_fn = {}, out_obj._decode

        for k, td_k in out_obj.tokenizer_dict.items():
            out_obj._td_decode.update({k: [None] * len(td_k)})

            for tok_str, tok_id in td_k.items():
                out_obj._td_decode[k][tok_id] = tok_str

        return out_obj

    @classmethod
    def init_for_pretraining(
            cls,
            filepath: Optional[str] = None,  # tokenizer config
            grammar: Optional[str] = None,  # str => path to grm, None => loaded from "filepath" file
            tokenizer_dict: Optional[dict] = None,
            **kwargs
    ) -> "DMRSGraphParser":
        if filepath is None:
            assert grammar is not None
            graph_parser = cls(
                grammar,
                cons.dict_kwargs(tokenizer_dict, default=cons.DEFAULT_TOKENIZER_DICT),
                _fpt=True,
                **kwargs
            )
        else:
            assert grammar is None and tokenizer_dict is None
            graph_parser = cls.from_pretrained(filepath, grammar=grammar, tokenizer_dict=tokenizer_dict, **kwargs)

        graph_parser._tokenize_fn = graph_parser._tokenize_pt

        return graph_parser


def dmrs_to_swa_graph(gp: GraphParser, dmrs) -> "SWADirGraph":
    graph, pred_id_dict = SWADirGraph(), {}

    for pred in dmrs.predications:
        pred_name = pred.predicate.lower()

        if pred_name not in cons.PROHIBITED_TOKENS:  # {'focus_d', 'parg_d'}
            if '/' in pred_name:  # OOV
                pred_id = graph.add_node(gp._unk_toks['v'])
            else:
                pred_id = graph.add_node(gp._tokenize_fn(pred_name[(1 if pred_name[0] == '_' else 0):], 'v'))

            for feat, val in pred.properties.items():
                graph.add_feature(pred_id, gp._tokenize_fn(f'[{feat}:{val}]', 'f'))

            pred_id_dict.update({pred.id: pred_id})

    for link in dmrs.links:
        if link.start in pred_id_dict.keys() and link.end in pred_id_dict.keys():
            graph.add_edge(
                pred_id_dict[link.start],
                pred_id_dict[link.end],
                gp._tokenize_fn(cons.LINK_ROLE_MAP.get(link.role, link.role), 'e')
            )

    return graph
