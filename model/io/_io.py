
import gc
import torch
import random
import warnings
import model._constants as cons
from abc import ABC, abstractmethod
from model.graphtools._graphtools import SWADirGraph, DMRSGraphParser
from typing import Collection, Union, Optional, Iterable, Tuple, Generator, List, TypeVar, Type


SWATInputSubclass = TypeVar('SWATInputSubclass', bound="SWATInput")
SWATOutputSubclass = TypeVar('SWATOutputSubclass', bound="_SWATOutput")
SWATForClsInputSubclass = TypeVar('SWATForClsInputSubclass', bound="_SWATForClsInput")
SWATForClsOutputSubclass = TypeVar('SWATForClsOutputSubclass', bound="_SWATForClsOutput")


class _SWATIO(ABC):
    def __init__(
            self,
            edge_mask: torch.Tensor,
            tokens: torch.Tensor,
            features: torch.Tensor,
            device: Union[torch.device, int, str] = 'cpu'
    ):
        assert len(tuple(tokens.shape)) == 2
        assert len(tuple(features.shape)) == 3
        assert len(tuple(edge_mask.shape)) == 4
        assert edge_mask.size(0) == tokens.size(0) == features.size(0)  # batch_size
        assert edge_mask.size(2) == edge_mask.size(3) == tokens.size(1) == features.size(1)  # num tokens

        self.device = device
        self.tokens = tokens.to(device=device)  # B x N
        self.features = features.to(device=device)  # B x N x F_max
        self.edge_mask = edge_mask.to(device=device)  # B x L x N x N

    def to_dense_mask(self) -> None:
        self.edge_mask = self.edge_mask.to_dense()

    def to_sparse_mask(self, **kwargs) -> None:
        self.edge_mask = self.edge_mask.to_sparse(**kwargs)

    @property
    def token_len(self) -> int:
        return self.tokens.size(1)

    @property
    def feature_len(self) -> int:
        return self.features.size(2)

    @property
    def batch_size(self) -> int:
        return self.tokens.size(0)

    @property
    def dtypes(self) -> dict:
        return self._dtypes()

    def _dtypes(self):
        return {'token_dtype': self.tokens.dtype}

    @abstractmethod
    def clone(
            self,
            token_dtype: Optional[torch.dtype] = None,
            device: Optional[Union[torch.dtype, int, str]] = None,
            **kwargs
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, Optional[Union[int, str, torch.device]], dict]:
        return (
            self.edge_mask.to(copy=True, dtype=torch.bool, device=device, **kwargs),
            self.tokens.to(copy=True, dtype=token_dtype, device=device, **kwargs),
            self.features.to(copy=True, dtype=token_dtype, device=device, **kwargs),
            self.device if device is None else device,
            kwargs
        )

    @abstractmethod
    def to(self, token_dtype: Optional[torch.dtype] = None, **kwargs) -> None:
        token_dtype = self.tokens.dtype if token_dtype is None else token_dtype
        self.tokens = self.tokens.to(dtype=token_dtype, **kwargs)
        self.features = self.features.to(dtype=token_dtype, **kwargs)
        self.edge_mask = self.edge_mask.to(dtype=torch.bool, **kwargs)
        self.device = kwargs.get('device', self.device)


class _SWATForClsIO(_SWATIO, ABC):  # implements stuff for targets
    def __init__(
            self,
            edge_mask: torch.Tensor,
            tokens: torch.Tensor,
            features: torch.Tensor,
            targets: Optional[torch.Tensor],
            device: Union[torch.device, int, str] = 'cpu',
            _to_device=True  # for SWATForMLMOutput.from_swat_input()
    ):
        _SWATIO.__init__(self, edge_mask, tokens, features, device=device)
        assert targets.size(0) == self.batch_size
        assert len(tuple(targets.shape)) == 2

        if _to_device:
            self._long_type = _long_type(self.device)
            self.targets = targets.to(device=device).type(self._long_type)
        else:
            self._long_type, self.targets = None, None

    def to(
            self,
            target_dtype: Optional[torch.dtype] = None,
            token_dtype: Optional[torch.dtype] = None,
            **kwargs
    ) -> None:
        _SWATIO.to(self, token_dtype=token_dtype, **kwargs)
        self._long_type = _long_type(self.device)
        self.targets = self.targets.to(dtype=target_dtype, **kwargs).type(self._long_type)

    def _dtypes(self):
        return {**_SWATIO._dtypes(self), **{'target_dtype': self.targets.dtype}}

    @abstractmethod
    def clone(
            self,
            target_dtype: Optional[torch.dtype] = None,
            **kwargs
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, Optional[Union[int, str, torch.device]], dict]:
        edge_mask, toks, feats, dev, kw = _SWATIO.clone(self, **kwargs)

        return edge_mask, toks, feats, self.targets.to(copy=True, dtype=target_dtype, device=dev), dev, kw


class SWATInput(_SWATIO):  # input for SWATransformer model
    def __init__(
            self,
            edge_mask: torch.Tensor,
            tokens: torch.Tensor,
            features: torch.Tensor,
            device: Union[torch.device, int, str] = 'cpu'
    ):
        _SWATIO.__init__(self, edge_mask, tokens, features, device=device)
        self._n_out = 3  # for batching

    def clone(self, **kwargs) -> "SWATInput":
        edge_mask, toks, feats, dev, _ = _SWATIO.clone(self, **kwargs)

        return SWATInput(edge_mask, toks, feats, device=dev)

    def to(self, **kwargs) -> None:
        _SWATIO.to(self, **kwargs)

    def batch(
            self: SWATInputSubclass,
            swat_input: Union[SWATInputSubclass, Collection[SWATInputSubclass]],
            pad_id: int = cons.PAD_TOKEN_ID,
            feature_pad_id: Optional[int] = None,
            in_place: bool = True,
            _max_tok_feat=None,
            **_kwargs  # for subclasses
    ) -> SWATInputSubclass:
        if isinstance(swat_input, SWATInput):
            return self.batch(
                (swat_input,),
                pad_id=pad_id,
                in_place=in_place,
                _max_tok_feat=_max_tok_feat,
                feature_pad_id=feature_pad_id,
                **_kwargs
            )

        len_input, f_pad = len(swat_input), (pad_id if feature_pad_id is None else feature_pad_id)

        if len(swat_input) > 0:
            if _max_tok_feat is None:
                max_tok, max_feat = self.token_len, self.feature_len

                for x in swat_input:
                    max_tok, max_feat = max(max_tok, x.token_len), max(max_feat, x.feature_len)
            else:
                max_tok, max_feat = _max_tok_feat

            out_lists, tok_dtype = tuple([None] * (len_input + 1) for _ in range(self._n_out)), self.tokens.dtype
            _ = self._batch_tensors(out_lists, 0, max_tok, max_feat, pad_id, f_pad, self, **_kwargs)

            for i, x in enumerate(swat_input, start=1):
                _ = x._batch_tensors(out_lists, i, max_tok, max_feat, pad_id, f_pad, self, **_kwargs)

            batch_tensors = tuple(map(lambda z: torch.cat(z, dim=0), out_lists))

            if in_place:
                self._assign_batch_tensors(batch_tensors)

                return self

            return type(self)(*batch_tensors, device=self.device)

        if in_place:
            return self

        return self.clone()

    def _assign_batch_tensors(self, tensors):
        self.edge_mask, self.tokens, self.features = tensors

    def _batch_tensors(self, out_lists, idx, max_tok, max_feat, pad_id, f_pad_id, swat_obj, **_):
        tok_dtype, device = swat_obj.tokens.dtype, swat_obj.device
        out_tokens, out_mask = self.tokens.to(dtype=tok_dtype, device=device), self.edge_mask.to(device=device)
        size_diff_tok, size_diff_feat = max_tok - self.tokens.size(1), max_feat - self.features.size(2)
        out_features = self.features.to(dtype=tok_dtype, device=device)

        if size_diff_tok > 0:
            pad_toks = torch.full((self.tokens.size(0), size_diff_tok), pad_id, device=device, dtype=tok_dtype)
            pad_mask_1 = torch.full(
                tuple(self.edge_mask.shape)[:2] + (size_diff_tok, self.edge_mask.size(3)),
                False,
                device=device,
                dtype=torch.bool
            )
            pad_mask_2 = torch.full(
                tuple(self.edge_mask.shape)[:2] + (self.edge_mask.size(2) + size_diff_tok, size_diff_tok),
                False,
                device=device,
                dtype=torch.bool
            )
            pad_feats_1 = torch.full(
                (self.features.size(0), size_diff_tok, self.features.size(2)),
                f_pad_id,
                device=device,
                dtype=tok_dtype
            )

            out_mask = torch.cat((out_mask, pad_mask_1), dim=2)
            out_mask = torch.cat((out_mask, pad_mask_2), dim=3)
            out_tokens = torch.cat((out_tokens, pad_toks), dim=1)
            out_features = torch.cat((out_features, pad_feats_1), dim=1)
        if size_diff_feat > 0:
            pad_feats_2 = torch.full(
                tuple(out_features.shape)[:2] + (size_diff_feat,),
                pad_id,
                device=device,
                dtype=tok_dtype
            )
            out_features = torch.cat((out_features, pad_feats_2), dim=2)

        out_lists[0][idx], out_lists[1][idx], out_lists[2][idx] = out_mask, out_tokens, out_features

        return size_diff_tok, device  # for subclasses

    @classmethod
    def from_dir_graph(
            cls,
            graph: Union[SWADirGraph, dict],
            edge_mask_kwargs: Optional[dict] = None,
            token_kwargs: Optional[dict] = None,
            n_edge_labels: int = cons.EDGE_SIZE,
            feature_pad_idx: int = cons.PAD_TOKEN_ID,
            **kwargs
    ) -> "SWATInput":
        if isinstance(graph, dict):
            assert graph.get('x', True), 'Node indices must be assigned to construct SWATInput from graph dictionary'

            if len(graph['n']) == 0:
                raise ValueError('Empty graph dictionary')

            nodes, features, adj_in = graph['n'], graph['f'], graph['i']
            feature_len, to_str = graph.get('l', max(map(len, features.values()), default=1)), str
        else:
            graph._assign_idxs()
            nodes, features, adj_in = graph.nodes, graph.features, graph.adj_in
            feature_len, to_str = max(map(len, features.values()), default=1), lambda x: x

        node_len, tok_kwargs = len(nodes), {**{'dtype': torch.int}, **cons.dict_kwargs(token_kwargs)}
        device = kwargs.get('device', 'cpu')
        mask_tensor = torch.full(
            (1, n_edge_labels, node_len, node_len),
            False,
            dtype=torch.bool,
            device=device,
            **cons.dict_kwargs(edge_mask_kwargs)
        )
        feature_tensor = torch.full((1, node_len, feature_len), feature_pad_idx, device=device, **tok_kwargs)
        token_tensor = torch.zeros((1, node_len), device=device, **tok_kwargs)

        for n_id, n in nodes.items():
            n_idx = n[cons.IDX]
            token_tensor[0, n_idx] = n[cons.LABEL]

            for i, val in enumerate(features.get(n_id, ())):
                feature_tensor[0, n_idx, i] = val

            for m, lbl in adj_in.get(n_id, ()):
                mask_tensor[0, lbl, n_idx, nodes[to_str(m)][cons.IDX]] = True

        return cls(mask_tensor, token_tensor, feature_tensor, **kwargs)

    @classmethod
    def generate_batches(
            cls,
            inputs: Iterable[Union[str, dict, SWADirGraph, "SWATInput", "SWATForMLMInput"]],
            as_generator: bool = True,
            batch_size: int = 1,
            graph_parser: Optional[DMRSGraphParser] = None,
            swat_input_kwargs: Optional[dict] = None,  # no effect if inputs is Iterable[SWATInput/SWATForMLMInput]
            batch_kwargs: Optional[dict] = None,
            collect: bool = False,
            sparse_mask: Union[bool, dict] = False,
            to_after_batch_kwargs: Optional[dict] = None,
            max_seq_len: Optional[int] = cons.MAX_SEQ_LEN
    ) -> Union[List["SWATInput"], Generator["SWATInput", None, None]]:
        return _generate_batches_wrapper(
            as_generator, cls, inputs, batch_size, graph_parser, swat_input_kwargs, batch_kwargs, collect, sparse_mask,
            to_after_batch_kwargs, max_seq_len
        )


class _SWATOutput(_SWATIO, ABC):  # implements debatch + stuff for embeddings
    def __init__(
            self,
            edge_mask: torch.Tensor,
            tokens: torch.Tensor,
            features: torch.Tensor,
            embeddings: torch.Tensor,
            device: Union[torch.device, int, str] = 'cpu',
            _to_device=True  # for SWATOutput.from_swat_input()
    ):
        _SWATIO.__init__(self, edge_mask, tokens, features, device=device)
        self.embeddings = self._emb_init(embeddings, self.tokens.size(1), 3, _to_device)

    def to(
            self,
            embedding_dtype: Optional[torch.dtype] = None,
            token_dtype: Optional[torch.dtype] = None,
            **kwargs
    ) -> None:
        _SWATIO.to(self, token_dtype=token_dtype, **kwargs)
        self.embeddings = self.embeddings.to(dtype=embedding_dtype, **kwargs)

    def clone(self, embedding_dtype: Optional[torch.dtype] = None, **kwargs) -> "SWATOutput":
        edge_mask, toks, feats, dev, kw = _SWATIO.clone(self, **kwargs)  # mask, tokens, features
        embs = self.embeddings.to(copy=True, dtype=embedding_dtype, device=dev, **kw)

        return SWATOutput(edge_mask, toks, feats, embs, device=dev)

    def debatch(
            self: SWATOutputSubclass,
            as_generator: bool = True,
            pad_id: int = cons.PAD_TOKEN_ID,
            **kwargs
    ) -> Union[Generator[SWATOutputSubclass, None, None], List[SWATOutputSubclass]]:
        self_type = type(self)

        def db():
            for i in range(self.batch_size):
                (edge_mask, other), pad_len = self._get_debatch_args(i), self.token_len

                while self.tokens[i, pad_len - 1] == pad_id:
                    pad_len -= 1

                out_i = self_type(
                    edge_mask[..., :pad_len, :pad_len],
                    *tuple(map(lambda x: None if x is None else x[:, :pad_len, ...], other)),
                    device=self.device,
                    _to_device=False
                )

                if len(kwargs) > 0:
                    out_i.to(**kwargs)

                yield out_i

        return (lambda x: x if as_generator else list(x))(db())

    def _emb_init(self, embs, e_size, e_rank, to_device):
        assert embs.size(0) == self.batch_size, str(self.batch_size)
        assert embs.size(1) == e_size, str(e_size)
        assert len(tuple(embs.shape)) == e_rank, str(e_rank)

        return embs.to(device=self.device) if to_device else embs

    def _dtypes(self):
        return {**_SWATIO._dtypes(self), **{'embedding_dtype': self.embeddings.dtype}}

    @abstractmethod
    def _get_debatch_args(self, i):
        pass


class _SWATForClsInput(_SWATForClsIO, SWATInput, ABC):  # implements more concrete stuff for targets
    def __init__(
            self,
            edge_mask: torch.Tensor,
            tokens: torch.Tensor,
            features: torch.Tensor,
            targets: Union[int, torch.Tensor],
            target_init_size: Optional[int] = None,
            device: Union[torch.device, int, str] = 'cpu'
    ):
        if isinstance(targets, int):
            if target_init_size is None:
                raise TypeError('Parameter \'target_init_size\' must be specified if \'targets\' is int')

            targets = torch.full((tokens.size(0), target_init_size), targets)

        assert target_init_size is None or targets.size(1) == target_init_size, str(target_init_size)
        _SWATForClsIO.__init__(self, edge_mask, tokens, features, targets, device=device)
        self._n_out = 4  # batching

    def clone(
            self: SWATForClsInputSubclass,
            target_dtype: Optional[torch.dtype] = None,
            **kwargs
    ) -> SWATForClsInputSubclass:
        edge_mask, toks, feats, trgts, dev, _ = _SWATForClsIO.clone(self, **kwargs)

        return type(self)(edge_mask, toks, feats, trgts, device=dev)

    def batch(  # adds ignore_id kwarg
            self: SWATForClsInputSubclass,
            *args,
            ignore_id: int = cons.IGNORE_ID,
            **kwargs
    ) -> SWATForClsInputSubclass:
        return SWATInput.batch(self, *args, ignore_id=ignore_id, **kwargs)

    def _assign_batch_tensors(self, tensors):
        self.edge_mask, self.tokens, self.features, self.targets = tensors

    @abstractmethod
    def _batch_tensors(self, out_lists, idx, max_tok, max_feat, pad_id, f_pad_id, swat_obj, ignore_id=cons.IGNORE_ID):
        pass

    @classmethod
    def from_swat_input(cls: Type[SWATForClsInputSubclass], swat_input: SWATInput, **kwargs) -> SWATForClsInputSubclass:
        return cls(swat_input.edge_mask, swat_input.tokens, swat_input.features, **kwargs)

    @classmethod
    def from_dir_graph(
            cls: Type[SWATForClsInputSubclass],
            graph: Union[SWADirGraph, dict],
            edge_mask_kwargs: Optional[dict] = None,
            token_kwargs: Optional[dict] = None,
            n_edge_labels: int = cons.EDGE_SIZE,
            feature_pad_idx: int = cons.PAD_TOKEN_ID,
            **kwargs
    ) -> SWATForClsInputSubclass:
        return cls.from_swat_input(
            SWATInput.from_dir_graph(
                graph,
                edge_mask_kwargs=edge_mask_kwargs,
                token_kwargs=token_kwargs,
                n_edge_labels=n_edge_labels,
                feature_pad_idx=feature_pad_idx
            ),
            **kwargs
        )


class _SWATForClsOutput(_SWATForClsIO, _SWATOutput, ABC):
    def __init__(
            self,
            edge_mask: torch.Tensor,
            tokens: torch.Tensor,
            features: torch.Tensor,
            targets: torch.Tensor,
            logits: torch.Tensor,
            embeddings: Optional[torch.Tensor] = None,
            embedding_size: Optional[int] = None,
            embedding_rank: Optional[int] = None,
            device: Union[torch.device, int, str] = 'cpu',
            _to_device=True  # for from_swat_input
    ):
        _SWATForClsIO.__init__(self, edge_mask, tokens, features, targets, device=device, _to_device=_to_device)
        assert logits.size(0) == self.batch_size
        self._sm_logits, self._preds = None, None

        if embeddings is None:
            self.embeddings = None
        else:
            if embedding_rank is None or embedding_size is None:
                raise TypeError(
                    'Parameters \'embedding_rank\' and \'embedding_size\' must be specified if '
                    '\'embeddings\' is not None'
                )

            self.embeddings = self._emb_init(embeddings, embedding_size, embedding_rank, _to_device)

        if _to_device:  # puts everything onto 'device'
            self.logits = logits.to(device=device)
        else:  # puts edge_mask, toks, feats onto device, no change to logits/embs device, targets => logits.device
            self._long_type = _long_type(logits.device)
            self.logits, self.targets = logits, targets.to(device=logits.device).type(self._long_type)

    def to(
            self,
            logit_dtype: Optional[torch.dtype] = None,
            target_dtype: Optional[torch.dtype] = None,
            embedding_dtype: Optional[torch.dtype] = None,
            token_dtype: Optional[torch.dtype] = None,
            **kwargs
    ) -> None:
        _SWATForClsIO.to(self, target_dtype=target_dtype, token_dtype=token_dtype, **kwargs)
        self.logits = self.logits.to(dtype=logit_dtype, **kwargs)

        if self.embeddings is not None:
            self.embeddings = self.embeddings.to(dtype=embedding_dtype, **kwargs)
        if self._sm_logits is not None:
            self._sm_logits = self._sm_logits.to(dtype=logit_dtype, **kwargs)
        if self._preds is not None:
            self._preds = self._preds.to(dtype=target_dtype, **kwargs)

    def clone(
            self: SWATForClsOutputSubclass,
            logit_dtype: Optional[torch.dtype] = None,
            target_dtype: Optional[torch.dtype] = None,
            embedding_dtype: Optional[torch.dtype] = None,
            **kwargs
    ) -> SWATForClsOutputSubclass:
        edge_mask, toks, feats, trgts, dev, kw = _SWATForClsIO.clone(self, **kwargs)
        logits = self.logits.to(copy=True, dtype=logit_dtype, device=dev, **kw)

        if self.embeddings is None:
            embs = None
        else:
            embs = self.embeddings.to(copy=True, dtype=embedding_dtype, device=dev, **kw)

        out_obj = type(self)(edge_mask, toks, feats, logits, trgts, embeddings=embs, device=dev)

        if self._sm_logits is not None:
            _ = out_obj.softmax_logits  # re-compute sm logits
        if self._preds is not None:
            _ = out_obj.predictions  # re-compute preds

        return out_obj

    def loss(self, **kwargs) -> torch.Tensor:
        return torch.nn.functional.cross_entropy(
            self.logits.view(self.logits.size(0) * self.logits.size(1), -1),
            self.targets.to(device=self.logits.device).view(-1),
            **kwargs
        )

    def accuracy(self, ignore_index=cons.IGNORE_ID) -> float:
        non_ignore = (self.targets.view(-1) != ignore_index).to(self.logits.device)
        acc_ten = self.predictions.view(-1) == self.targets.to(self.logits.device).view(-1)

        return (torch.sum(acc_ten & non_ignore) / torch.sum(non_ignore)).item()

    @property
    def softmax_logits(self) -> torch.Tensor:
        if self._sm_logits is None:
            self._sm_logits = self._softmax_logits_fn()

        return self._sm_logits

    @property
    def predictions(self) -> torch.Tensor:
        if self._preds is None:
            self._preds = self._predictions_fn()

        return self._preds

    def _softmax_logits_fn(self):
        return torch.nn.functional.softmax(self.logits, dim=-1)

    def _predictions_fn(self):
        return torch.argmax(self.logits, dim=-1).to(dtype=self.targets.dtype)

    def _dtypes(self):
        out_dict = {'logit_dtype': self.logits.dtype}

        if self.embeddings is not None:
            out_dict.update({'embedding_dtype': self.embeddings.dtype})

        return {**_SWATForClsIO._dtypes(self), **out_dict}

    def _get_debatch_args(self, i):
        return (
            self.edge_mask[i].unsqueeze(0),
            (
                self.tokens[i].unsqueeze(0),
                self.features[i].unsqueeze(0),
                self.targets[i].unsqueeze(0),
                self.logits[i].unsqueeze(0),
                None if self.embeddings is None else self.embeddings[i].unsqueeze(0)
            )
        )

    @classmethod
    def from_swat_output(
            cls: Type[SWATForClsOutputSubclass],
            swat_output: "SWATOutput",
            targets: torch.Tensor,
            logits: torch.Tensor,
            include_embeddings: bool = False,
            device: Optional[Union[torch.device, int, str]] = None
    ) -> SWATForClsOutputSubclass:
        return cls(
            swat_output.edge_mask,
            swat_output.tokens,
            swat_output.features,
            targets,
            logits,
            embeddings=(swat_output.embeddings if include_embeddings else None),
            device=(swat_output.device if device is None else device),
            _to_device=(device is not None)
        )


class SWATForMLMInput(_SWATForClsInput):  # input for SWATForMaskedLM model
    def __init__(
            self,
            edge_mask: torch.Tensor,
            tokens: torch.Tensor,
            features: torch.Tensor,
            targets: Union[int, torch.Tensor] = cons.IGNORE_ID,
            device: Union[torch.device, int, str] = 'cpu'
    ):
        _SWATForClsInput.__init__(self, edge_mask, tokens, features, targets, tokens.size(1), device=device)

    def mask(  # TODO: ew
            self,
            *idx: Union[int, Tuple[int], Iterable[Tuple[int]]],
            mask_id: int = cons.MASK_TOKEN_ID
    ) -> None:
        if len(idx) == 1:
            if isinstance(idx[0], tuple):
                self.mask(*idx[0], mask_id=mask_id)
            else:
                for i in idx[0]:
                    self.mask(*i, mask_id=mask_id)
        elif len(idx) == 2:  # (elem idx, token idx)
            batch_idx, tok_idx = idx

            if self.tokens[batch_idx, tok_idx] == mask_id:
                warnings.warn(
                    f'SWATForMLMInput.mask(): token {tok_idx} of batch {batch_idx} is already masked! Skipping...',
                    UserWarning
                )
            else:
                self.targets[batch_idx, tok_idx] = self.tokens[batch_idx, tok_idx]
                self.tokens[batch_idx, tok_idx] = mask_id
        else:
            raise TypeError(f'SWATForMLMInput.mask() takes 1-2 arguments ({len(idx)} given)')

    def _batch_tensors(self, out_lists, idx, max_tok, max_feat, pad_id, f_pad_id, swat_obj, ignore_id=cons.IGNORE_ID):
        size_diff_tok, device = SWATInput._batch_tensors(
            self, out_lists, idx, max_tok, max_feat, pad_id, f_pad_id, swat_obj
        )
        trgt_dtype, long_type = swat_obj.targets.dtype, swat_obj._long_type
        out_targets = self.targets.to(device=device, dtype=trgt_dtype).type(long_type)

        if size_diff_tok > 0:
            pad_trgt = torch.full((self.batch_size, size_diff_tok), ignore_id, device=device, dtype=trgt_dtype)
            out_targets = torch.cat((out_targets, pad_trgt.type(long_type)), dim=1)

        out_lists[3][idx] = out_targets

    @classmethod
    def from_swat_input(  # perturbs tokens
            cls,
            swat_input: SWATInput,
            ignore_id: int = cons.IGNORE_ID,
            mask_id: int = cons.MASK_TOKEN_ID,
            vocab_size: int = cons.VOCAB_SIZE,
            spec_tok_rng: int = cons.SPEC_TOK_RNG,
            perturb_prob: float = 0.15,
            mask_prob: float = 0.8,
            swap_prob: float = 0.1,
            spec_tok_perturb_prob: Optional[float] = 0.0,
            **kwargs
    ) -> "SWATForMLMInput":
        assert swat_input.batch_size == 1
        dtype = swat_input.tokens.dtype

        if perturb_prob == 0.0 and spec_tok_perturb_prob == 0.0:
            return cls(swat_input.edge_mask, swat_input.tokens, swat_input.features, **kwargs)

        target_toks = torch.flatten(torch.clone(swat_input.tokens))
        input_toks = torch.clone(target_toks)
        raw_perturb_probs = torch.rand(input_toks.shape, device=swat_input.device)

        if spec_tok_perturb_prob is None:
            if not 0 < perturb_prob <= 1:
                raise ValueError

            perturb_inputs, spec_toks = raw_perturb_probs < perturb_prob, None
        else:
            if not (0 <= perturb_prob <= 1 and 0 <= spec_tok_perturb_prob <= 1 and
                    0 < perturb_prob + spec_tok_perturb_prob):
                raise ValueError

            spec_toks = input_toks < spec_tok_rng
            perturb_inputs = (raw_perturb_probs < perturb_prob) & ~spec_toks

            if spec_tok_perturb_prob > 0:
                perturb_inputs = perturb_inputs | ((raw_perturb_probs < spec_tok_perturb_prob) & spec_toks)

        n_perturbed = perturb_inputs.sum().item()

        if n_perturbed > 0:
            if mask_prob == 0:
                masked_inputs = torch.full(input_toks.shape, False, dtype=torch.bool)
            else:
                masked_inputs = perturb_inputs & (torch.rand(input_toks.shape) < mask_prob)
                mask_toks = torch.full((masked_inputs.sum().item(),), mask_id, dtype=dtype, device=swat_input.device)
                input_toks[torch.nonzero(masked_inputs, as_tuple=True)] = mask_toks

            if swap_prob > 0:
                swap_prob_ = swap_prob / (1 - mask_prob)
                replaced_inputs = perturb_inputs & (torch.rand(input_toks.shape) < swap_prob_) & ~masked_inputs
                rand_toks = torch.randint(
                    spec_tok_rng,
                    vocab_size,
                    (replaced_inputs.sum().item(),),
                    dtype=dtype,
                    device=swat_input.device
                )
                input_toks[torch.nonzero(replaced_inputs, as_tuple=True)] = rand_toks
        else:  # make sure at least one target != ignore_id, else we get NaN loss
            if spec_toks is None:
                p_tok = random.randint(0, input_toks.size(0) - 1)
            else:
                spec_tok_ids = [i for i in range(input_toks.size(0)) if spec_toks[i].item()]
                nonspec_ids = [i for i in range(input_toks.size(0)) if not spec_toks[i].item()]
                n_spec = len(spec_tok_ids)

                if spec_tok_perturb_prob == 0:
                    if len(nonspec_ids) == 0:
                        raise ValueError('spec_tok_perturb_prob=0, but all input tokens are special...')

                    p_tok = random.choice(nonspec_ids)
                elif perturb_prob == 0:
                    if n_spec == 0:
                        err_str = f'perturb_prob=0 and spec_tok_perturb_prob={spec_tok_perturb_prob}, '
                        err_str += 'but all input tokens are non-special...'
                        raise ValueError(err_str)

                    p_tok = random.choice(spec_tok_ids)
                else:
                    p_spec = (spec_tok_perturb_prob / (perturb_prob + spec_tok_perturb_prob))
                    p_spec = (n_spec * p_spec) / ((n_spec * p_spec) + (len(nonspec_ids) * (1 - p_spec)))
                    p_tok = random.choice(spec_tok_ids if random.random() < p_spec else nonspec_ids)

            perturb_inputs[p_tok], n_perturbed = True, 1

            if random.random() < mask_prob:
                input_toks[p_tok] = mask_id
            elif random.random() < swap_prob:
                input_toks[p_tok] = random.randint(spec_tok_rng, vocab_size - 1)

        ignore_toks = torch.full((input_toks.size(0) - n_perturbed,), ignore_id, dtype=dtype, device=swat_input.device)
        target_toks[torch.nonzero(~perturb_inputs, as_tuple=True)] = ignore_toks
        target_toks = torch.reshape(target_toks, (1, target_toks.size(0)))
        input_toks = torch.reshape(input_toks, (1, input_toks.size(0)))

        return cls(swat_input.edge_mask, input_toks, swat_input.features, targets=target_toks, **kwargs)


# TODO: generate_batches ONLY works for iter[SWATForSCInput] (NOT graph dicts)
class SWATForSCInput(_SWATForClsInput):  # input for SWATForSequenceClassification model
    def __init__(
            self,
            edge_mask: torch.Tensor,
            tokens: torch.Tensor,
            features: torch.Tensor,
            targets: Union[int, torch.Tensor] = cons.IGNORE_ID,
            device: Union[torch.device, int, str] = 'cpu'
    ):
        _SWATForClsInput.__init__(self, edge_mask, tokens, features, targets, 1, device=device)

    def _batch_tensors(self, out_lists, idx, max_tok, max_feat, pad_id, f_pad_id, swat_obj, ignore_id=cons.IGNORE_ID):
        device = SWATInput._batch_tensors(self, out_lists, idx, max_tok, max_feat, pad_id, f_pad_id, swat_obj)[1]
        out_lists[3][idx] = self.targets.to(device=device, dtype=swat_obj.targets.dtype).type(swat_obj._long_type)


class SWATForTCInput(_SWATForClsInput):  # input for SWATForTokenClassification model
    def __init__(
            self,
            edge_mask: torch.Tensor,
            tokens: torch.Tensor,
            features: torch.Tensor,
            targets: Union[int, torch.Tensor] = cons.IGNORE_ID,
            device: Union[torch.device, int, str] = 'cpu'
    ):
        _SWATForClsInput.__init__(self, edge_mask, tokens, features, targets, tokens.size(1), device=device)

    def _batch_tensors(self, out_lists, idx, max_tok, max_feat, pad_id, f_pad_id, swat_obj, ignore_id=cons.IGNORE_ID):
        SWATForMLMInput._batch_tensors(self, out_lists, idx, max_tok, max_feat, pad_id, f_pad_id, swat_obj, ignore_id)


class SWATOutput(_SWATOutput):  # just exists to implement from_swat_input (so SWATForMLMOutput doesn't inherit it)
    def __init__(self, *args, **kwargs):
        _SWATOutput.__init__(self, *args, **kwargs)

    def _get_debatch_args(self, i):
        return (
            self.edge_mask[i].unsqueeze(0),
            tuple(map(lambda x: x[i].unsqueeze(0), (self.tokens, self.features, self.embeddings)))
        )

    @classmethod
    def from_swat_input(
            cls,
            swat_input: SWATInput,
            embeddings: torch.Tensor,
            device: Optional[Union[torch.device, int, str]] = None
    ) -> "SWATOutput":
        return cls(
            swat_input.edge_mask,
            swat_input.tokens,
            swat_input.features,
            embeddings,
            device=(swat_input.device if device is None else device),
            _to_device=(device is not None)
        )


class SWATForMLMOutput(_SWATForClsOutput):
    def __init__(
            self,
            edge_mask: torch.Tensor,
            tokens: torch.Tensor,
            features: torch.Tensor,
            targets: torch.Tensor,
            logits: torch.Tensor,
            embeddings: Optional[torch.Tensor] = None,
            device: Union[torch.device, int, str] = 'cpu',
            _to_device=True  # for from_swat_input
    ):
        _SWATForClsOutput.__init__(
            self,
            edge_mask,
            tokens,
            features,
            targets,
            logits,
            embeddings=embeddings,
            embedding_size=tokens.size(1),
            embedding_rank=3,
            device=device,
            _to_device=_to_device
        )
        assert targets.size(1) == logits.size(1) == self.token_len


# TODO: generate_batches ONLY works for iter[SWATForTCOutput] (NOT graph dicts)
class SWATForTCOutput(_SWATForClsOutput):
    def __init__(
            self,
            edge_mask: torch.Tensor,
            tokens: torch.Tensor,
            features: torch.Tensor,
            targets: torch.Tensor,
            logits: torch.Tensor,
            embeddings: Optional[torch.Tensor] = None,
            device: Union[torch.device, int, str] = 'cpu',
            _to_device=True  # for from_swat_input
    ):
        _SWATForClsOutput.__init__(
            self,
            edge_mask,
            tokens,
            features,
            targets,
            logits,
            embeddings=embeddings,
            embedding_size=tokens.size(1),
            embedding_rank=3,
            device=device,
            _to_device=_to_device
        )
        assert targets.size(1) == logits.size(1) == self.token_len
        self._bin_logits, self._bin_targets, self._bin_cls = None, None, logits.size(2) == 1

        if self._bin_cls:
            self.logits = self.logits.squeeze(-1)

    def loss(self, ignore_index=cons.IGNORE_ID, **kwargs) -> torch.Tensor:
        if self._bin_cls:
            ni_idxs = torch.nonzero(self.targets.flatten() != ignore_index).flatten()
            targets = torch.index_select(self.targets.flatten(), 0, ni_idxs).to(self.logits.device).float()
            logits = torch.index_select(self.logits.flatten(), 0, ni_idxs)

            return torch.nn.functional.binary_cross_entropy_with_logits(logits, targets, **kwargs)

        return _SWATForClsOutput.loss(self, ignore_index=ignore_index, **kwargs)

    def _softmax_logits_fn(self):
        return SWATForSCOutput._softmax_logits_fn(self)

    def _predictions_fn(self):
        return SWATForSCOutput._predictions_fn(self)


class SWATForSCOutput(_SWATForClsOutput):
    def __init__(
            self,
            edge_mask: torch.Tensor,
            tokens: torch.Tensor,
            features: torch.Tensor,
            targets: torch.Tensor,
            logits: torch.Tensor,
            embeddings: Optional[torch.Tensor] = None,
            device: Union[torch.device, int, str] = 'cpu',
            _to_device=True  # for from_swat_input
    ):
        _SWATForClsOutput.__init__(
            self,
            edge_mask,
            tokens,
            features,
            targets,
            logits,
            embeddings=embeddings,
            embedding_size=(0 if embeddings is None else embeddings.size(-1)),  # d_model
            embedding_rank=2,
            device=device,
            _to_device=_to_device
        )
        assert targets.size(1) == 1
        self._bin_cls = logits.size(1) == 1

    def loss(self, **kwargs) -> torch.Tensor:
        if self._bin_cls:
            return torch.nn.functional.binary_cross_entropy_with_logits(
                self.logits.view(-1),
                self.targets.to(device=self.logits.device).view(-1).float(),
                **kwargs
            )

        return torch.nn.functional.cross_entropy(
            self.logits,
            self.targets.to(device=self.logits.device).view(-1),
            **kwargs
        )

    def _softmax_logits_fn(self):
        if self._bin_cls:
            return torch.nn.functional.sigmoid(self.logits)

        return _SWATForClsOutput._softmax_logits_fn(self)

    def _predictions_fn(self):
        if self._bin_cls:
            return (self.logits >= 0.0).to(dtype=self.targets.dtype)  # sigmoid(0.0) = 0.5

        return _SWATForClsOutput._predictions_fn(self)


def _generate_batches_wrapper(as_generator, *args):
    return (lambda x: x if as_generator else list(x))(_generate_batches(*args))


def _generate_batches(
        cls, inputs, batch_size, graph_parser, swat_input_kwargs, batch_kwargs, collect, sparse_mask,
        to_after_batch_kwargs, max_seq_len
):
    assert batch_size > 0
    swa_kwargs = cons.dict_kwargs(swat_input_kwargs)
    batch_kwargs, batch_inputs, max_tok, max_feat = cons.dict_kwargs(batch_kwargs), [], 0, 0

    if max_seq_len is None:
        max_seq_len = float('inf')
    else:
        assert max_seq_len > 0

    def finalize_mem_0():
        batch_inputs.clear()

    if collect:
        def finalize_mem_1():
            finalize_mem_0()
            gc.collect()

        if torch.cuda.is_available():
            def finalize_mem():
                finalize_mem_1()
                torch.cuda.empty_cache()
        else:
            finalize_mem = finalize_mem_1
    else:
        finalize_mem = finalize_mem_0

    if to_after_batch_kwargs is not None and len(to_after_batch_kwargs) > 0:
        def finalize_input_0(swat_input_):
            swat_input_.to(**to_after_batch_kwargs)
    else:
        finalize_input_0 = lambda _: None
    if sparse_mask is False:
        finalize_input = finalize_input_0
    else:
        sparse_kwargs = {} if sparse_mask is True else sparse_mask

        def finalize_input(swat_input_):
            finalize_input_0(swat_input_)
            swat_input_.to_sparse_mask(**sparse_kwargs)

    def finalize(swat_input_):
        finalize_mem()
        finalize_input(swat_input_)

        return swat_input_

    for grph in (inputs if graph_parser is None else graph_parser(inputs, save=False)):  # main loop
        if grph is not None:  # only matters if mrs_graph_parser is not None
            try:
                if isinstance(grph, cls):
                    grph.to(**swa_kwargs)
                    swat_input = grph
                elif isinstance(grph, SWATInput):
                    swat_input = cls.from_swat_input(grph, **swa_kwargs)
                else:
                    swat_input = cls.from_dir_graph(grph, **swa_kwargs)
            except ValueError:  # graphs w/ 0 toks => ValueError
                continue

            if swat_input.token_len <= max_seq_len:
                max_tok, max_feat = max(max_tok, swat_input.token_len), max(max_feat, swat_input.feature_len)

                if len(batch_inputs) == batch_size - 1:
                    yield finalize(swat_input.batch(batch_inputs, _max_tok_feat=(max_tok, max_feat), **batch_kwargs))

                    max_tok, max_feat = 0, 0
                else:
                    batch_inputs.append(swat_input)

    if len(batch_inputs) > 0:
        yield finalize(batch_inputs[0].batch(batch_inputs[1:], _max_tok_feat=(max_tok, max_feat), **batch_kwargs))


def _long_type(device):
    if device == 'cpu' or (isinstance(device, torch.device) and device.index is None):
        return torch.LongTensor

    return torch.cuda.LongTensor
