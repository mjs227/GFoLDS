
import os
import torch
import pickle
import torch.nn as nn
import model._constants as cons
import model.configs._configs as model_config
from copy import deepcopy
from abc import ABC, abstractmethod
from typing import Union, Optional, Dict, Tuple, TypeVar, Type
from model.io._io import (
    SWATInput,
    SWATOutput,
    SWATForTCInput,
    SWATForTCOutput,
    SWATForSCInput,
    SWATForSCOutput,
    SWATForMLMInput,
    SWATForMLMOutput,
    SWATForClsInputSubclass,
    SWATForClsOutputSubclass
)


# TODO: implement output_attentions and output_hidden_states (esp. for ch. 6)
# TODO: replace assertions with exceptions (whole project)
# TODO: docstrings (whole project)
# TODO: nn.Identity() (whole project)
# TODO: A.__class__.__name__ (whole project)
SWATTopModuleSubclass = TypeVar('SWATTopModuleSubclass', bound="_SWATTopModule")
SWATWithClassificationHeadSubclass = TypeVar('SWATWithClassificationHeadSubclass', bound="_SWATWithClassificationHead")


# base class: implements to() and _set_device()
class _SWATModule(nn.Module):
    def __init__(self):
        super(_SWATModule, self).__init__()
        self.device = 'cpu'

    def to(self, *args, **kwargs) -> None:
        super(_SWATModule, self).to(*args, **kwargs)
        d = kwargs.get('device', next((a for a in args if not isinstance(a, torch.dtype)), None))
        self._set_device(('cpu' if d.index is None else d.index) if isinstance(d, torch.device) else d)

    def _set_device(self, device):
        if device is not None:
            for attr in map(lambda x: getattr(self, x), dir(self)):
                if isinstance(attr, _SWATModule):
                    attr._set_device(device)
                elif isinstance(attr, nn.ModuleList):
                    _set_module_list_device(attr, device)

            self.device = device


# base class for SWATransformer, SWATForSequenceClassification, SWATForMaskedLM:
# implements to(), module_param_count(), save(), from_pretrained(), from_config()
class _SWATTopModule(_SWATModule):
    def __init__(self, config: model_config.SWATTopConfig, _fpt=False):
        if not _fpt:
            self_type = cons.type_str(self)
            raise EnvironmentError(
                f'Instantiate using the \'{self_type}.from_pretrained()\' or \'{self_type}.from_config()\' methods'
            )

        super(_SWATTopModule, self).__init__()
        self.config = config

    def to(self, *args, **kwargs) -> None:
        device_idx, device = next(
            iter((
                (i, arg) for i, arg in enumerate(args) if
                isinstance(arg, dict) or isinstance(arg, torch.device) or isinstance(arg, int) or arg == 'cpu'
            )),
            (None, None)
        )
        device = kwargs.pop('device', None) if device is None else device
        args = args if device_idx is None else (args[:device_idx] + args[device_idx + 1:])

        if isinstance(device, dict) and '' in device.keys():
            assert len(device.keys()) == 1 and not isinstance(device[''], dict)
            device = device['']
        if not isinstance(device, dict):
            super(_SWATTopModule, self).to(*args, device=device, **kwargs)
            return

        assert len(args) == 0 or (len(args) == 1 and isinstance(args[0], torch.dtype))
        self.device, self_attr_keys = device, set()

        for attr_dev, attr_name, attr in map(lambda a: (device.get(a, None), a, getattr(self, a)), dir(self)):
            if isinstance(attr, nn.Module):
                attr.to(*args, device=attr_dev, **kwargs)
            elif isinstance(attr, nn.ModuleList):
                if isinstance(attr_dev, dict):
                    assert set(attr_dev.keys()) == set(range(len(attr)))
                    attr_dev_iter = attr_dev.items()
                else:
                    attr_dev_iter = ((i, attr_dev) for i in range(len(attr)))

                for k, v in attr_dev_iter:
                    attr[k].to(*args, device=v, **kwargs)
            else:
                continue

            assert attr_dev is not None, str(attr)
            self_attr_keys.add(attr_name)

        assert set(device.keys()) == self_attr_keys, str(set(device.keys()) - self_attr_keys)

    def module_param_count(self, proportion: bool = False, **kwargs) -> "SWATParamCount":
        return (SWATParamProp if proportion else SWATParamCount).from_swat_model(self, **kwargs)

    def save(self, *filepath: str) -> Optional[Tuple[Dict[str, torch.Tensor], model_config.SWATTopConfig]]:
        if len(filepath) == 1:
            with open(os.path.abspath(filepath[0]), 'wb') as f:
                pickle.dump(self.save(), f)
        elif len(filepath) == 0:
            device = self.device
            self.to('cpu')
            state_dict = deepcopy(self.state_dict())
            self.to(device=device)

            return state_dict, self.config
        else:
            raise TypeError(f'{cons.type_str(self)}.save() takes <= 1 arguments ({len(filepath)} given).')

    def load_checkpoint(
            self,
            checkpoint: Union[str, Tuple[Dict[str, torch.Tensor], model_config.SWATTopConfig], Dict[str, torch.Tensor]]
    ) -> None:
        device = self.device
        self.to(device='cpu')

        if isinstance(checkpoint, str):
            with open(os.path.abspath(checkpoint), 'rb') as f:
                checkpoint = pickle.load(f)

        self.load_state_dict(checkpoint[0] if isinstance(checkpoint, tuple) else checkpoint)
        self.to(device=device)

    @classmethod
    def from_config(
            cls: Type[SWATTopModuleSubclass],
            config: model_config.SWATTopConfig,
            **kwargs
    ) -> SWATTopModuleSubclass:
        model = cls(config, _fpt=True)
        model.to(**kwargs)

        return model

    @classmethod
    def from_pretrained(cls: Type[SWATTopModuleSubclass], filepath: str, **kwargs) -> SWATTopModuleSubclass:
        with open(os.path.abspath(filepath), 'rb') as f:
            state_dict, config = pickle.load(f)

        model = cls.from_config(config)
        model.load_state_dict(state_dict)
        model.to(**kwargs)

        return model


class _SWATWithClassificationHead(_SWATTopModule, ABC):
    def __init__(self, config: model_config.SWATWithClsHeadConfig, _swat=None, **kwargs):
        super(_SWATWithClassificationHead, self).__init__(config, **kwargs)
        self.swat = SWATransformer(config.swat_config, _fpt=True) if _swat is None else _swat
        n_cls = config.n_classes if config.n_classes > 2 else 1

        if isinstance(config.head_config, str):  # config.head_config == 'linear'
            self.head = nn.Linear(config.swat_config.d_model, n_cls)
            self._head_device = self._head_device_linear
        else:
            self.head = SWATFeedForward(config.swat_config.d_model, config.d_hidden, n_cls, config.head_config)
            self._head_device = self._head_device_ff

    def forward(
            self,
            x: SWATForClsInputSubclass,
            embeddings: bool = False,
            pad_id: int = cons.PAD_TOKEN_ID
    ) -> SWATForClsOutputSubclass:
        swat_out = self.swat(x, pad_id=pad_id)

        return self.forward_head(x, swat_out, embeddings, pad_id)

    def head_device(self):
        return self._head_device()

    def _head_device_linear(self):
        return self.head.weight.device

    def _head_device_ff(self):
        return self.head.device

    @abstractmethod
    def forward_head(
            self,
            x: SWATForClsInputSubclass,
            swat_out: SWATOutput,
            embeddings: bool,
            pad_id: int
    ) -> SWATForClsOutputSubclass:
        pass

    @classmethod
    def from_swat_model(
            cls: Type[SWATWithClassificationHeadSubclass],
            swat_model: Union[SWATTopModuleSubclass, str],
            config: model_config.SWATWithClsHeadConfig,
            **kwargs
    ) -> SWATWithClassificationHeadSubclass:
        swat = SWATransformer.from_swat_module(swat_model)
        config.swat_config = swat.config
        model = cls(config, _swat=swat, _fpt=True)
        model.to(**kwargs)

        return model


class SWATransformer(_SWATTopModule):
    def __init__(self, config: model_config.SWATConfig, **kwargs):
        super(SWATransformer, self).__init__(config, **kwargs)
        self.embedding_module = SWATEmbeddingModule(config.d_model, config.embedding_config)
        self.encoder_layers = nn.ModuleList([SWATEncoderLayer(config.d_model, cfg) for cfg in config.encoder_config])

    def forward(self, x: SWATInput, pad_id: int = cons.PAD_TOKEN_ID) -> SWATOutput:
        pad_mask = x.tokens == pad_id
        x_embs = self.embedding_module(x)

        for layer in self.encoder_layers:
            pad_mask = pad_mask.to(device=layer.device)
            x_embs = layer(x_embs.to(device=layer.device), key_padding_mask=pad_mask)

        return SWATOutput.from_swat_input(x, x_embs)

    @classmethod
    def from_swat_module(cls, swat_model: Union[SWATTopModuleSubclass, str]) -> "SWATransformer":
        if isinstance(swat_model, str):
            try:
                with open(os.path.abspath(swat_model), 'rb') as f:
                    state_dict, config = pickle.load(f)
            except ValueError:
                raise ValueError(f'Invalid SWAT model file format: \'{swat_model}\'')

            if isinstance(config, model_config.SWATConfig):
                swat_model = SWATransformer.from_config(config)
            elif isinstance(config, model_config.SWATForMaskedLMConfig):
                swat_model = SWATForMaskedLM.from_config(config)
            elif isinstance(config, model_config.SWATForSequenceClassificationConfig):
                swat_model = SWATForSequenceClassification.from_config(config)
            elif isinstance(config, model_config.SWATForTokenClassificationConfig):
                swat_model = SWATForTokenClassification.from_config(config)
            else:
                raise ValueError(f'Invalid SWAT model config class: {type(config).__name__}')

            swat_model.load_state_dict(state_dict)
        if type(swat_model) is cls:
            return swat_model

        swat_list = [y for y in (getattr(swat_model, x) for x in dir(swat_model)) if type(y) is cls]

        if len(swat_list) == 1:
            return swat_list[0]

        raise ValueError(f'Invalid SWAT model: {type(swat_model).__name__}')


class SWATForMaskedLM(_SWATWithClassificationHead):
    def __init__(self, config: model_config.SWATForMaskedLMConfig, **kwargs):
        super(SWATForMaskedLM, self).__init__(config, **kwargs)

    def forward_head(self, x: SWATForMLMInput, swat_out: SWATOutput, embeddings: bool, _):
        logits = self.head(swat_out.embeddings.to(device=self.head_device()))

        return SWATForMLMOutput.from_swat_output(swat_out, x.targets, logits, include_embeddings=embeddings)

    @classmethod
    def from_prev_version(cls, filepath, **kwargs):
        from collections import OrderedDict

        with open(os.path.abspath(filepath), 'rb') as f:
            state_dict, config = pickle.load(f)

        new_sd = OrderedDict()

        for k, v in state_dict.items():
            if k.startswith('decoder.'):
                new_sd.update({k.replace('decoder.', 'head.'): v})
            else:
                new_sd.update({k: v})

        new_config = model_config.SWATForMaskedLMConfig(
            head_config__activ_fn=config.decoder_config.activ_fn,
            head_config__layer_norm=config.decoder_config.layer_norm,
            head_config__layer_norm_kwargs=config.decoder_config.layer_norm_kwargs,
            head_config__dropout_kwargs=config.decoder_config.dropout_kwargs,
            swat_config=config.swat_config
        )
        model = cls.from_config(new_config)
        model.load_state_dict(new_sd)
        model.to(**kwargs)

        return model


class SWATForTokenClassification(_SWATWithClassificationHead):
    def __init__(self, config: model_config.SWATForTokenClassificationConfig, **kwargs):
        super(SWATForTokenClassification, self).__init__(config, **kwargs)

    def forward_head(self, x: SWATForTCInput, swat_out: SWATOutput, embeddings: bool, _) -> SWATForTCOutput:
        logits = self.head(swat_out.embeddings.to(device=self.head_device()))

        return SWATForTCOutput.from_swat_output(swat_out, x.targets, logits, include_embeddings=embeddings)


class SWATForSequenceClassification(_SWATWithClassificationHead):
    def __init__(self, config: model_config.SWATForSequenceClassificationConfig, **kwargs):
        super(SWATForSequenceClassification, self).__init__(config, **kwargs)

        if config.pooling == 'mean':
            def pool_fn(model_output, pad_id, device):
                mask = (model_output.tokens != pad_id).to(device)
                embs = model_output.embeddings.to(device)

                return torch.sum(embs * mask[..., None], dim=1) / torch.clamp(mask.sum(dim=1)[..., None], min=1e-9)
        elif config.pooling == 'first':
            def pool_fn(model_output, _, device):
                return model_output.embeddings[:, 0, ...].to(device)
        else:
            raise ValueError(f'\'{config.pooling}\' is not a valid pooling function specification')

        self._pool_fn = pool_fn

    def forward_head(self, x: SWATForSCInput, swat_out: SWATOutput, embeddings: bool, pad_id: int):
        embs_pooled = self._pool_fn(swat_out, pad_id, self.head_device())
        logits = self.head(embs_pooled)

        return SWATForSCOutput.from_swat_output(swat_out, x.targets, logits, include_embeddings=embeddings)


class MaskDropout(_SWATModule):
    def __init__(self, config: model_config.MaskDropoutConfig):
        super(MaskDropout, self).__init__()
        self._p, self._default_val = 1 - config.p, -config.default_val
        self._dropout_fn = self._id_fn if config.p == 0 else (self._dropout_ip if config.inplace else self._dropout)
        self._default_val_fn = self._id_fn if config.default_val == 0 else self._fill_default_val
        self._forward_fn = self._dropout_fn if self.training else self._id_fn

    def train(self, *args, **kwargs) -> None:
        super(MaskDropout, self).train(*args, **kwargs)
        self._forward_fn = self._dropout_fn

    def eval(self) -> None:
        super(MaskDropout, self).eval()
        self._forward_fn = self._id_fn

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self._forward_fn(x)

    def _id_fn(self, x, *_):
        return x

    def _dropout_ip(self, x):
        dropout_tensor = self._get_dropout_tensor(x)
        x.mul_(dropout_tensor)

        return self._default_val_fn(x, dropout_tensor)

    def _dropout(self, x):
        dropout_tensor = self._get_dropout_tensor(x)

        return self._default_val_fn(x * dropout_tensor, dropout_tensor)

    def _get_dropout_tensor(self, x):
        return (torch.rand(x.shape, device=x.device) < self._p).to(dtype=x.dtype)

    def _fill_default_val(self, x, dropout_tensor):
        dropout_tensor.add_(-1)
        dropout_tensor.mul_(self._default_val)
        x.add_(dropout_tensor)

        return x


class SWATFeedForward(_SWATModule):
    def __init__(self, input_size: int, hidden_size: int, output_size: int, config: model_config.SWATFeedForwardConfig):
        super(SWATFeedForward, self).__init__()
        self.linear1 = nn.Linear(input_size, hidden_size)
        self.linear2 = nn.Linear(hidden_size, output_size)

        if config.activ_fn is None:
            self.activ_fn, self._linear1_1 = None, self.linear1
        else:
            self.activ_fn, self._linear1_1 = config.activ_fn, self._linear1_activ

        if config.layer_norm:
            self.layer_norm = nn.LayerNorm(hidden_size, **config.layer_norm_kwargs)
            self._linear1_2 = self._linear1_layer_norm
        else:
            self.layer_norm, self._linear1_2 = None, self._linear1_1

        if config.dropout_kwargs.get('p', 0.0) == 0.0:
            self.dropout, self._linear1_3 = None, self._linear1_2
        else:
            self.dropout = nn.Dropout(**config.dropout_kwargs)
            self._linear1_3 = self._linear1_dropout

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self._linear1_3(x)

        return self.linear2(x)

    def _linear1_activ(self, x):
        return self.activ_fn(self.linear1(x))

    def _linear1_layer_norm(self, x):
        return self.layer_norm(self._linear1_1(x))

    def _linear1_dropout(self, x):
        return self.dropout(self._linear1_2(x))


class SWATEmbeddingModule(_SWATModule):
    def __init__(self, d_model: int, config: model_config.SWATEmbeddingConfig):
        super(SWATEmbeddingModule, self).__init__()
        self.feature_pad_idx = config.feature_pad_idx

        self.token_embedding_layer = nn.Embedding(config.vocab_size, d_model, **config.token_embedding_kwargs)
        self.feature_embedding_layer = nn.Embedding(config.feature_size, d_model, **config.feature_embedding_kwargs)
        self.feature_layer_norm = nn.LayerNorm(d_model, **config.feature_layer_norm_kwargs)
        self.feature_dropout = MaskDropout(config.feature_dropout_config)
        self.edge_dropout = MaskDropout(config.edge_dropout_config)
        self.embedding_dropout = nn.Dropout(**config.embedding_dropout_kwargs)
        self.embedding_layer_norm = nn.LayerNorm(config.d_swa, **config.embedding_layer_norm_kwargs)

        self.swa_layers = nn.ModuleList([SWALayer(config.edge_size, config.d_swa, c) for c in config.swa_layer_config])
        self.swa_in_proj_layer = nn.Linear(d_model, config.d_swa, **config.swa_in_projection_kwargs)
        self.swa_out_proj_layer = nn.Linear(config.d_swa, d_model, **config.swa_out_projection_kwargs)

    def forward(self, x: SWATInput) -> torch.Tensor:
        x_feats = self.feature_dropout(x.features.to(self.device))
        feat_mask = x_feats != self.feature_pad_idx
        emb_feat = self.feature_embedding_layer(x_feats) * feat_mask[..., None]
        emb_feat = self.feature_layer_norm(torch.sum(emb_feat, dim=2))
        emb_tok_feat = self.token_embedding_layer(x.tokens.to(self.device)) + emb_feat

        emb_swa = self.swa_in_proj_layer(emb_tok_feat)
        mask_forward = self.edge_dropout(x.edge_mask.to(dtype=emb_swa.dtype, device=self.device))
        mask_backward = torch.transpose(mask_forward, -2, -1)

        for layer in self.swa_layers:
            emb_swa = layer(emb_swa, mask_forward, mask_backward)

        emb_swa = self.embedding_layer_norm(emb_swa)
        emb_swa = self.swa_out_proj_layer(emb_swa)
        emb_out = emb_tok_feat + emb_swa  # "pseudo" skip conn.: "emb_tok + emb_pos"

        return self.embedding_dropout(emb_out)


class SWALayer(_SWATModule):
    def __init__(self, n_edge_labels: int, d_swa: int, config: model_config.SWALayerConfig):
        super(SWALayer, self).__init__()
        self.feed_forward = SWATFeedForward(d_swa, config.d_ff, d_swa, config.feed_forward_config)
        self.swa_forward = SWAModule(n_edge_labels, d_swa, config.swa_config[0])
        self.swa_backward = SWAModule(n_edge_labels, d_swa, config.swa_config[1])

        self.swa_layer_norm = nn.LayerNorm(d_swa, **config.swa_layer_norm_kwargs)
        self.dropout = nn.Dropout(**config.dropout_kwargs)

    def forward(self, x: torch.Tensor, mask_f: torch.Tensor, mask_b: torch.Tensor) -> torch.Tensor:
        x_ln = self.swa_layer_norm(x)
        x_f = self.swa_forward(x_ln, mask_f)
        x_b = self.swa_backward(x_ln, mask_b)

        x_swa = self.feed_forward(x_f + x_b)
        x_swa = self.dropout(x_swa)

        return x + x_swa  # skip connection


class SWAModule(_SWATModule):
    def __init__(self, n_edge_labels: int, d_swa: int, config: model_config.SWAModuleConfig):
        super(SWAModule, self).__init__()
        self.n_edge_labels, self.d_swa = n_edge_labels, d_swa
        self.edge_projection = nn.Linear(d_swa, n_edge_labels * d_swa)
        self.dropout = nn.Dropout(**config.dropout_kwargs)
        self.layer_norm = nn.LayerNorm(d_swa, **config.layer_norm_kwargs)

    def forward(self, x: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        x = self.edge_projection(x)
        x = x.view(x.size(0), x.size(1), self.n_edge_labels, self.d_swa)
        x = torch.permute(x, (0, 2, 1, 3))
        x = self.dropout(x)

        x = torch.matmul(mask, x)
        x = torch.permute(x, (0, 2, 1, 3))
        x = torch.sum(x, dim=2)

        return self.layer_norm(x)


class SWATEncoderLayer(_SWATModule):
    def __init__(self, d_model: int, config: model_config.SWATEncoderConfig):
        super(SWATEncoderLayer, self).__init__()
        self.mha = nn.MultiheadAttention(d_model, config.num_heads, batch_first=True, **config.mha_kwargs)
        self.mha_dropout = nn.Dropout(**config.mha_dropout_kwargs)
        self.mha_layer_norm = nn.LayerNorm(d_model, **config.mha_layer_norm_kwargs)

        self.feed_forward = SWATFeedForward(d_model, config.d_ff, d_model, config.feed_forward_config)
        self.feed_forward_dropout = nn.Dropout(**config.feed_forward_dropout_kwargs)
        self.feed_forward_layer_norm = nn.LayerNorm(d_model, **config.feed_forward_layer_norm_kwargs)

    def forward(self, x: torch.Tensor, **kwargs) -> torch.Tensor:
        x_mha = self.mha_layer_norm(x)
        x_mha = self.mha(x_mha, x_mha, x_mha, need_weights=False, **kwargs)[0]
        x_mha = self.mha_dropout(x_mha)
        x = x + x_mha  # skip connection

        x_ff = self.feed_forward_layer_norm(x)
        x_ff = self.feed_forward(x_ff)
        x_ff = self.feed_forward_dropout(x_ff)

        return x + x_ff  # skip connection


class SWATParamCount:
    def __init__(self, *kv, top: bool = False):
        super(SWATParamCount, self).__init__()
        self.top, self.total, self._udict = top, 0, {}

        if len(kv) == 1:
            self._udict.update(kv[0])
        elif not len(kv) == 0:
            raise TypeError(f'SWATParamCount.__init__() takes <= 1 arguments ({len(kv)} given).')

    def items(self):
        return self._udict.items()

    def values(self):
        return self._udict.values()

    def keys(self):
        return self._udict.keys()

    def __getitem__(self, key):
        return dict.__getitem__(self._udict, key)

    def __len__(self):
        return len(self._udict)

    def __str__(self):
        return self._print_str(0)

    def _out_str(self):
        return f'{"model" if self.top else ""}={self.total}: '

    def _print_str(self, n):
        if len(self) == 0:
            return self._out_str() + '{}'
        if len(self) == 1:
            self_key, self_value = next(iter(self.items()))

            if not isinstance(self_key, SWATParamCount):
                return self._out_str() + '{' + str(self_key) + ': ' + str(self_value) + '}'

        print_strs, indent = [], ' ' * (n + cons.INDENT_FACTOR)

        for k, v in self.items():
            if isinstance(v, SWATParamCount):
                print_strs.append(f'{k}{v._print_str(n + cons.INDENT_FACTOR)},')
            else:
                print_strs.append(f'{k}: {v},')

        return self._out_str() + '{\n' + indent + (f'\n{indent}'.join(print_strs)[:-1]) + '\n' + (' ' * n) + '}'

    @classmethod
    def from_swat_model(cls, swat_model: _SWATTopModule, _top=True):
        out_cls = cls(top=_top)

        for attr_name, attr in map(lambda a: (a, getattr(swat_model, a)), dir(swat_model)):
            if isinstance(attr, _SWATTopModule):
                out_cls._udict.update({attr_name: SWATParamCount.from_swat_model(attr, _top=False)})
            elif isinstance(attr, _SWATModule):
                out_cls._udict.update({attr_name: sum(p.numel() for p in attr.parameters())})
            elif isinstance(attr, nn.ModuleList):
                out_cls._udict.update({
                    attr_name: SWATParamCount({i: sum(p.numel() for p in x.parameters()) for i, x in enumerate(attr)})
                })
                out_cls[attr_name].total = sum(out_cls[attr_name].values(), start=0)

        out_cls.total = sum(v.total if isinstance(v, SWATParamCount) else v for v in out_cls.values())

        return out_cls


class SWATParamProp(SWATParamCount):
    def __init__(self, _fpt=False):
        if not _fpt:
            raise EnvironmentError(
                'Instantiate using the \'SWATParamProp.from_param_count()\' or \'SWATParamProp.from_swat_model()\''
                ' methods'
            )

        super(SWATParamProp, self).__init__()
        self._pdict, self._total = {}, 0.0

    def round(self, factor: Optional[int] = 5):
        if factor is None:
            round_fn = lambda x: x
        else:
            assert factor > 0
            round_fn = lambda x: round(x, factor)

        self.total = round_fn(self._total)

        for k, v in self._pdict.items():
            if isinstance(v, SWATParamProp):
                self._udict.update({k: v})
                v.round(factor=factor)
            else:
                self._udict.update({k: round_fn(v)})

    def unround(self):
        self.round(factor=None)

    def _out_str(self):
        return ('model' if self.top else f'={self.total}') + ': '

    @classmethod
    def from_param_count(cls, p: SWATParamCount, round_factor: Optional[int] = None):
        out_cls, total = cls(_fpt=True), p.total

        def param_prop_from_param_count(pp, pc):
            pp._total = pc.total / total

            for k, v in pc._udict.items():
                if isinstance(v, SWATParamCount):
                    pp_ = SWATParamProp(_fpt=True)
                    param_prop_from_param_count(pp_, v)
                    pp._pdict.update({k: pp_})
                else:
                    pp._pdict.update({k: v / total})

        param_prop_from_param_count(out_cls, p)
        out_cls.round(factor=round_factor)
        out_cls.top = True

        return out_cls

    @classmethod
    def from_swat_model(cls, swat_model: _SWATTopModule, **kwargs):
        return cls.from_param_count(SWATParamCount.from_swat_model(swat_model), **kwargs)


def _set_module_list_device(module_list, device):
    for mod in module_list:
        if isinstance(mod, nn.ModuleList):
            _set_module_list_device(mod, device)
        elif isinstance(mod, _SWATModule):
            mod._set_device(device)
