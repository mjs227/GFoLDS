
import torch
import inspect
import model._constants as cons
from copy import deepcopy
from re import search, findall
from collections.abc import Iterable
from model.configs._get_option_type import got
from typing import Optional, Union, get_type_hints


class _Config:  # base class: implements __str__() and _sub_check()
    def __init__(self, **kwargs):
        sub_kwargs, kwarg_types = _get_kwarg_types(type(self))
        sub_kwargs = {k: (v, {}) for k, v in sub_kwargs.items()}

        for k, v in kwargs.items():  # sorting "__"-notation kwargs by top-level kwargs
            if '__' not in k:
                raise TypeError(
                    f'{cons.type_str(self, incl_path=True)}.__init__() got an unexpected keyword argument \'{k}\''
                )

            k_sup, k_sub = k.split('__', maxsplit=1)

            if k_sup not in kwarg_types:
                raise TypeError(
                    f'{cons.type_str(self, incl_path=True)}.__init__() got an unexpected keyword argument \'{k_sup}\''
                )
            if k_sup not in sub_kwargs.keys():
                raise TypeError(
                    f'(\'{k}={v}\'): Only dictionary- or _Config-subclass-valued kwargs may be specified using '
                    f'\'__\' notation in {cons.type_str(self, incl_path=True)}.__init__()'
                )

            sub_kwargs[k_sup][1].update({k_sub: v})

        for k, (v_type, v_kwargs) in sub_kwargs.items():  # instatiating attributes specified by "__"-notation kwargs
            if len(v_kwargs) > 0:
                setattr(self, k, got(v_type)(**v_kwargs))

    def _sub_check(self, var_kw, var_sub, cls):
        if hasattr(self, var_sub):  # config class attrs specified in kwargs
            assert var_kw is None, var_sub

            return getattr(self, var_sub)

        if var_kw is None:  # default
            return cls()

        return var_kw  # actual config class specified in kwargs

    def __str__(self):
        return self._print_str(0)

    def _print_str(self, n):
        n_, self_str = n + cons.INDENT_FACTOR, cons.type_str(self)
        indent, ld_indent = ' ' * n_, ' ' * (n_ + cons.INDENT_FACTOR)
        top_strs, dict_strs, cfg_strs = [], [], []

        for attr_name in dir(self):
            if not attr_name[0] == '_':
                attr_val, attr_str = getattr(self, attr_name), f'\n{indent}{attr_name}='

                if isinstance(attr_val, _Config):
                    cfg_strs.append(attr_str + attr_val._print_str(n_))
                elif isinstance(attr_val, list):
                    if len(attr_val) == 0:
                        top_strs.append(attr_str + '[]')
                    else:
                        attr_str += '['
                        has_cfg = False

                        for i, cfg in enumerate(attr_val):
                            attr_str += f'\n{ld_indent}({i})='

                            if isinstance(cfg, _Config):
                                attr_str += cfg._print_str(n_ + cons.INDENT_FACTOR)
                                has_cfg = True
                            elif isinstance(cfg, dict):
                                attr_str += _print_dict(cfg, n_ + cons.INDENT_FACTOR)
                            else:
                                attr_str += cons.str_(cfg)

                        (cfg_strs if has_cfg else top_strs).append(attr_str + f'\n{indent}]')
                elif isinstance(attr_val, dict):
                    dict_strs.append(attr_str + _print_dict(attr_val, n_))
                else:
                    top_strs.append(attr_str + cons.str_(attr_val))

        return f'{self_str}: ' + (''.join(top_strs)) + (''.join(dict_strs)) + (''.join(cfg_strs))


# super class for SWAT top-level modules (SWATransformer, SWATWithHead, etc.)
# basically only exists to simplify typing declarations
class SWATTopConfig(_Config):
    def __init__(self, **kwargs):
        super(SWATTopConfig, self).__init__(**kwargs)


class SWATWithClsHeadConfig(SWATTopConfig):
    def __init__(
            self,
            n_classes: int = 2,
            d_hidden: int = -1,
            linear_head: bool = False,
            head_config: Optional["SWATFeedForwardConfig"] = None,
            swat_config: Optional["SWATConfig"] = None,
            **kwargs
    ):
        super(SWATWithClsHeadConfig, self).__init__(**kwargs)
        assert n_classes >= 2
        self.swat_config, self.n_classes = self._sub_check(swat_config, 'swat_config', SWATConfig), n_classes

        if linear_head:
            assert d_hidden == -1
            assert head_config is None
            assert all(not k.startswith('head_config') for k in kwargs.keys())
            self.head_config, self.d_hidden = 'linear', 'n/a'
        else:
            assert d_hidden == -1 or d_hidden > 0

            if d_hidden == -1:
                self.d_hidden = self.swat_config.d_model
            elif d_hidden > 0:
                self.d_hidden = d_hidden
            else:
                raise ValueError('\'d_hidden\' must be -1 (default) or > 0')

            self.head_config = self._sub_check(head_config, 'head_config', SWATDecoderConfig)


class SWATForMaskedLMConfig(SWATWithClsHeadConfig):
    def __init__(self, head_config: Optional["SWATDecoderConfig"] = None, **kwargs):
        assert 'd_hidden' not in kwargs.keys()
        assert 'n_classes' not in kwargs.keys()

        super(SWATForMaskedLMConfig, self).__init__(head_config=head_config, **kwargs)
        self.n_classes = self.swat_config.embedding_config.vocab_size
        self.d_hidden = self.swat_config.d_model


class SWATForSequenceClassificationConfig(SWATWithClsHeadConfig):
    def __init__(self, pooling: str = 'mean', **kwargs):
        super(SWATForSequenceClassificationConfig, self).__init__(**kwargs)
        self.pooling = pooling


class SWATForTokenClassificationConfig(SWATWithClsHeadConfig):
    def __init__(self, **kwargs):
        super(SWATForTokenClassificationConfig, self).__init__(**kwargs)


class SWATConfig(SWATTopConfig):
    def __init__(
            self,
            d_model: int = cons.DEFAULT_D_MODEL,
            n_encoder_layers: int = cons.DEFAULT_N_ENC,
            embedding_config: Optional["SWATEmbeddingConfig"] = None,
            encoder_config: Optional[Union["SWATEncoderConfig", Iterable["SWATEncoderConfig"]]] = None,
            **kwargs
    ):
        super(SWATConfig, self).__init__(**kwargs)
        assert n_encoder_layers > 0
        assert d_model > 0

        self.d_model = d_model
        self.embedding_config = self._sub_check(embedding_config, 'embedding_config', SWATEmbeddingConfig)
        self.encoder_config = _module_list_init(
            self._sub_check(encoder_config, 'encoder_config', SWATEncoderConfig),
            n_encoder_layers,
            d_ff=(d_model * 2)
        )


class SWATEncoderConfig(_Config):  # TODO: "SWATEncoderLayerConfig"
    def __init__(
            self,
            num_heads: int = cons.DEFAULT_N_HEADS,
            d_ff: int = -1,  # -1 = default (=> d_model * 2)
            feed_forward_config: Optional["SWATFeedForwardConfig"] = None,
            mha_kwargs: Optional[dict] = None,
            mha_layer_norm_kwargs: Optional[dict] = None,
            mha_dropout_kwargs: Optional[dict] = None,
            feed_forward_layer_norm_kwargs: Optional[dict] = None,
            feed_forward_dropout_kwargs: Optional[dict] = None,
            **kwargs
    ):
        super(SWATEncoderConfig, self).__init__(**kwargs)
        assert num_heads > 0
        assert d_ff == -1 or d_ff > 0
        self.num_heads, self.d_ff = num_heads, d_ff

        self.mha_kwargs = self._sub_check(mha_kwargs, 'mha_kwargs', dict)
        self.mha_layer_norm_kwargs = self._sub_check(mha_layer_norm_kwargs, 'mha_layer_norm_kwargs', dict)
        self.mha_dropout_kwargs = self._sub_check(mha_dropout_kwargs, 'mha_dropout_kwargs', dict)
        self.mha_dropout_kwargs.update({'p': self.mha_dropout_kwargs.get('p', cons.DEFAULT_P_DROPOUT)})

        self.feed_forward_config = self._sub_check(feed_forward_config, 'feed_forward_config', SWATFeedForwardConfig)
        self.feed_forward_layer_norm_kwargs = self._sub_check(
            feed_forward_layer_norm_kwargs,
            'feed_forward_layer_norm_kwargs',
            dict
        )
        self.feed_forward_dropout_kwargs = self._sub_check(
            feed_forward_dropout_kwargs,
            'feed_forward_dropout_kwargs',
            dict
        )
        self.feed_forward_dropout_kwargs.update({
            'p': self.feed_forward_dropout_kwargs.get('p', cons.DEFAULT_P_DROPOUT)
        })


class SWATFeedForwardConfig(_Config):
    def __init__(
            self,
            activ_fn: Optional[torch.nn.Module] = torch.nn.GELU(),
            layer_norm: bool = False,  # bool
            layer_norm_kwargs: Optional[dict] = None,  # no effect if layerNorm = False
            dropout_kwargs: Optional[dict] = None,
            **kwargs
    ):
        super(SWATFeedForwardConfig, self).__init__(**kwargs)
        self.activ_fn, self.layer_norm = activ_fn, layer_norm
        self.layer_norm_kwargs = self._sub_check(layer_norm_kwargs, 'layer_norm_kwargs', dict)
        self.dropout_kwargs = self._sub_check(dropout_kwargs, 'dropout_kwargs', dict)


class SWATDecoderConfig(SWATFeedForwardConfig):  # layer_norm defaults to True for decoder only
    def __init__(self, layer_norm: bool = True, **kwargs):
        super(SWATDecoderConfig, self).__init__(layer_norm=layer_norm, **kwargs)


class SWATEmbeddingConfig(_Config):
    def __init__(
            self,
            n_swa_layers: int = cons.DEFAULT_N_SWA,
            d_swa: int = cons.DEFAULT_D_SWA,
            vocab_size: int = cons.VOCAB_SIZE,
            feature_size: int = cons.FEATURE_SIZE,
            edge_size: int = cons.EDGE_SIZE,
            feature_pad_idx: int = cons.PAD_TOKEN_ID,
            embedding_dropout_kwargs: Optional[dict] = None,
            token_embedding_kwargs: Optional[dict] = None,
            feature_embedding_kwargs: Optional[dict] = None,
            feature_layer_norm_kwargs: Optional[dict] = None,
            embedding_layer_norm_kwargs: Optional[dict] = None,
            swa_in_projection_kwargs: Optional[dict] = None,
            swa_out_projection_kwargs: Optional[dict] = None,
            feature_dropout_config: Optional["MaskDropoutConfig"] = None,
            edge_dropout_config: Optional["MaskDropoutConfig"] = None,  # Optional[MaskDropoutConfig]
            swa_layer_config: Optional[Union["SWALayerConfig", Iterable["SWALayerConfig"]]] = None,
            **kwargs
    ):
        super(SWATEmbeddingConfig, self).__init__(**kwargs)
        assert n_swa_layers > 0
        assert d_swa > 0
        self.vocab_size, self.feature_size, self.edge_size = vocab_size, feature_size, edge_size
        self.d_swa, self.feature_pad_idx = d_swa, feature_pad_idx

        self.embedding_dropout_kwargs = self._sub_check(embedding_dropout_kwargs, 'embedding_dropout_kwargs', dict)
        self.embedding_dropout_kwargs.update({'p': self.embedding_dropout_kwargs.get('p', cons.DEFAULT_P_DROPOUT)})

        self.feature_embedding_kwargs = self._sub_check(feature_embedding_kwargs, 'feature_embedding_kwargs', dict)
        self.token_embedding_kwargs = self._sub_check(token_embedding_kwargs, 'token_embedding_kwargs', dict)
        self.feature_layer_norm_kwargs = self._sub_check(feature_layer_norm_kwargs, 'feature_layer_norm_kwargs', dict)
        self.swa_in_projection_kwargs = self._sub_check(swa_in_projection_kwargs, 'swa_in_projection_kwargs', dict)
        self.swa_out_projection_kwargs = self._sub_check(swa_out_projection_kwargs, 'swa_out_projection_kwargs', dict)
        self.embedding_layer_norm_kwargs = self._sub_check(
            embedding_layer_norm_kwargs,
            'embedding_layer_norm_kwargs',
            dict
        )

        self.edge_dropout_config = self._sub_check(edge_dropout_config, 'edge_dropout_config', MaskDropoutConfig)
        self.feature_dropout_config = self._sub_check(
            feature_dropout_config,
            'feature_dropout_config',
            MaskDropoutConfig
        )
        self.swa_layer_config = _module_list_init(
            self._sub_check(swa_layer_config, 'swa_layer_config', SWALayerConfig),
            n_swa_layers,
            d_ff=(d_swa * 2)
        )


class MaskDropoutConfig(_Config):
    def __init__(self, p: float = 0.0, inplace: bool = False, default_val: int = cons.PAD_TOKEN_ID, **kwargs):
        super(MaskDropoutConfig, self).__init__(**kwargs)
        self.p, self.inplace, self.default_val = p, inplace, default_val
        assert 0 <= p < 1


class SWALayerConfig(_Config):
    def __init__(
            self,
            d_ff: int = -1,  # default,
            swa_layer_norm_kwargs: Optional[dict] = None,
            dropout_kwargs: Optional[dict] = None,
            feed_forward_config: Optional["SWATFeedForwardConfig"] = None,  # Optional[FeedForwardConfig]
            swa_config: Optional["SWAModuleConfig"] = None,  # 0 => f, 1 => b
            **kwargs
    ):
        super(SWALayerConfig, self).__init__(**kwargs)
        assert d_ff == -1 or d_ff > 0
        self.d_ff = d_ff

        self.swa_layer_norm_kwargs = self._sub_check(swa_layer_norm_kwargs, 'swa_layer_norm_kwargs', dict)
        self.dropout_kwargs = self._sub_check(dropout_kwargs, 'dropout_kwargs', dict)
        self.dropout_kwargs.update({'p': self.dropout_kwargs.get('p', cons.DEFAULT_P_DROPOUT)})

        self.feed_forward_config = self._sub_check(feed_forward_config, 'feed_forward_config', SWATFeedForwardConfig)
        self.swa_config = _module_list_init(self._sub_check(swa_config, 'swa_config', SWAModuleConfig), 2)

    def _print_str(self, *args, **kwargs):
        print_str = super(SWALayerConfig, self)._print_str(*args, **kwargs)

        return print_str.replace('(0)=', '(Forward)=').replace('(1)=', '(Backward)=')


class SWAModuleConfig(_Config):
    def __init__(self, dropout_kwargs: Optional[dict] = None, layer_norm_kwargs: Optional[dict] = None, **kwargs):
        super(SWAModuleConfig, self).__init__(**kwargs)
        self.layer_norm_kwargs = self._sub_check(layer_norm_kwargs, 'layer_norm_kwargs', dict)
        self.dropout_kwargs = self._sub_check(dropout_kwargs, 'dropout_kwargs', dict)
        self.dropout_kwargs.update({'p': self.dropout_kwargs.get('p', cons.DEFAULT_P_DROPOUT)})


def _module_list_init(var, n, **defaults):
    if isinstance(var, Iterable):
        out_var = var if isinstance(var, list) else list(var)
        assert len(var) == n
    else:
        out_var = [deepcopy(var) for _ in range(n)]

    for k, v in defaults.items():
        for x in out_var:
            if (lambda y: y is None or y == -1)(getattr(x, k)):
                setattr(x, k, v)

    return out_var


def _print_dict(d, n):
    if len(d) == 0:
        return '{}'
    if len(d) == 1:
        return (lambda x: '{' + cons.str_(x[0]) + ': ' + cons.str_(x[1]) + '}')(next(iter(d.items())))

    print_strs, indent = [], ' ' * (n + cons.INDENT_FACTOR)

    for k, v in d.items():
        if isinstance(v, dict):
            print_strs.append(f'{cons.str_(k)}: {_print_dict(v, n + cons.INDENT_FACTOR)},')
        else:
            print_strs = [f'{cons.str_(k)}: {cons.str_(v)},'] + print_strs

    return '{\n' + indent + (f'\n{indent}'.join(print_strs)[:-1]) + '\n' + (' ' * n) + '}'  # [:-1] removes last comma


def _get_kwarg_types(in_cls):
    if in_cls is _Config:
        return {}, set()

    assert len(in_cls.__bases__) == 1
    type_defaults = in_cls.__init__.__defaults__

    if type_defaults is None:
        kwargs, out_type_hints = {}, set()
    else:
        type_hints, kwargs = get_type_hints(in_cls.__init__), {}
        out_type_hints = set(type_hints.keys())

        if not all(
                v.default is not inspect.Parameter.empty or k == 'self' or str(v.kind) == 'VAR_KEYWORD'
                for k, v in inspect.signature(in_cls.__init__).parameters.items()
        ):
            raise NameError(
                f'({cons.type_str(in_cls, incl_path=True)}) all parameters in the __init__ method of a model config '
                'class should be specified as kwargs (arguments are not permitted)'
            )
        if not len(type_defaults) == len(type_hints.keys()):
            raise NameError(
                f'({cons.type_str(in_cls, incl_path=True)}) all kwargs in the __init__ method of a model config class '
                'must be type-hinted'
            )

        # compiling all potential "__"-notation kwargs (i.e. dict- or _Config subclass-valued) and their types
        for i, (k, type_str) in enumerate(map(lambda z: (z[0], str(z[1])), type_hints.items())):
            if type_str.startswith('typing.'):
                trgt_type_re = search(
                    r'(model\.configs\._configs\.[A-Za-z0-9_]+|[^A-Za-z0-9_.]+dict[^A-Za-z0-9_.]+)',
                    type_str
                )

                if trgt_type_re is not None:
                    if type_defaults[i] is not None:
                        raise NameError(
                            f'({cons.type_str(in_cls, incl_path=True)}; \'{k}\'): model_config class __init__ kwargs '
                            'that take dictionaries or other model config class objects must default to None'
                        )

                    all_types = set(map(lambda x: x[:-1], findall(r'[A-Za-z0-9_.]+[,\]]', type_str))) - {'NoneType'}
                    trgt_type = (lambda x: 'dict' if x[1:-1] == 'dict' else x)(trgt_type_re.group())
                    kwargs.update({k: trgt_type})

                    if not all(x == trgt_type for x in all_types):
                        raise NameError(
                            f'({cons.type_str(in_cls, incl_path=True)}; \'{k}\') if a model_config class __init__ '
                            'kwarg takes a dictionary or another model_config class object as an argument---no other '
                            'classes may be type-hinted'
                        )

    super_kwargs, super_type_hints = _get_kwarg_types(in_cls.__bases__[0])
    out_type_hints.update(super_type_hints)

    for k, v in super_kwargs.items():
        if k in kwargs.keys():
            assert issubclass(got(kwargs[k]), got(v))  # check subclass.__init__ kwarg has same type as super.__init__
        else:
            kwargs.update({k: v})

    return kwargs, out_type_hints
