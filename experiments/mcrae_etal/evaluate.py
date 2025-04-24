
import gc
import math
import torch
import pickle
import numpy as np
from tqdm import tqdm
from copy import deepcopy
from scipy.stats import spearmanr
from global_filepath import GLOBAL_FP
from typing import Optional, Callable, Union, Dict, Any, List, Tuple


FP = f'{GLOBAL_FP}/prop_inf/mcrae_data_bert_comp_base.pkl'
FOLD_FP = f'{GLOBAL_FP}/prop_inf/mcrae_fold_words'
SIM_TYPE = 'cos-clipped'
SHIFT_NEG_WGT = 30.0
SHIFT_POS_WGT = 30.0
UNK_SIZE = 50
DEVICE = 'cpu'
HYPERPARAMS = {
    'mu_cont': [1e-8, 1e-4, 1e-2, 1, 10, 100, 1000],
    'mu_abdn': [1e-8, 1e-4, 1e-2, 1, 10, 100, 1000],
    'nn': [{'k': 1}, {'k': 5}, {'k': 10}, {'k': 20}]
}


class ModAds:
    def __init__(
            self,
            labeled_nodes: Dict[Any, Union[torch.Tensor, List[float]]],
            unlabeled_nodes: Union[list, tuple, set],
            weights: Union[Dict[Any, float], torch.Tensor],
            mu_inj: float = 1.0,
            mu_abdn: float = 1e-4,
            mu_cont: float = 1e-4,
            beta: float = 2.0,
            nn: Optional[Dict[str, Union[int, bool]]] = None,
            device: Union[str, int, torch.device] = 'cpu',
            dtype: torch.dtype = torch.float32,
            transform_w: Optional[Callable[[torch.Tensor], torch.Tensor]] = None
    ):
        self.node_indices = list(labeled_nodes.keys())
        self.unlabeled_index = len(self.node_indices)
        self.node_indices += list(unlabeled_nodes)
        self.node_len = len(self.node_indices)
        node_labels = [
            (lambda x: (x if isinstance(x, list) else x.tolist()) + [0.0])(labeled_nodes[self.node_indices[i]])
            for i in range(self.unlabeled_index)
        ]
        node_labels += [[0.0 for _ in range(len(node_labels[0]))] for _ in range(self.unlabeled_index, self.node_len)]
        node_index_dict = {self.node_indices[i]: i for i in range(self.node_len)}

        assert all(len(x) == len(node_labels[0]) for x in node_labels[1:])
        assert nn is None or set(nn.keys()) <= {'k', 'bin'}

        with torch.no_grad():
            if isinstance(weights, dict):
                temp_w = torch.zeros((self.node_len, self.node_len), dtype=dtype)

                for a, b in weights.keys():
                    temp_w[node_index_dict[a], node_index_dict[b]] = weights[(a, b)]
            else:
                assert tuple(weights.shape) == (self.node_len, self.node_len)
                assert all(weights[i, i].item() == 0.0 for i in range(self.node_len))
                temp_w = weights.to(dtype=dtype)

            if nn is None:
                self.W = temp_w
            else:
                assert self.node_len > nn['k']
                self.W = torch.zeros((self.node_len, self.node_len), dtype=dtype)

                for i in range(self.node_len):
                    w_i = temp_w[i]

                    for _ in range(nn['k']):
                        argmax = torch.argmax(w_i).item()
                        argmax_val = w_i[argmax]
                        self.W[i, argmax] = argmax_val
                        w_i[argmax] = 0.0

                if nn.get('bin', False):
                    self.W.apply_(lambda x: 1 if x > 0 else 0)

            self.W.apply_(lambda x: max(x, 1e-10))
            self.W = self.W.to(device=device, dtype=dtype)

            if transform_w is not None:
                try:
                    self.W = transform_w(self.W)
                except TypeError:
                    self.W = transform_w(self.W.to('cpu')).to(device=device, dtype=dtype)

            beta = torch.tensor([beta], dtype=dtype, device=device)
            self.mu_cont, self.mu_abdn, self.mu_inj = mu_cont, mu_abdn, mu_inj
            self.Y = torch.tensor(node_labels, dtype=dtype, device=device)
            self.Y_hat = torch.clone(self.Y)
            self.r = torch.tensor([0 for _ in range(self.Y.size(1) - 1)] + [1], dtype=dtype, device=device)

            p_cont = [0 for _ in range(self.node_len)]
            p_inj = [0 for _ in range(self.node_len)]
            p_abdn = [0 for _ in range(self.node_len)]

            for i in range(self.node_len):
                h_v = torch.dot(self.W[i], torch.log(self.W[i])) * -1
                c_v = (torch.log(beta) / torch.log(beta + torch.exp(h_v))).item()
                d_v = (1 - c_v) * torch.sqrt(h_v).item()
                z_v = max(c_v + d_v, 1)
                p_cont[i], p_inj[i] = c_v / z_v, d_v / z_v
                p_abdn[i] = 1 - (p_cont[i] + p_inj[i])

            self.p_cont = torch.tensor(p_cont, dtype=dtype, device=device)
            self.p_inj = torch.tensor(p_inj, dtype=dtype, device=device)
            self.p_abdn = torch.tensor(p_abdn, dtype=dtype, device=device)
            self.M = [0 for _ in range(self.node_len)]

            for i in range(self.node_len):
                main_sum = torch.sum((p_cont[i] * self.W[i]) + (self.p_cont * torch.flatten(self.W[:, i])))
                main_sum -= 2 * p_cont[i] * self.W[i][i]
                self.M[i] = 1 / ((self.mu_inj * p_inj[i]) + (self.mu_cont * main_sum.item()) + self.mu_abdn)

    def __call__(
            self,
            epsilon: float = 1e-3,
            patience: Optional[int] = 100,
            return_delta: bool = False,
            return_labeled: bool = False,
            return_abdn: bool = False
    ) -> Union[Dict[Any, float], Dict[str, Union[Dict[Any, float], Tuple[float]]]]:
        assert epsilon > 0.0
        assert patience is None or patience > 0
        delta, patience_cnt = (epsilon + 1,), 0
        add_patience, patience = (0, 1) if patience is None else (1, patience)

        with torch.no_grad():
            prev_y_hat = torch.clone(self.Y_hat)

            while delta[-1] > epsilon and patience_cnt < patience:
                for i in range(self.node_len):
                    cont_t = (self.p_cont[i] * self.W[i]) + (self.p_cont * torch.flatten(self.W[:, i]))
                    d_v = torch.sum(cont_t[:, None] * self.Y_hat, dim=0)
                    y = (self.mu_inj * self.p_inj[i] * self.Y[i]) + (self.mu_cont * d_v) + (self.p_abdn[i] * self.r)
                    self.Y_hat[i] = self.M[i] * y

                delta += (torch.sum(torch.abs(prev_y_hat - self.Y_hat)).item(),)
                prev_y_hat = torch.clone(self.Y_hat)
                patience_cnt += add_patience

        y_hat_list = self.Y_hat.tolist()
        r_fn = (lambda z: z) if return_abdn else (lambda z: z[:-1])
        unlabeled = {self.node_indices[i]: r_fn(y_hat_list[i]) for i in range(self.unlabeled_index, self.node_len)}

        if not (return_labeled or return_delta):
            return unlabeled

        out_dict = {'unlabeled': unlabeled}

        if return_labeled:
            out_dict.update({
                'labeled': {self.node_indices[i]: r_fn(y_hat_list[i]) for i in range(self.unlabeled_index)}
            })
        if return_delta:
            out_dict.update({'delta': delta[1:]})

        return out_dict


def gen_fold_data(fold_idx):
    gc.collect()
    torch.cuda.empty_cache()
    unk_terms, labeled_terms = {}, {}

    for term_ in mcrae_file:
        (unk_terms if term_['term'] in folds[fold_idx] else labeled_terms).update({term_['term']: term_['features']})

    return unk_terms, labeled_terms


def eval_fold(modads_out, unk_terms):
    modads_out, sr = modads_out.get('unlabeled', modads_out), 0.0
    assert set(modads_out.keys()) == set(unk_terms.keys())

    for k_, v_ in modads_out.items():
        sr += (lambda rho: 0.0 if math.isnan(rho) else rho)(spearmanr(np.array(v_), np.array(unk_terms[k_]))[0])

    return sr / len(modads_out.keys())


def hp_rec(depth):
    if depth == len(hp_keys):
        yield {}
    else:
        for v_ in HYPERPARAMS[hp_keys[depth]]:
            for d in hp_rec(depth + 1):
                yield {**{hp_keys[depth]: v_}, **d}


def hp_str(kwargs_):
    return ', '.join(f'{k_}={v_}' for k_, v_ in kwargs_.items())


SHIFT_POS_WGT = 1.0 if SHIFT_POS_WGT is None else SHIFT_POS_WGT
assert isinstance(SHIFT_POS_WGT, float)
assert SHIFT_NEG_WGT >= 0.0 if isinstance(SHIFT_NEG_WGT, float) else True
assert SIM_TYPE in {'cos-clipped', 'cos-shifted', 'euclidean'}
folds = []

with open(FP, 'rb') as f:
    mcrae_file = pickle.load(f)

with open(FOLD_FP, 'r') as f:
    for line in map(lambda z_: z_.strip().lower(), f):
        if line.startswith('fold'):
            folds.append(set())
        elif len(line) > 0:
            line = line[1:] if line.startswith('-') else line
            folds[-1].add((line.split('-')[0] if '-' in line else line).strip())

mean_feat_num, mean_feat_den, feat_len, term_weights = 0.0, 0, 0, {}

for term in mcrae_file:
    if not SIM_TYPE == 'euclidean':
        term['embedding'] = torch.nn.functional.normalize(term['embedding'], dim=0)

    feat_len = max(feat_len, max(term['features'].keys()))
    mean_feat_num += sum(term['features'].values())
    mean_feat_den += len(term['features'].keys())

max_wgt, min_wgt = float('-inf'), float('inf')
mean_feat = -mean_feat_num / mean_feat_den
feat_len += 1

for idx, term in enumerate(mcrae_file):
    if SHIFT_NEG_WGT is None:
        feat_arr = [0.0] * feat_len
    elif isinstance(SHIFT_NEG_WGT, str):
        if SHIFT_NEG_WGT == 'micro':
            feat_arr = [-sum(term['features'].values()) / len(term['features'].keys())] * feat_len
        elif SHIFT_NEG_WGT == 'macro':
            feat_arr = [mean_feat] * feat_len
        else:
            raise NotImplementedError(SHIFT_NEG_WGT)
    elif isinstance(SHIFT_NEG_WGT, float):
        feat_arr = [-SHIFT_NEG_WGT] * feat_len
    else:
        raise NotImplementedError('\'SHIFT_NEG_WGT\' must be None, str, or float')

    for k, v in term['features'].items():
        feat_arr[k] = v * SHIFT_POS_WGT

    term.update({'features': feat_arr})

    for term2 in mcrae_file[idx + 1:]:
        e1, e2 = term['embedding'], term2['embedding']

        if SIM_TYPE == 'euclidean':
            sim_val = torch.sqrt(torch.sum((e1 - e2) ** 2)).item()
            max_wgt, min_wgt = max(max_wgt, sim_val), min(min_wgt, sim_val)
        else:
            vdot = torch.dot(e1, e2)
            vdot = ((1 + vdot) / 2) if SIM_TYPE == 'cos-shifted' else vdot
            sim_val = torch.clip(vdot, min=0.0, max=1.0).item()

        term_weights.update({(term['term'], term2['term']): sim_val})

wgt_keys = list(term_weights.keys())

if SIM_TYPE == 'euclidean':
    max_wgt = max_wgt - min_wgt

    for x_, y_ in wgt_keys:
        sim_val = min(max((term_weights[(x_, y_)] - min_wgt) / max_wgt, 0.0), 1.0)
        term_weights.update({(x_, y_): sim_val, (y_, x_): sim_val})
else:
    for x_, y_ in wgt_keys:
        term_weights.update({(y_, x_): term_weights[(x_, y_)]})

hp_keys = list(HYPERPARAMS.keys())
n_hp_configs = 1

for hp in HYPERPARAMS.values():
    n_hp_configs *= len(hp)

unk_dev, labeled_dev = gen_fold_data(0)
hp_gen, best_dev_kwargs, best_dev_sr = hp_rec(0), {}, -1.0

print('HYPERPARAMETER SELECTION:')
print()
print()

for _ in tqdm(range(n_hp_configs)):
    kwargs = next(hp_gen)
    modads = ModAds(labeled_dev, set(unk_dev.keys()), term_weights, **kwargs)
    spearman_rho = eval_fold(modads(), unk_dev)

    print()
    print(f'   {hp_str(kwargs)}: {spearman_rho}')
    print()

    if spearman_rho > best_dev_sr:
        best_dev_kwargs, best_dev_sr = deepcopy(kwargs), spearman_rho

    del modads

print()
print()
print(f'BEST HP CONFIG: {hp_str(best_dev_kwargs)}')
print(f'   RHO: {best_dev_sr}\n\nTEST EVALUATION:')

total_sr = 0.0

for i_ in range(1, len(folds)):
    print(f'   FOLD {i_ + 1}/{len(folds)}:')

    unk_gold, labeled = gen_fold_data(i_)
    modads = ModAds(labeled, set(unk_gold.keys()), term_weights, **best_dev_kwargs)
    spearman_rho = eval_fold(modads(), unk_gold)
    total_sr += spearman_rho

    print(f'      RHO: {spearman_rho}')

print('\n\nSTATS:')
print(f'   MEAN TEST SR: {round(total_sr / (len(folds) - 1), 5)}')
print(f'   BEST DEV SR:  {round(best_dev_sr, 5)}')
print('   HYPERPARAMS:  ' + hp_str(best_dev_kwargs))
