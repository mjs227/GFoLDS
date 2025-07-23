
import json
import random
import torch
from tqdm import tqdm
from setup import GLOBAL_FP
from model.io import SWATForTCInput
from model.graphtools import SWADirGraph
from model.model import SWATForTokenClassification
from model.configs import SWATForTokenClassificationConfig


DEV_FP = GLOBAL_FP + '/nli/data/dev'
TEST_FP = GLOBAL_FP + '/nli/data/test'
DEVICE = 0
BATCH_SIZE = 16
MODEL_CONFIG = SWATForTokenClassificationConfig(
    n_classes=2,
    head_config__layer_norm=True,
    head_config__dropout_kwargs__p=0.1,
    swat_config__d_model=1024,
    swat_config__n_encoder_layers=10,
    swat_config__embedding_config__d_swa=1024,
    swat_config__embedding_config__n_swa_layers=2
)


def generate_data(data_list):
    for x in data_list:
        yield SWATForTCInput.from_dir_graph(
            x['g'],
            targets=torch.tensor([x['y']])
        )


def prep_data(ds_elem):
    ds_g = SWADirGraph.from_dict(ds_elem['g'])
    if_then_id = min(ds_g.nodes.keys())
    d = {x: float('inf') for x in ds_g.nodes.keys()}
    d.update({if_then_id: 0})
    q = list(ds_g.nodes.keys())
    y = {x: None for x in ds_g.nodes.keys()}
    y.update({if_then_id: -100})
    y.update({next(x for x, y in ds_g.adj_out[if_then_id] if y == 0): 0})
    y.update({next(x for x, y in ds_g.adj_out[if_then_id] if y == 1): 1})

    while len(q) > 0:
        q.sort(key=lambda x: d[x])
        v = q[0]
        q.remove(v)

        for u, _ in ds_g.adj_out[v]:
            if d[u] > d[v] + 1:
                d.update({u: d[v] + 1})

                if y[u] is None:
                    y.update({u: y[v]})

        for u, _ in ds_g.adj_in[v]:
            if d[u] > d[v] + 1:
                d.update({u: d[v] + 1})

                if y[u] is None:
                    y.update({u: y[v]})

    valid = all(z is not None for z in y.values())
    valid = valid and all(0 <= z < float('inf') for z in d.values())

    node_ids = sorted(list(ds_g.nodes.keys()))
    ds_elem.update({'y': [y[x] for x in node_ids]})
    ds_elem.update({'d': [d[x] for x in node_ids]})

    if valid:
        return ds_elem

    return None


with open(DEV_FP, 'r') as f:
    dev_file = json.load(f)

with open(TEST_FP, 'r') as f:
    test_file = json.load(f)

random.shuffle(test_file)
dev_file = [w for w in map(prep_data, dev_file) if w is not None]
test_file = [w for w in map(prep_data, test_file) if w is not None]

model = SWATForTokenClassification.from_config(MODEL_CONFIG)
model.to(device=DEVICE)
model.train()
optimizer = torch.optim.AdamW(model.parameters(), lr=5e-6, weight_decay=5e-6)
optimizer.zero_grad()
loss_fn = torch.nn.BCEWithLogitsLoss()
prev_acc, curr_acc = -1.0, 0.0

while prev_acc < curr_acc:
    random.shuffle(dev_file)
    batches = SWATForTCInput.generate_batches(
        generate_data(dev_file),
        batch_size=BATCH_SIZE
    )
    prev_acc = curr_acc
    running_loss = 0.0

    for _ in tqdm(range(int(-(-len(dev_file) // BATCH_SIZE)))):
        try:
            model_in = next(batches)
            model_out = model(model_in)
            loss = model_out.loss()

            loss.backward()
            optimizer.step()
            optimizer.zero_grad()

            running_loss += loss.item()
        except StopIteration:
            break

    print()
    print(f'LOSS: {running_loss / len(dev_file)}')
    acc_by_dist, cnt_by_dist = {0: {}, 1: {}}, {0: {}, 1: {}}

    with torch.no_grad():
        test_batches = SWATForTCInput.generate_batches(
            generate_data(test_file),
            batch_size=1
        )

        for tb, tb_item in zip(test_batches, test_file):
            logits = model(tb).predictions.flatten().tolist()

            for pred, trgt, dist in zip(logits, tb_item['y'][1:], tb_item['d'][1:]):
                if dist not in cnt_by_dist[trgt].keys():
                    cnt_by_dist[trgt].update({dist: 1})
                    acc_by_dist[trgt].update({dist: 1 if pred == trgt else 0})
                else:
                    cnt_by_dist[trgt][dist] += 1
                    acc_by_dist[trgt][dist] += pred == trgt

    acc_by_dist_s2, cnt_by_dist_s2 = sum(acc_by_dist[0].values()), sum(cnt_by_dist[0].values())
    acc_by_dist_s1, cnt_by_dist_s1 = sum(acc_by_dist[1].values()), sum(cnt_by_dist[1].values())
    acc_s1 = round(acc_by_dist_s1 * 100 / cnt_by_dist_s1, 3)
    acc_s2 = round(acc_by_dist_s2 * 100 / cnt_by_dist_s2, 3)
    curr_acc = round((acc_by_dist_s1 + acc_by_dist_s2) * 100 / (cnt_by_dist_s1 + cnt_by_dist_s2), 3)
    acc_keys = sorted(list(set(acc_by_dist[0].keys()).union(set(acc_by_dist[1].keys()))))
    total_acc_dict = {k_: v_ for k_, v_ in acc_by_dist[0].items()}
    total_cnt_dict = {k_: v_ for k_, v_ in cnt_by_dist[0].items()}

    for k_, v_ in acc_by_dist[1].items():
        if k_ in total_acc_dict.keys():
            total_acc_dict[k_] += v_
            total_cnt_dict[k_] += cnt_by_dist[1][k_]
        else:
            total_acc_dict.update({k_: v_})
            total_cnt_dict.update({k_: cnt_by_dist[1][k_]})

    def calc_acc(acc, cnt, k_):
        return round(acc[k_] * 100 / cnt[k_], 3) if k_ in cnt.keys() else None

    print(f'ACCURACY={acc_s1}%, {acc_s2}%, ({curr_acc}%)')

    for k in acc_keys:
        acc_s1 = calc_acc(acc_by_dist[1], cnt_by_dist[1], k)
        acc_s2 = calc_acc(acc_by_dist[0], cnt_by_dist[0], k)
        total_acc = calc_acc(total_acc_dict, total_cnt_dict, k)

        print(f'   D={k}: {acc_s1}%, {acc_s2}%, ({total_acc}%)')

    print()
    print()






