
import gc
import json
import torch
import random
from model.io import SWATForSCInput
from global_filepath import GLOBAL_FP
from model.graphtools import SWADirGraph
from model.model import SWATForSequenceClassification
from model.configs import SWATForSequenceClassificationConfig


N_SWA = list(range(1, 5))
EVAL_N = [list(range(1, 13))] * len(N_SWA)
BATCH_SIZE = 24
LR = 1e-5
DEVICE = 0
PATIENCE = 100
INIT_CHK_FP = GLOBAL_FP + '/nli/checkpoints/ep4.chk'
TOK_FP = GLOBAL_FP + '/tokenizer_config.json'


def generate_data(data_list):
    for x, y in data_list:
        yield SWATForSCInput.from_dir_graph(x, targets=y)


with open(TOK_FP, 'r') as f:
    tok_dict = json.load(f)['tokenizer_dict']

neg, arg1 = tok_dict['v']['neg'], tok_dict['e']['ARG1']
neg_feats = (
    tok_dict['f']['[SF:prop]'], tok_dict['f']['[PROG:-]'], tok_dict['f']['[PERF:-]'], tok_dict['f']['[TENSE:untensed]']
)


N_SWA = ((N_SWA,) * len(EVAL_N)) if isinstance(N_SWA, int) else N_SWA
assert len(EVAL_N) == len(N_SWA)
stats_dict, model = {}, None

for n_swa, eval_n in zip(N_SWA, EVAL_N):
    stats_dict.update({(n_swa, e): False for e in eval_n})

for eval_n, n_swa in zip(EVAL_N, N_SWA):
    del model
    gc.collect()
    torch.cuda.empty_cache()

    for en in eval_n:
        data_templates, train_data, test_data = [], [], []

        for n in range(1, en + 1):
            temp_n, last_id = SWADirGraph(), None

            for _ in range(n):
                neg_id = temp_n.add_node(neg)

                for f_id in neg_feats:
                    temp_n.add_feature(neg_id, f_id)

                if last_id is not None:
                    temp_n.add_edge(neg_id, last_id, arg1)

                last_id = neg_id

            data_templates.append((temp_n, n % 2))

        for n in range(en):
            train_data.extend(data_templates[n] for _ in range(BATCH_SIZE))
            test_data.append(data_templates[n])

        MODEL_CONFIG = SWATForSequenceClassificationConfig(
            n_classes=2,
            head_config__layer_norm=True,
            head_config__dropout_kwargs__p=0.25,
            pooling='mean',
            swat_config__d_model=1024,
            swat_config__n_encoder_layers=10,
            swat_config__embedding_config__d_swa=1024,
            swat_config__embedding_config__n_swa_layers=n_swa
        )
        model = SWATForSequenceClassification.from_config(MODEL_CONFIG)
        model.to(device=DEVICE)
        model.train()

        optimizer = torch.optim.AdamW(model.parameters(), lr=LR, weight_decay=1e-2)
        optimizer.zero_grad()

        for e in range(1, PATIENCE + 1):
            random.shuffle(train_data)
            batches = SWATForSCInput.generate_batches(generate_data(train_data), batch_size=BATCH_SIZE)
            total_loss = 0.0

            for batch in batches:
                model_out = model(batch)
                loss = model_out.loss()

                loss.backward()
                total_loss += loss.item() * batch.batch_size
                optimizer.step()
                optimizer.zero_grad()

                del model_out, loss

            print(f'\n(N_SWA={n_swa}, EVAL_N={en}, ITER={e}) LOSS: {total_loss / len(train_data)}')

            model.eval()
            acc_cnt = 0.0

            with torch.no_grad():
                for i, z in enumerate(generate_data(test_data), start=1):
                    model_out = model(z)
                    acc = model_out.accuracy()
                    acc_cnt += acc
                    print(f'{i}: {acc}')

            model.train()

            if acc_cnt == en:
                stats_dict.update({(n_swa, en): True})
                break
        else:
            break

out_dict = {}

for (k0, k1), v in stats_dict.items():
    if k0 in out_dict.keys():
        out_dict[k0].update({k1: v})
    else:
        out_dict.update({k0: {k1: v}})

with open(f'{GLOBAL_FP}/double_negation/stats/can_it_count.json', 'w') as f:
    json.dump(out_dict, f)
