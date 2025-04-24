
import json
import torch
from tqdm import tqdm
from model.io import SWATForSCInput
from global_filepath import GLOBAL_FP
from model.model import SWATForSequenceClassification
from model.configs import SWATForSequenceClassificationConfig


DATA_FP = GLOBAL_FP + '/factuality/data/mv_data_bin_guc.json'
CHK_FP = GLOBAL_FP + '/pretraining/run_2/checkpoints/ep3_mb199.chk'
PATIENCE = 5
LR = 1e-6
WEIGHT_DECAY = 1e-5
BATCH_SIZE = 8
DEVICE = 0
MODEL_CONFIG = SWATForSequenceClassificationConfig(
    n_classes=2,
    head_config__layer_norm=True,
    head_config__dropout_kwargs__p=0.1,
    pooling='mean'
)
SEED = 25


def generate_data(data_list):
    for x in data_list:
        yield SWATForSCInput.from_dir_graph(x['g'], targets=x['l'])


torch.manual_seed(SEED)
torch.cuda.manual_seed(SEED)

model = SWATForSequenceClassification.from_swat_model(CHK_FP, MODEL_CONFIG)
model.to(device=DEVICE)
model.train()
optimizer = torch.optim.AdamW(model.parameters(), lr=LR, weight_decay=WEIGHT_DECAY)

with open(DATA_FP, 'r') as f:
    data_file = json.load(f)

train_data, test_data = data_file['train'], data_file['test']
ep_cnt, patience_cnt, best_acc = 0, 0, 0.0

while patience_cnt < PATIENCE:
    ep_cnt += 1
    print(f'EPOCH {ep_cnt}:')
    print()

    optimizer.zero_grad()
    permutation = torch.randperm(len(train_data)).tolist()
    batch_perm = map(lambda i_: train_data[i_], permutation)
    train_batches = SWATForSCInput.generate_batches(generate_data(batch_perm), batch_size=BATCH_SIZE)
    epoch_loss, loss_cnt = 0.0, 0

    for _ in tqdm(range(-(-len(train_data) // BATCH_SIZE))):
        batch = next(train_batches)
        model_out = model(batch)
        loss = model_out.loss()

        loss.backward()
        optimizer.step()
        epoch_loss += loss.item()
        loss_cnt += batch.batch_size

        optimizer.zero_grad()
        del batch, model_out, loss

    print()
    print(f'LOSS: {epoch_loss / loss_cnt}')
    print()

    test_batches = SWATForSCInput.generate_batches(generate_data(test_data), batch_size=BATCH_SIZE)
    epoch_acc, acc_cnt = 0.0, 0
    model.eval()

    with torch.no_grad():
        for _ in tqdm(range(-(-len(test_data) // BATCH_SIZE))):
            batch = next(test_batches)
            model_out = model(batch)

            epoch_acc += model_out.accuracy() * batch.batch_size
            acc_cnt += batch.batch_size
            del batch, model_out

    acc = epoch_acc / acc_cnt

    if acc > best_acc:
        best_acc, patience_cnt = acc, 0
    else:
        patience_cnt += 1

    print()
    print(f'ACC: {round(acc * 100, 5)}% (BEST={round(best_acc * 100, 5)}%; PATIENCE={patience_cnt})')
    print()
