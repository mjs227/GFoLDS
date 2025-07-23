
import gc
import json
import torch
from tqdm import tqdm
from setup import GLOBAL_FP
from model.io import SWATForSCInput
from model.model import SWATForSequenceClassification


BATCH_SIZE = 64
LEARN_RATE = 5e-6
WEIGHT_DECAY = 1e-1
SAVE_PARAMS = False  # True => save best state dict; False => save last state dict
NUM_NEG = 4  # train/adv datasets (number of repeated external negation prefixes)
FP = GLOBAL_FP + '/double_negation/'
INIT_CHK_FP = GLOBAL_FP + '/nli/runs/gfolds/checkpoints/ep4.chk'
ADV_FP = GLOBAL_FP + '/double_negation/'
SEED = 22
DEVICE = 0


def generate_data(data_list):
    for x in data_list:
        yield SWATForSCInput.from_dir_graph(x['g'], targets=label_map[x['lbl']])


def evaluate(model_, eval_file, use_tqdm=True):
    eval_batches = SWATForSCInput.generate_batches(generate_data(eval_file), batch_size=BATCH_SIZE)
    tqdm_ = tqdm if use_tqdm else lambda z: z
    acc_total, acc_cnt = 0, 0
    model_.eval()

    with torch.no_grad():
        for _ in tqdm_(range(-(-len(eval_file) // BATCH_SIZE))):
            try:
                eval_batch = next(eval_batches)
                eval_out = model_(eval_batch)
                acc_total += eval_out.accuracy() * eval_batch.batch_size
                acc_cnt += eval_batch.batch_size

                del eval_batch, eval_out
            except StopIteration:
                break

    model_.train()

    return acc_total * 100 / acc_cnt


label_map = {y: x for x, y in enumerate(('e', 'n', 'c'))}
torch.manual_seed(SEED)
torch.cuda.manual_seed(SEED)

with open(f'{ADV_FP}data/train_{NUM_NEG}_MULTI.json', 'r') as f:
    train_file = json.load(f)

with open(GLOBAL_FP + '/NLI/dev', 'r') as f:
    dev_file = json.load(f)

with open(f'{ADV_FP}data/adv_{NUM_NEG}_MULTI.json', 'r') as f:
    adv_file = json.load(f)

model = SWATForSequenceClassification.from_pretrained(INIT_CHK_FP)
model.to(device=DEVICE)
model.train()

if WEIGHT_DECAY == 0.0:
    optimizer = torch.optim.Adam(model.parameters(), lr=LEARN_RATE)
else:
    optimizer = torch.optim.AdamW(model.parameters(), lr=LEARN_RATE, weight_decay=WEIGHT_DECAY)

optimizer.zero_grad()
best_state_dict = None

num_batch, patience_cnt = -(-len(train_file) // BATCH_SIZE), 0
train_stats = {'acc': [], 'loss': []}
prev_best_acc = evaluate(model, dev_file)

print(f'Initial dev-set accuracy: {round(prev_best_acc, 3)}%')

# init_adv_acc = evaluate(model, adv_file)
#
# print(f'Initial challenge set accuracy: {round(init_adv_acc, 3)}%')

while patience_cnt < 5:
    print()
    print(f'Iteration {len(train_stats["acc"]) + 1}:')
    print()

    permutation, iter_loss = torch.randperm(len(train_file)).tolist(), 0.0
    batch_perm = map(lambda i_: train_file[i_], permutation)
    train_batches = SWATForSCInput.generate_batches(generate_data(batch_perm), batch_size=BATCH_SIZE)
    model.train()

    for i in tqdm(range(num_batch)):
        try:
            model_input = next(train_batches)
            model_output = model(model_input)
            loss = model_output.loss()

            loss.backward()
            optimizer.step()
            optimizer.zero_grad()
            iter_loss += loss.item() * model_input.batch_size

            del model_input, model_output
        except StopIteration:
            break

    curr_acc = evaluate(model, dev_file)

    if curr_acc > prev_best_acc:
        patience_cnt, prev_best_acc = 0, curr_acc

        if SAVE_PARAMS:
            best_state_dict, _ = model.save()
    else:
        if SAVE_PARAMS and best_state_dict is None:
            best_state_dict, _ = model.save()

        patience_cnt += 1

    train_stats['loss'].append(iter_loss / len(train_file))
    train_stats['acc'].append(curr_acc)

    print(f'Acc={round(curr_acc, 3)}%, Best={round(prev_best_acc, 3)}%, Patience={patience_cnt}')

    gc.collect()
    torch.cuda.empty_cache()

print()
print('\nInoculation complete!\n')

if SAVE_PARAMS:
    model.to('cpu')
    model.load_state_dict(best_state_dict)
    model.to(DEVICE)

inoc_adv_acc = evaluate(model, adv_file)

print(f'Inoculated challenge set accuracy: {round(inoc_adv_acc, 3)}%')

with open(f'{ADV_FP}stats/depth{NUM_NEG}_train.json', 'w') as f:
    json.dump(train_stats, f)

model.save(f'{ADV_FP}checkpoints/depth{NUM_NEG}.chk')
