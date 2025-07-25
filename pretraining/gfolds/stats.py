
import re
import os
import json
from setup import GLOBAL_FP


RUN = 'run_0'

epochs, min_loss = {}, float('inf')

for fn in os.listdir(f'{GLOBAL_FP}/pretraining/{RUN}/stats/'):
    epoch = int(re.sub(r'\D', '', fn.split('_')[1]))

    if epoch not in epochs.keys():
        epochs.update({epoch: {}})

    with open(f'{GLOBAL_FP}/pretraining/{RUN}/stats/{fn}', 'r') as f:
        fn_file = json.load(f)

    for k, v in fn_file.items():
        k_sum, k_cnt = 0.0, 0

        for a, b in v['loss']:
            k_sum += a * b
            k_cnt += b

        epochs[epoch].update({int(k): (round(k_sum / k_cnt, 4), v.get('lr', v.get('learn_rate', None)))})

for ep in sorted(epochs.keys()):
    for mb in sorted(epochs[ep].keys()):
        mb_loss, mb_lr = epochs[ep][mb]
        min_loss = min(min_loss, mb_loss)

        print(f'EP={ep}, MB={mb}: {mb_loss} (min={min_loss}; LR={mb_lr})')
