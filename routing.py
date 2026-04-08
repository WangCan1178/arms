from mmm import *
import os
import json

import sys
idx = sys.argv[1]
idx = int(idx)

model_name__ = f"{idx}perc_3m_open_mmm"
config = json.load(open(f"{model_name__}.json"))
config['top_k'] = 4
model = MMMux(config)
model.load_state_dict(torch.load(model_name__ + ".pth"))
model = model.to('cuda:2', torch.bfloat16)
model.eval()
use_alm = False
use_helm = True
use_close = False

pkl_test_data_files = [
    "./open_alm_test.pkl", "./open_no_alm_test.pkl"
]
# test_dataset = JudgedRecordIterableDataset(pkl_test_data_files, 0, 4899 * 2 if use_alm else 996 * 2, 1)
test_dataset = JudgedRecordIterableDataset(pkl_test_data_files, 0, 1000000000, 1)

true_positive = 0
true_negative = 0
false_positive = 0
false_negative = 0

from tqdm import tqdm

import json
acc_judge_mapper = json.load(open("acc_judge_mapper.json", "r"))
import numpy as np

true_correct = 0
true_incorrect = 0
count_by_lang = {}
test_loader = DataLoader(test_dataset, collate_fn=lambda x: x, num_workers=24, prefetch_factor=8)
recorded_id = set()
total = 0
correct = 0
each_correct = [0, 0, 0, 0]
for each in tqdm(test_loader):
    each: JudgedRecord = each[0]
    record_idx = 0
    record_idx = int(each['record_id'].split('_')[-1])
    dataset_name = each['source_dataset']
    if (record_idx, dataset_name) in recorded_id:
        continue
    else:
        recorded_id.add((record_idx, dataset_name))
    if each is None:
        break
    with torch.no_grad():
        
        image = [each["image"], each["image"], each["image"], each["image"]]
        text = [each["question"], each["question"], each["question"], each["question"]]
        name = ['llama-3.2-11b-vision-instruct', 'qwen2_5-vl-7b-instruct', 'idefics2-8b', 'smolvlm-2_2b-instruct']

        # somebody_can_answer = False
        # for model_name in name:
        #     if acc_judge_mapper[f"{dataset_name}_{model_name}_{record_idx}"]:
        #         somebody_can_answer = True
        #         break
        # if not somebody_can_answer:
        #     continue
        total += 1
        for model_name in name:
            if acc_judge_mapper[f"{dataset_name}_{model_name}_{record_idx}"]:
                each_correct[name.index(model_name)] += 1
      
        output = model(image, text, name)
        # argmax_idx = int(torch.argmax(output, dim=0).item())
        output_no_tensor = output.detach().cpu().tolist()
        argmax_idx = np.argmax(output_no_tensor)
        distributed_name = name[argmax_idx]
        judge_mapper_key = f"{dataset_name}_{distributed_name}_{record_idx}"
        if acc_judge_mapper[judge_mapper_key]:
            correct += 1
            print(f"{correct/total}, {[e / total for e in each_correct]}")
        
        
filename = f"{model_name__}_result"
filename += "_routing"
filename += ".txt"
with open(filename, "w") as f:
    f.write(f"{correct} over {total}\n")
    f.write(f"{each_correct}\n")
