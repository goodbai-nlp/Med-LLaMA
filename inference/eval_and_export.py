# coding:utf-8
import sys
import json

with open(sys.argv[1], 'r', encoding='utf-8') as fgold:
    gold_data = [json.loads(line.strip()) for line in fgold]
    
with open(sys.argv[2], 'r', encoding='utf-8') as fpred:
    pred_data = [line.strip().lower() for line in fpred]
    

assert len(gold_data) == len(pred_data)

right = 0
pred_res = {}
for gold, pred in zip(gold_data, pred_data):
    gold_label = gold["final_decision"]
    if pred.startswith(gold_label):
        right += 1
    pred_res[gold['id']] = pred

print("acc:", right/len(gold_data))

with open(sys.argv[3], 'w', encoding='utf-8') as fout:
    json.dump(pred_res, fout, indent=4)
