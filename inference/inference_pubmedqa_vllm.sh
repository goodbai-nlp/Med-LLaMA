#!/bin/bash
export CUDA_VISIBLE_DEVICES=0
#export CUDA_VISIBLE_DEVICES=1
export PATH="/opt/conda/bin:$PATH"
source ~/.bashrc

MODEL=$1
DATA=$2
python -u inference_vllm.py --test_file ${DATA} --model_name_or_path ${MODEL} --num_beams 1 --max_new_tokens 128 --out_prefix "test-pred" --instruction "" --prompt_template "pubmedqa" 2>&1 | tee $MODEL/eval.log
python -u eval_and_export.py ${DATA} $MODEL/test-pred_test.jsonl_pred_vllm $MODEL/test-pred.json 2>&1 | tee $MODEL/eval-res.log
