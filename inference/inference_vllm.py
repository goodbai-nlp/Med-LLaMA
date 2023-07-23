# Copyright (c) Microsoft Corporation.
# SPDX-License-Identifier: Apache-2.0

# DeepSpeed Team
import argparse
import logging
import torch
import json
import sys
import os
from tqdm import tqdm
from transformers import (
    AutoConfig,
    LlamaConfig,
    LlamaTokenizer,
    AutoModelForCausalLM,
    LlamaForCausalLM,
    AutoTokenizer,
)
from vllm import LLM, SamplingParams

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), os.path.pardir)))

logger = logging.getLogger(__name__)


def parse_args():
    parser = argparse.ArgumentParser(description="Eval the finetued SFT model")
    parser.add_argument(
        "--model_name",
        type=str,
        help="Path to pretrained model",
        required=False,
    )
    parser.add_argument(
        "--model_name_or_path",
        type=str,
        help="Path to pretrained model",
        required=True,
    )
    parser.add_argument(
        "--test_file",
        type=str,
        help="Path to test_file",
        required=True,
    )
    parser.add_argument(
        "--num_beams",
        type=int,
        default=5,
        help="Specify num of beams",
    )
    parser.add_argument(
        "--beam_search",
        action="store_true",
        default=False,
        help="use beam search or not",
    )
    parser.add_argument(
        "--device_id",
        type=int,
        default=0,
        help="Specify num of beams",
    )
    parser.add_argument(
        "--ngpus",
        type=int,
        default=1,
        help="Specify num of beams",
    )
    parser.add_argument(
        "--top_k",
        type=int,
        default=4,
        help="Specify num of beams",
    )
    parser.add_argument(
        "--presence_penalty",
        type=float,
        default=0.0,
        help="Specify num of beams",
    )
    parser.add_argument(
        "--frequency_penalty",
        type=float,
        default=0.0,
        help="Specify num of beams",
    )
    parser.add_argument(
        "--num_return_sequences",
        type=int,
        default=1,
        help="Specify num of return sequences",
    )
    parser.add_argument(
        "--max_new_tokens",
        type=int,
        default=400,
        help="Specify num of return sequences",
    )
    parser.add_argument(
        "--local_rank", type=int, default=-1, help="local_rank for distributed training on gpus"
    )
    parser.add_argument(
        "--use_deepspeed", action="store_true", default=False, help="whether to use deepspeed for inference"
    )
    parser.add_argument(
        "--out_prefix",
        type=str,
        help="testset",
        required=True,
    )
    parser.add_argument(
        "--instruction",
        type=str,
        default="Generate the constituent tree for a given sentence.",
        help="testset",
    )
    parser.add_argument(
        "--prompt_template",
        type=str,
        default="None",
        help="testset",
    )
    parser.add_argument(
        "--input_key",
        type=str,
        default="sentence",
        help="input key of test set",
    )
    parser.add_argument(
        "--bit_8",
        action="store_true", default=False
    )

    args = parser.parse_args()

    return args


def create_prompt(data, args):
    if args.prompt_template == "None":
        prompts = [itm[args.input_key] for itm in data]
    elif args.prompt_template == "pubmedqa":
        prompts = [f"{sample[args.input_key].rstrip()}\nQuestion:\n{sample['QUESTION']}\nPlease respond with yes, no or maybe. The answer to the question is:" for sample in data]
    else:
        print("Invalid Prompt template, exit ...")
        exit()
    return prompts


world_size = torch.cuda.device_count()


def main():
    args = parse_args()
    model = LLM(model=args.model_name if args.model_name else args.model_name_or_path, download_dir=args.model_name_or_path, tensor_parallel_size=world_size, gpu_memory_utilization=0.95)
    print(args.test_file)
    with open(args.test_file, "r", encoding="utf-8") as fin:
        data = [json.loads(line.strip()) for line in fin]
        prompts = create_prompt(data, args)

    if args.beam_search:
        gen_params = SamplingParams(n=args.num_beams, use_beam_search=True, temperature=0.0, best_of=args.num_beams, max_tokens=args.max_new_tokens, stop=["</s>", "<pad>", "<|endoftext|>"])
    else:
        gen_params = SamplingParams(n=1, use_beam_search=False, best_of=1, temperature=0, frequency_penalty=args.frequency_penalty, presence_penalty=args.presence_penalty, max_tokens=args.max_new_tokens, stop=["</s>", "<pad>", "<|endoftext|>", "###"])

    gen_res = ["" for _ in range(len(prompts))]
    prompt_res = ["" for _ in range(len(prompts))]
    outputs = model.generate(prompts, gen_params)
    # print("Output:", outputs)
    for output in outputs:
        prompt = output.prompt
        generated_text = output.outputs[0].text
        iid = int(output.request_id)
        # print(f"Prompt: {prompt!r}\nGenerated text: {generated_text!r}")
        # gen_res.append(generated_text)
        gen_res[iid] = generated_text.replace("\n", "")
        prompt_res[iid] = prompt
    
    out_prefix = args.test_file.split("/")[-1]
    with open(f"{args.model_name_or_path}/{args.out_prefix}_{out_prefix}_pred_vllm", "w", encoding="utf-8") as fout:
        fout.write("\n".join(gen_res) + "\n")


if __name__ == "__main__":
    main()
