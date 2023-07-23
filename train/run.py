#!/usr/bin/env python
# coding=utf-8
"""
This file is modified from the huggingface example for finetuning language models
[run_clm.py](https://github.com/huggingface/transformers/blob/main/examples/pytorch/language-modeling/run_clm.py)
"""

import logging
import os
import sys
import math
import random
from dataclasses import dataclass, field
from typing import Optional
from functools import partial
import datasets
import torch
from datasets import load_dataset, load_from_disk
from data_interface.dataset import get_raw_dataset, DataCollatorForCausalLM
from torch.utils.data import DataLoader, RandomSampler, SequentialSampler, ConcatDataset
import transformers
from transformers import (
    AutoConfig,
    AutoModelForCausalLM,
    AutoTokenizer,
    LlamaTokenizer,
    HfArgumentParser,
    DataCollatorForSeq2Seq,
    set_seed,
    GPTNeoXTokenizerFast,
    GPT2Tokenizer,
    OPTForCausalLM
)
from transformers.trainer_utils import get_last_checkpoint
from lm_trainer import LMTrainer
from common.options import LLMTrainingArguments

logger = logging.getLogger(__name__)


@dataclass
class ModelArguments:
    """
    Arguments pertaining to which model/config/tokenizer we are going to fine-tune, or train from scratch.
    """

    model_name_or_path: Optional[str] = field(
        default=None,
        metadata={
            "help": (
                "The model checkpoint for weights initialization. Don't set if you want to train a model from scratch."
            )
        },
    )
    config_name: Optional[str] = field(
        default=None, metadata={"help": "Pretrained config name or path if not the same as model_name"}
    )
    tokenizer_name: Optional[str] = field(
        default=None, metadata={"help": "Pretrained tokenizer name or path if not the same as model_name"}
    )
    cache_dir: Optional[str] = field(
        default=None,
        metadata={"help": "Where do you want to store the pretrained models downloaded from huggingface.co"},
    )
    use_fast_tokenizer: bool = field(
        default=False,
        metadata={"help": "Whether to use one of the fast tokenizer (backed by the tokenizers library) or not."},
    )
    model_revision: str = field(
        default="main",
        metadata={"help": "The specific model version to use (can be a branch name, tag name or commit id)."},
    )
    use_auth_token: bool = field(
        default=False,
        metadata={
            "help": (
                "Will use the token generated when running `huggingface-cli login` (necessary to use this script "
                "with private models)."
            )
        },
    )
    torch_dtype: Optional[str] = field(
        default=None,
        metadata={
            "help": (
                "Override the default `torch.dtype` and load the model under this dtype. If `auto` is passed, the "
                "dtype will be automatically derived from the model's weights."
            ),
            "choices": ["auto", "bfloat16", "float16", "float32"],
        },
    )


@dataclass
class DataTrainingArguments:
    """
    Arguments pertaining to what data we are going to input our model for training and eval.
    """
    data_path: Optional[str] = field(
        default="Dahoas/rm-static",
        metadata={"help":'Path to the training dataset. Accepted format: 1) a single data path, 2) multiple datasets in the form: dataset1-path dataset2-path ...'}
    )
    dataset_name: Optional[str] = field(
        default=None, metadata={"help": "The name of the dataset to use (via the datasets library)."}
    )
    dataset_config_name: Optional[str] = field(
        default=None, metadata={"help": "The configuration name of the dataset to use (via the datasets library)."}
    )
    max_train_samples: Optional[int] = field(
        default=None,
        metadata={
            "help": (
                "For debugging purposes or quicker training, truncate the number of training examples to this "
                "value if set."
            )
        },
    )
    streaming: bool = field(default=False, metadata={"help": "Enable streaming mode"})
    data_cache_dir: Optional[str] = field(
        default="./cache", metadata={"help": "The configuration name of the dataset to use (via the datasets library)."}
    )
    overwrite_cache: bool = field(
        default=False, metadata={"help": "Overwrite the cached training and evaluation sets"}
    )
    preprocessing_num_workers: Optional[int] = field(
        default=None,
        metadata={"help": "The number of processes to use for the preprocessing."},
    )
    max_seq_length: Optional[int] = field(
        default=None,
        metadata={
            "help": ("The maximum total input sequence length after tokenization. Sequences longer than this will be truncated,")
        },
    )
    bit8_training: Optional[bool] = field(
        default=False,
        metadata={"help": "The number of processes to use for the preprocessing."},
    )

    def __post_init__(self):
        if self.dataset_name is None and self.data_path is None:
            raise ValueError("Need either a dataset name or a training file.")
        else:
            if self.data_path is not None:
                self.data_path = self.data_path.split()


def main():
    parser = HfArgumentParser((ModelArguments, DataTrainingArguments, LLMTrainingArguments))
    if len(sys.argv) == 2 and sys.argv[1].endswith(".json"):
        model_args, data_args, training_args = parser.parse_json_file(json_file=os.path.abspath(sys.argv[1]))
    else:
        model_args, data_args, training_args = parser.parse_args_into_dataclasses()

    # Setup logging
    logging.basicConfig(
        format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
        datefmt="%m/%d/%Y %H:%M:%S",
        handlers=[logging.StreamHandler(sys.stdout)],
    )

    if training_args.should_log:
        # The default of training_args.log_level is passive, so we set log level at info here to have that default.
        transformers.utils.logging.set_verbosity_info()

    log_level = training_args.get_process_log_level()
    logger.setLevel(log_level)
    datasets.utils.logging.set_verbosity(log_level)
    transformers.utils.logging.set_verbosity(log_level)
    transformers.utils.logging.enable_default_handler()
    transformers.utils.logging.enable_explicit_format()

    # Log on each process the small summary:
    logger.warning(
        f"Process rank: {training_args.local_rank}, device: {training_args.device}, n_gpu: {training_args.n_gpu}"
        + f"distributed training: {bool(training_args.local_rank != -1)}, 16-bits training: {training_args.fp16}"
    )
    logger.info(f"Training parameters {training_args}")

    # Detecting last checkpoint.
    last_checkpoint = None
    if os.path.isdir(training_args.output_dir) and training_args.do_train and not training_args.overwrite_output_dir:
        last_checkpoint = get_last_checkpoint(training_args.output_dir)
        if last_checkpoint is None and len(os.listdir(training_args.output_dir)) > 0:
            raise ValueError(
                f"Output directory ({training_args.output_dir}) already exists and is not empty. "
                "Use --overwrite_output_dir to overcome."
            )
        elif last_checkpoint is not None and training_args.resume_from_checkpoint is None:
            logger.info(
                f"Checkpoint detected, resuming training at {last_checkpoint}. To avoid this behavior, change "
                "the `--output_dir` or add `--overwrite_output_dir` to train from scratch."
            )

    # Set seed before initializing model.
    set_seed(training_args.seed)
    
    config_kwargs = {
        "cache_dir": model_args.cache_dir,
        "revision": model_args.model_revision,
        "use_auth_token": True if model_args.use_auth_token else None,
    }
    if model_args.config_name:
        config = AutoConfig.from_pretrained(model_args.config_name, **config_kwargs)
    elif model_args.model_name_or_path:
        config = AutoConfig.from_pretrained(model_args.model_name_or_path, **config_kwargs)
    else:
        raise ValueError(
            "You are instantiating a new config instance from scratch. This is not supported by this finetuning script."
        )

    tokenizer_kwargs = {
        "cache_dir": model_args.cache_dir,
        "use_fast": model_args.use_fast_tokenizer,
        "revision": model_args.model_revision,
        "use_auth_token": True if model_args.use_auth_token else None,
        "add_eos_token": True,                                                                      # eos_token is necessary for fine-tuning
    }
    if model_args.tokenizer_name:
        tokenizer = AutoTokenizer.from_pretrained(model_args.tokenizer_name, **tokenizer_kwargs)
    elif model_args.model_name_or_path:
        tokenizer = AutoTokenizer.from_pretrained(model_args.model_name_or_path, **tokenizer_kwargs)
    else:
        raise ValueError(
            "You are instantiating a new tokenizer from scratch. This is not supported by this fine-tuning script."
        )
    # no default pad token for llama!
    # here we add all special tokens again, because the default ones are not in the special_tokens_map
    if isinstance(tokenizer, LlamaTokenizer):
        num_added_tokens = tokenizer.add_special_tokens({
            "bos_token": "<s>",
            "eos_token": "</s>",
            "unk_token": "<unk>",
        })
        assert num_added_tokens == 0, "LlamaTokenizer should only add zero special token"
        tokenizer.pad_token = "<pad>"
        tokenizer.pad_token_id = tokenizer.eos_token_id
        # exit()
        
    elif isinstance(tokenizer, GPTNeoXTokenizerFast):
        num_added_tokens = tokenizer.add_special_tokens({
            "pad_token": "<pad>",
        })
        assert num_added_tokens == 1, "GPTNeoXTokenizer should only add one special token - the pad_token."
    elif isinstance(tokenizer, GPT2Tokenizer) and isinstance(model, OPTForCausalLM):
        num_added_tokens = tokenizer.add_special_tokens({'unk_token': '<unk>'})
    else:
        logger.info(f"Unsupportted Tokenizer Type:{type(tokenizer)}")
        exit()
        
    def tokenize_function_lm(examples):
        return tokenizer(
            examples["text"], max_length=data_args.max_seq_length, padding=False, truncation=True
        )
    
    def tokenize_function_clm(examples):
        inputs = tokenizer(
            examples["text"], max_length=data_args.max_seq_length, padding=False, truncation=True
        )
        prompts = tokenizer(
            examples["prompt"], max_length=data_args.max_seq_length, padding=False, truncation=True
        )
        labels = [
            [-100 for _ in range(len(prompt_ids)-1)] + input_ids[(len(prompt_ids)-1):] for input_ids, prompt_ids in zip(inputs["input_ids"], prompts["input_ids"])
        ]
        inputs["labels"] = labels
        return inputs
    
    tokenize_function = tokenize_function_clm if training_args.conditional_gen else tokenize_function_lm
    
    assert len(data_args.data_path) >= 1, "Invalid dataset, there should at least one dataset"
    train_cache_path = data_args.data_cache_dir + "/train-dump"
    valid_cache_path = data_args.data_cache_dir + "/valid-dump"
    cache_found = os.path.exists(train_cache_path) and os.path.exists(valid_cache_path)
    with training_args.main_process_first(desc="Processing instruction data"):
        if not cache_found or data_args.overwrite_cache:
            # with torch_distributed_zero_first:  # only preprocess and cache dataset on main process
            train_datasets, eval_datasets = [], []
            for dataset_itm in data_args.data_path:
                raw_dataset = get_raw_dataset(
                    dataset_itm,
                    data_args.data_cache_dir,
                    training_args.seed,
                    training_args.local_rank,
                )
                train_dataset = raw_dataset.get_train_data()
                eval_dataset = raw_dataset.get_eval_data()
                column_names = list(train_dataset.features)

                if hasattr(raw_dataset, "process_function"):
                    if not data_args.streaming:
                        train_dataset = train_dataset.map(
                            raw_dataset.process_function,
                            batched=True,
                            num_proc=data_args.preprocessing_num_workers,
                            remove_columns=column_names,
                            load_from_cache_file=not data_args.overwrite_cache,
                            desc="Running preprocessing on train dataset",
                        )
                    else:
                        train_dataset = train_dataset.map(
                            raw_dataset.process_function,
                            batched=True,
                            remove_columns=column_names,
                        )
                if eval_dataset:
                    if not data_args.streaming:
                        eval_dataset = eval_dataset.map(
                            raw_dataset.process_function,
                            batched=True,
                            num_proc=data_args.preprocessing_num_workers,
                            remove_columns=column_names,
                            load_from_cache_file=not data_args.overwrite_cache,
                            desc="Running preprocessing on valid dataset",
                        )
                    else:
                        eval_dataset = eval_dataset.map(
                            raw_dataset.process_function,
                            batched=True,
                            remove_columns=column_names,
                        )
                    eval_datasets.append(eval_dataset)
                train_datasets.append(train_dataset)

            train_dataset = ConcatDataset(train_datasets) if len(train_datasets) > 1 else train_datasets[0]
            eval_dataset = ConcatDataset(eval_datasets) if len(eval_datasets) > 1 else eval_datasets[0]

            column_names = list(train_dataset.features)
            if not data_args.streaming:
                train_dataset = train_dataset.map(
                    tokenize_function,
                    batched=True,
                    num_proc=data_args.preprocessing_num_workers,
                    remove_columns=column_names,
                    load_from_cache_file=not data_args.overwrite_cache,
                    desc="Running tokenizer on train dataset",
                )
                eval_dataset = eval_dataset.map(
                    tokenize_function,
                    batched=True,
                    num_proc=data_args.preprocessing_num_workers,
                    remove_columns=column_names,
                    load_from_cache_file=not data_args.overwrite_cache,
                    desc="Running tokenizer on valid dataset",
                )
            else:
                train_dataset = train_dataset.map(
                    tokenize_function,
                    batched=True,
                    remove_columns=column_names,
                )
                eval_dataset = eval_dataset.map(
                    tokenize_function,
                    batched=True,
                    remove_columns=column_names,
                )
            print("Saving cached test data ...")
            train_dataset.save_to_disk(train_cache_path)
            eval_dataset.save_to_disk(valid_cache_path)

    logger.info("Loading from cache ...")
    train_dataset = load_from_disk(train_cache_path)
    eval_dataset = load_from_disk(valid_cache_path)
    
    if model_args.model_name_or_path:
        torch_dtype = (
            model_args.torch_dtype
            if model_args.torch_dtype in ["auto", None]
            else getattr(torch, model_args.torch_dtype)
        )
        model = AutoModelForCausalLM.from_pretrained(
            model_args.model_name_or_path,
            from_tf=bool(".ckpt" in model_args.model_name_or_path),
            config=config,
            cache_dir=model_args.cache_dir,
            revision=model_args.model_revision,
            use_auth_token=True if model_args.use_auth_token else None,
            torch_dtype=torch_dtype,
            load_in_8bit=True if data_args.bit8_training else False,
        )
    else:
        logger.warning("No pretrained model_name_or_path is given. Training new model from scratch.")
        model = AutoModelForCausalLM.from_config(config)
        n_params = sum({p.data_ptr(): p.numel() for p in model.parameters()}.values())
        logger.info(f"Training new model from scratch - Total size={n_params/2**20:.2f}M params")

    # resize embeddings if needed (e.g. for LlamaTokenizer)
    embedding_size = model.get_input_embeddings().weight.shape[0]
    if len(tokenizer) > embedding_size:
        model.resize_token_embeddings(int(8 * math.ceil(len(tokenizer) / 8.0)))         # make the vocab size multiple of 8
        # model.resize_token_embeddings(len(tokenizer))

    if training_args.do_train:
        if not train_dataset:
            raise ValueError("--do_train requires a train dataset")
        if data_args.max_train_samples is not None:
            max_train_samples = min(len(train_dataset), data_args.max_train_samples)
            train_dataset = train_dataset.select(range(max_train_samples))
    
    for index in random.sample(range(len(train_dataset)), 3):
        logger.info(f"Sample {index} of the training set: {train_dataset[index]}.")

    data_collator = DataCollatorForCausalLM(
        tokenizer,
        padding="longest",
        max_length=data_args.max_seq_length,
        pad_to_multiple_of=8,
    )
    
    # initalize a trainer
    # here we use a custom trainer that moves the model to CPU when saving the checkpoint in FSDP mode
    # we can switch to the default trainer after moving to deepspeed (let's don't change too much for now)
    
    trainer = LMTrainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset if training_args.do_train else None,
        eval_dataset=eval_dataset if training_args.do_eval else None,
        tokenizer=tokenizer,
        data_collator=data_collator,
        compute_metrics=None,
    )

    # Training
    if training_args.do_train:
        checkpoint = None
        if training_args.resume_from_checkpoint is not None:
            checkpoint = training_args.resume_from_checkpoint
        elif last_checkpoint is not None:
            checkpoint = last_checkpoint
        train_result = trainer.train(resume_from_checkpoint=checkpoint)
        trainer.save_model(f"{training_args.output_dir}/checkpoint-last/")                    # Saves the tokenizer too for easy upload

        metrics = train_result.metrics

        max_train_samples = (
            data_args.max_train_samples if data_args.max_train_samples is not None else len(train_dataset)
        )
        metrics["train_samples"] = min(max_train_samples, len(train_dataset))

        trainer.log_metrics("train", metrics)
        trainer.save_metrics("train", metrics)
        trainer.save_state()

    kwargs = {"finetuned_from": model_args.model_name_or_path, "tasks": "text-generation"}
    if data_args.dataset_name is not None:
        kwargs["dataset_tags"] = data_args.dataset_name
        if data_args.dataset_config_name is not None:
            kwargs["dataset_args"] = data_args.dataset_config_name
            kwargs["dataset"] = f"{data_args.dataset_name} {data_args.dataset_config_name}"
        else:
            kwargs["dataset"] = data_args.dataset_name

    if training_args.push_to_hub:
        trainer.push_to_hub(**kwargs)
    else:
        trainer.create_model_card(**kwargs)


if __name__ == "__main__":
    main()