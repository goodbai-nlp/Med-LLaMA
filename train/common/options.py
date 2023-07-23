# coding:utf-8

from transformers.training_args import TrainingArguments
from typing import Any, Dict, List, Optional, Union
from dataclasses import asdict, dataclass, field, fields


@dataclass
class LLMTrainingArguments(TrainingArguments):
    """
    Args:
        sortish_sampler (`bool`, *optional*, defaults to `False`):
            Whether to use a *sortish sampler* or not. Only possible if the underlying datasets are *Seq2SeqDataset*
            for now but will become generally available in the near future.
            It sorts the inputs according to lengths in order to minimize the padding size, with a bit of randomness
            for the training set.
        predict_with_generate (`bool`, *optional*, defaults to `False`):
            Whether to use generate to calculate generative metrics (ROUGE, BLEU).
        generation_max_length (`int`, *optional*):
            The `max_length` to use on each evaluation loop when `predict_with_generate=True`. Will default to the
            `max_length` value of the model configuration.
        generation_num_beams (`int`, *optional*):
            The `num_beams` to use on each evaluation loop when `predict_with_generate=True`. Will default to the
            `num_beams` value of the model configuration.
    """
    eval_dataloader_num_workers: int = field(
        default=0,
        metadata={
            "help": (
                "Number of subprocesses to use for data loading (PyTorch only). 0 means that the data will be loaded"
                " in the main process."
            )
        },
    )
    sortish_sampler: bool = field(
        default=False, metadata={"help": "Whether to use SortishSampler or not."}
    )
    early_stopping: Optional[int] = field(
        default=5, metadata={"help": "Early stopping patience for training"}
    )
    smart_init: bool = field(
        default=False,
        metadata={"help": "Whether to use initialize AMR embeddings with their sub-word embeddings."},
    )
    predict_with_generate: bool = field(
        default=False,
        metadata={"help": "Whether to use generate to calculate generative metrics (ROUGE, BLEU)."},
    )
    task: str = field(
        default="amr2text",
        metadata={"help": "The name of the task, (amr2text or text2amr)."},
    )
    generation_max_length: Optional[int] = field(
        default=None,
        metadata={
            "help": (
                "The `max_length` to use on each evaluation loop when `predict_with_generate=True`. Will default "
                "to the `max_length` value of the model configuration."
            )
        },
    )
    generation_num_beams: Optional[int] = field(
        default=None,
        metadata={
            "help": (
                "The `num_beams` to use on each evaluation loop when `predict_with_generate=True`. Will default "
                "to the `num_beams` value of the model configuration."
            )
        },
    )
    length_penalty: Optional[float] = field(
        default=1.0, metadata={"help": "Exponential penalty to the length that is used with beam-based generation"}
    )
    max_new_tokens: Optional[int] = field(
        default=128,
        metadata={
            "help": (
                "The maximum new generated sequence length (excluding prompt)"
            )
        },
    )
    min_length: Optional[int] = field(
        default=0,
        metadata={
            "help": (
                "The minimum length of the sequence to be generated"
            )
        },
    )
    ignore_opt_states: bool = field(
        default=False,
        metadata={
            "help": (
                "When doing a multinode distributed training, whether to log once per node or just once on the main"
                " node."
            )
        },
    )
    conditional_gen: bool = field(
        default=False,
        metadata={
            "help": (
                "When doing a multinode distributed training, whether to log once per node or just once on the main"
                " node."
            )
        },
    )
