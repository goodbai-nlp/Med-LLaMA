# Copyright (c) Microsoft Corporation.
# SPDX-License-Identifier: Apache-2.0

# DeepSpeed Team
from dataclasses import dataclass
from datasets import load_dataset
from typing import Any, Callable, Dict, List, NewType, Optional, Tuple, Union
from transformers.tokenization_utils_base import PreTrainedTokenizerBase
from transformers.utils import PaddingStrategy
from torch.utils.data import Subset
import re


def padding_func(
    features,
    padding_side="right",
    pad_token_id=1,
    key="label",
    pad_to_multiple_of=1,
    max_length=None,
):
    assert key in features[0].keys(), f"{key} not in {features[0].keys()}"
    max_label_length = max(len(feature[key]) for feature in features)
    if pad_to_multiple_of > 1:
        if max_length is not None:
            max_label_length = min(
                max_length,
                (max_label_length + pad_to_multiple_of - 1)
                // pad_to_multiple_of
                * pad_to_multiple_of,
            )
        else:
            max_label_length = (
                (max_label_length + pad_to_multiple_of - 1)
                // pad_to_multiple_of
                * pad_to_multiple_of
            )

    for feature in features:
        remainder = [pad_token_id] * (max_label_length - len(feature[key]))
        feature[key] = (
            feature[key] + remainder if padding_side == "right" else remainder + feature[key]
        )
    return


@dataclass
class DataCollatorForCausalLM:
    """
    Data collator that will dynamically pad the inputs received.

    Args:
        tokenizer ([`PreTrainedTokenizer`] or [`PreTrainedTokenizerFast`]):
            The tokenizer used for encoding the data.
        padding (`bool`, `str` or [`~utils.PaddingStrategy`], *optional*, defaults to `True`):
            Select a strategy to pad the returned sequences (according to the model's padding side and padding index)
            among:

            - `True` or `'longest'` (default): Pad to the longest sequence in the batch (or no padding if only a single
              sequence is provided).
            - `'max_length'`: Pad to a maximum length specified with the argument `max_length` or to the maximum
              acceptable input length for the model if that argument is not provided.
            - `False` or `'do_not_pad'`: No padding (i.e., can output a batch with sequences of different lengths).
        max_length (`int`, *optional*):
            Maximum length of the returned list and optionally padding length (see above).
        pad_to_multiple_of (`int`, *optional*):
            If set will pad the sequence to a multiple of the provided value.

            This is especially useful to enable the use of Tensor Cores on NVIDIA hardware with compute capability >=
            7.5 (Volta).
        return_tensors (`str`):
            The type of Tensor to return. Allowable values are "np", "pt" and "tf".
    """

    tokenizer: PreTrainedTokenizerBase
    padding: Union[bool, str, PaddingStrategy] = True
    max_length: Optional[int] = None
    pad_to_multiple_of: Optional[int] = None
    return_tensors: str = "pt"
    label_pad_token_id: int = -100

    def __call__(self, features: List[Dict[str, Any]]) -> Dict[str, Any]:
        if "labels" in features[0].keys():
            padding_func(
                features,
                padding_side=self.tokenizer.padding_side,
                pad_token_id=self.label_pad_token_id,
                key="labels",
                pad_to_multiple_of=self.pad_to_multiple_of,
            )
        batch = self.tokenizer.pad(
            features,
            padding=self.padding,
            max_length=self.max_length,
            pad_to_multiple_of=self.pad_to_multiple_of,
            return_tensors=self.return_tensors,
        )
        if "labels" not in batch:
            batch["labels"] = batch["input_ids"].clone()
        return batch
    

def get_raw_dataset(dataset_name, output_path, seed, local_rank):
    if dataset_name.endswith("pubmed-abs"):
        return PubMedDataset(output_path, seed, local_rank, dataset_name)
    elif dataset_name.endswith("pubmedqa"):
        return TrucatedPubMedQADataset(output_path, seed, local_rank, dataset_name)
    else:
        raise RuntimeError(
            f"We do not have configs for dataset {dataset_name}, but you can add it by yourself in raw_datasets.py."
        )


# The template prompt dataset class that all new dataset porting needs to
# follow in order to have a unified API and unified data format.
class PromptRawDataset(object):
    def __init__(self, output_path, seed, local_rank, data_path, eos_token=""):
        self.output_path = output_path
        self.seed = seed
        self.local_rank = local_rank
        # self.eos_token = "<|endoftext|>"
        self.eos_token = eos_token
        self.raw_datasets = load_dataset(data_path)

    def get_train_data(self):
        return

    def get_eval_data(self):
        return

    # The prompt should be in the format of: " Human: " + actual_prompt_sentence + " Assistant:"
    def get_prompt(self, sample):
        return

    # The chosen response should be in the format of: " " + actual_response_sentence
    def get_chosen(self, sample):
        return

    # The rejected response should be in the format of: " " + actual_response_sentence
    # If the dataset does not have rejected response, return None
    def get_rejected(self, sample):
        return

    def get_prompt_and_chosen(self, sample):
        return

    def get_prompt_and_rejected(self, sample):
        return


class PubMedQADataset(PromptRawDataset):
    def __init__(self, output_path, seed, local_rank, data_path):
        super().__init__(output_path, seed, local_rank, data_path)
        self.dataset_name = "pubmedqa"
        self.dataset_name_clean = "pubmedqa"
        self.input_key = "input"
        self.output_key = "final_decision"
        print("Loaded dataset", self.raw_datasets)

    def get_train_data(self):
        return self.raw_datasets["train"]

    def get_eval_data(self):
        return self.raw_datasets["validation"]

    def get_prompt(self, sample):
        assert len(sample["CONTEXTS"]) == len(sample["LABELS"])
        context = " ".join([f'{topic.lower()}: {con}' for con, topic in zip(sample["CONTEXTS"], sample["LABELS"])][:250])
        return f"CONTEXT:\n{context}\nQUESTION:\n{sample['QUESTION']}\nANSWER:\n{sample['LONG_ANSWER']}:\nGiven the CONTEXT, QUESTION and ANSWER, judge whether the provided ANSWER correctly addresses the QUESTION under the given CONTEXT. Please output yes, no or maybe. The output is:"

    def get_chosen(self, sample):
        return sample[self.output_key]

    def get_rejected(self, sample):
        print(f"Warning: dataset {self.dataset_name} does not include rejected response.")
        return None

    def get_prompt_and_chosen(self, sample):
        return f"{self.get_prompt(sample)}{sample[self.output_key]}"

    def get_prompt_and_rejected(self, sample):
        print(f"Warning: dataset {self.dataset_name} does not include rejected response.")
        return None

    def process_function(self, samples):
        input_prompt = [
            self.get_prompt({"CONTEXTS": context, "QUESTION": question, "LABELS": label, "LONG_ANSWER": answer})
            for context, question, label, answer in zip(
                samples["CONTEXTS"], samples["QUESTION"], samples["LABELS"], samples["LONG_ANSWER"]
            )
        ]
        input_full = [
            self.get_prompt_and_chosen({"CONTEXTS": context, "QUESTION": question, "LABELS": label, "LONG_ANSWER": answer, "final_decision": decision})
            for context, question, label, answer, decision in zip(
                samples["CONTEXTS"], samples["QUESTION"], samples["LABELS"], samples["LONG_ANSWER"], samples["final_decision"]
            )
        ]
        return {"text": input_full, "prompt": input_prompt}


class TrucatedPubMedQADataset(PromptRawDataset):
    def __init__(self, output_path, seed, local_rank, data_path):
        super().__init__(output_path, seed, local_rank, data_path)
        self.dataset_name = "trucated-pubmedqa"
        self.dataset_name_clean = "trucated-pubmedqa"
        self.input_key = "context"
        self.output_key = "final_decision"
        print("Loaded dataset", self.raw_datasets)

    def get_train_data(self):
        return self.raw_datasets["train"]

    def get_eval_data(self):
        return self.raw_datasets["validation"]

    def get_prompt(self, sample):
        return f"{sample['context'].rstrip()}\nQuestion:\n{sample['QUESTION']}\nPlease respond with yes, no or maybe. The answer to the question is:"

    def get_chosen(self, sample):
        return sample[self.output_key]

    def get_rejected(self, sample):
        print(f"Warning: dataset {self.dataset_name} does not include rejected response.")
        return None

    def get_prompt_and_chosen(self, sample):
        return f"{self.get_prompt(sample)}{sample[self.output_key]}"

    def get_prompt_and_rejected(self, sample):
        print(f"Warning: dataset {self.dataset_name} does not include rejected response.")
        return None

    def process_function(self, samples):
        input_prompt = [
            self.get_prompt({"context": context, "QUESTION": question})
            for context, question in zip(
                samples["context"], samples["QUESTION"]
            )
        ]
        input_full = [
            self.get_prompt_and_chosen({"context": context, "QUESTION": question, "final_decision": decision})
            for context, question, decision in zip(
                samples["context"], samples["QUESTION"], samples["final_decision"]
            )
        ]
        return {"text": input_full, "prompt": input_prompt}


class PubMedDataset(PromptRawDataset):
    def __init__(self, output_path, seed, local_rank, data_path):
        super().__init__(output_path, seed, local_rank, data_path)
        self.dataset_name = "pubmed-abs"
        self.dataset_name_clean = "pubmed-abs"
        print("Loaded dataset", self.raw_datasets)

    def get_train_data(self):
        return self.raw_datasets["train"]

    def get_eval_data(self):
        return self.raw_datasets["test"]

    def get_prompt(self, sample):
        return sample["abstract"]

    def get_chosen(self, sample):
        return sample["abstract"]

    def get_rejected(self, sample):
        print(f"Warning: dataset {self.dataset_name} does not include rejected response.")
        return None

    def get_prompt_and_chosen(self, sample):
        print(f"Warning: dataset {self.dataset_name} does not include chosen response.")

    def get_prompt_and_rejected(self, sample):
        print(f"Warning: dataset {self.dataset_name} does not include rejected response.")
        return None

    def process_function(self, samples):
        input_text = samples["abstract"]
        return {"text": input_text}