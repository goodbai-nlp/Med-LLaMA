# coding:utf-8
import re
import json
import torch


def convert_text(text):
    # return text
    text = " ".join(re.split("(\W)", text))
    text = " ".join(text.split())
    return text


def shift_tokens_right(input_ids: torch.Tensor, pad_token_id: int, decoder_start_token_id: int):
    """
    Shift input ids one token to the right.
    """
    shifted_input_ids = input_ids.new_zeros(input_ids.shape)
    shifted_input_ids[:, 1:] = input_ids[:, :-1].clone()
    shifted_input_ids[:, 0] = decoder_start_token_id

    assert pad_token_id is not None, "self.model.config.pad_token_id has to be defined."
    # replace possible -100 values in labels by `pad_token_id`
    shifted_input_ids.masked_fill_(shifted_input_ids == -100, pad_token_id)
    return shifted_input_ids


def trim_batch(input_ids, pad_token_id, attention_mask=None):
    """Remove columns that are populated exclusively by pad_token_id"""
    keep_column_mask = input_ids.ne(pad_token_id).any(dim=0)
    if attention_mask is None:
        return input_ids[:, keep_column_mask]
    else:
        return (input_ids[:, keep_column_mask], attention_mask[:, keep_column_mask])


def copy_weights(src, tgt):
    src_params = dict(src.named_parameters())
    tgt_params = tgt.named_parameters()
    for name, param in tgt_params:
        assert name in src_params, f"{name} not in {[n for n,v in src_params]}"
        param.data = src_params[name].data


def save_dummy_batch(batch, tokenizer, output_dir, cate="train"):
    print("Saving dummy inputs...")
    json_out_path = open(output_dir + f"/dummy_{cate}_input.json", "w", encoding="utf-8")
    ith_dict = {}
    for k, v in batch.items():
        if "_ids" in k and v is not None:
            ith_dict[k] = str(v.tolist())
            ith_dict[k.replace("ids", "tokens")] = tokenizer.batch_decode(v.tolist(), clean_up_tokenization_spaces=False)
        elif "labels" in k:
            ith_dict[k] = str(v.tolist())
            label_data_new = [[idx if idx != -100 else tokenizer.pad_token_id for idx in ith_label] for ith_label in v.tolist()]
            ith_dict[k+"_tokens"] = tokenizer.batch_decode(label_data_new, clean_up_tokenization_spaces=False)
        else:
            print(f"Skiping {k}...")
    json.dump(ith_dict, json_out_path, indent=4)
