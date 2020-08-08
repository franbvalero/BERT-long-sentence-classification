import math
import os
import pandas as pd
from sklearn.model_selection import train_test_split
import torch
from torch.utils.data import TensorDataset

def generate_dataset(X, Y, tokenizer, pad_to_max_length=True, add_special_tokens=True, max_length=256, return_attention_mask=True, return_tensors='pt',
    use_token_type_ids=False):
    tokens = tokenizer.batch_encode_plus(
        X, 
        pad_to_max_length=pad_to_max_length,
        add_special_tokens=add_special_tokens,
        max_length=max_length,
        return_attention_mask=return_attention_mask, # 0: padded tokens, 1: not padded tokens; taking into account the sequence length
        return_tensors=return_tensors,
    )
    if isinstance(Y, list):
        Y = torch.tensor(Y)
    if use_token_type_ids:
        dataset = TensorDataset(tokens['input_ids'], tokens['attention_mask'], tokens['token_type_ids'], Y)
    else:
        dataset = TensorDataset(tokens['input_ids'], tokens['attention_mask'], Y)
    return dataset

variations_20newsgroup = frozenset({
    "20_full",
    "20_simplified",
    "6_full",
    "6_simplified",
})
columns_20newsgroup = [
    'new', 
    'group'
]

def _load_20newsgroup(variation, max_length=512):
    root_dir = os.path.join("data","20newsgroups", variation)
    fname_train = os.path.join(root_dir, f"train_{max_length}.tsv")
    df_train = pd.read_csv(fname_train, delimiter="\t", names=columns_20newsgroup, skiprows=1)
    X_train, train_labels = df_train["new"].to_list(), df_train["group"].to_list()
    fname_test = os.path.join(root_dir, f"test_{max_length}.tsv")
    df_test = pd.read_csv(fname_test, delimiter="\t", names=columns_20newsgroup, skiprows=1)
    X_test, test_labels = df_test["new"].to_list(), df_test["group"].to_list()
    label2id = {label_:id_ for (id_, label_) in enumerate(frozenset(train_labels))}
    y_train = list(map(lambda label: label2id[label], train_labels))
    y_test = list(map(lambda label: label2id[label], test_labels))
    return X_train, y_train, X_test, y_test, label2id

def load_20newsgroup_segments(variation, max_length=512, size_segment=200, size_shift=50):
    X_train_full, y_train, X_test_full, y_test, label2id = _load_20newsgroup(variation, max_length=max_length)
    def get_segments(sentence):
        list_segments = []
        lenght_ = size_segment - size_shift
        tokens = sentence.split()
        num_tokens = len(tokens)
        num_segments = math.ceil(len(tokens) / lenght_)
        if num_tokens > lenght_:
            for i in range(0, num_tokens, lenght_):
                j = min(i+size_segment, num_tokens)
                list_segments.append(" ".join(tokens[i:j]))
        else:
                list_segments.append(sentence)    
        return list_segments, len(list_segments)
    def get_segments_from_section(sentences):
        list_segments = []
        list_num_segments = []
        for sentence in sentences:
            ls, ns = get_segments(sentence)
            list_segments += ls
            list_num_segments.append(ns)
        return list_segments, list_num_segments
    X_train, num_segments_train = get_segments_from_section(X_train_full)
    X_test, num_segments_test = get_segments_from_section(X_test_full)
    
    return X_train, y_train, num_segments_train, X_test, y_test, num_segments_test, label2id

def generate_dataset_20newsgroup_segments(X, Y, num_segments, tokenizer, pad_to_max_length=True, add_special_tokens=True, max_length=200, return_attention_mask=True, 
    return_tensors='pt'):
    tokens = tokenizer.batch_encode_plus(
        X, 
        pad_to_max_length=pad_to_max_length,
        add_special_tokens=add_special_tokens,
        max_length=max_length,
        return_attention_mask=return_attention_mask, # 0: padded tokens, 1: not padded tokens; taking into account the sequence length
        return_tensors="pt",
    )
    num_sentences = len(Y)
    max_segments = max(num_segments)
    input_ids = torch.zeros((num_sentences, max_segments, max_length), dtype=tokens["input_ids"].dtype)
    attention_mask = torch.zeros((num_sentences, max_segments, max_length), dtype=tokens["attention_mask"].dtype)
    token_type_ids = torch.zeros((num_sentences, max_segments, max_length), dtype=tokens["token_type_ids"].dtype)
    # pad_token = 0
    pos_segment = 0
    for idx_segment, n_segments in enumerate(num_segments):
        for n in range(n_segments):
            input_ids[idx_segment, n] = tokens["input_ids"][pos_segment]
            attention_mask[idx_segment, n] = tokens["attention_mask"][pos_segment]
            token_type_ids[idx_segment, n] = tokens["token_type_ids"][pos_segment]
            pos_segment += 1 
    dataset = TensorDataset(input_ids, attention_mask, token_type_ids, torch.tensor(num_segments), torch.tensor(Y))
    return dataset

def get_dataset(tokenizer, dataset, pad_to_max_length=True, add_special_tokens=True, max_length=256, return_attention_mask=True, return_tensors='pt',
    seed=42, train_size=0.9, shuffle=True, use_token_type_ids=False):
    X_train, y_train, X_test, y_test, label2id = _load_20newsgroup(dataset, max_length=max_length)
    num_train_examples, num_test_examples = len(y_train), len(y_test)
    train_set = generate_dataset(X_train, y_train, tokenizer, pad_to_max_length=pad_to_max_length, add_special_tokens=add_special_tokens, 
        max_length=max_length, return_attention_mask=return_attention_mask, return_tensors=return_tensors, use_token_type_ids=use_token_type_ids)
    test_set = generate_dataset(X_test, y_test, tokenizer, pad_to_max_length=pad_to_max_length, add_special_tokens=add_special_tokens, 
        max_length=max_length, return_attention_mask=return_attention_mask, return_tensors=return_tensors, use_token_type_ids=use_token_type_ids)
    return train_set, test_set, label2id, num_train_examples, num_test_examples
