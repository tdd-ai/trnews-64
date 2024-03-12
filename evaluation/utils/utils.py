#!/usr/bin/env python
# coding: utf-8

max_article_length = 4000  # This should not be changed.
# Otherwise, the results will be not be comparable with the results in the leaderboard.
# longer articles are filtered out to avoid different truncations in different tokenizers
# i.e. some tokenizer require more tokens to encode the same text.
# this way we ensure consistent inputs for different models


import re
import os
import math
import requests
import numpy as np

from tqdm import tqdm
from transformers import AutoTokenizer

DEFAULT_PAD_TOKEN_ID = 0
IGNORE_INDEX = (
    -100
)  # ignore_index is used to mask loss of padding tokens in cross_entropy


def get_vocab_size(model):
    if hasattr(model.config, "vocab_size"):
        vocab_size = model.config.vocab_size
        return vocab_size
    elif hasattr(model.config, "n_vocab"):
        vocab_size = model.config.n_vocab
        return vocab_size
    else:
        raise ValueError(f"Cannot determine vocab_size from {type(model)}")


def get_context_size(model):
    if hasattr(model.config, "n_ctx"):
        context_size = model.config.n_ctx
        return context_size
    elif hasattr(model.config, "n_positions"):
        context_size = model.config.n_positions
        return context_size
    elif hasattr(model.config, "max_position_embeddings"):
        context_size = model.config.max_position_embeddings
        return context_size
    else:
        raise ValueError(
            f"Cannot determine context_size from {type(model)}. Please provide one using --context_size."
        )


def load_raw(file):
    try:
        with open(file) as fi:
            articles = re.split("\n\n", fi.read())
            articles = [a for a in articles if len(a) < max_article_length]
    except FileNotFoundError:
        raise FileNotFoundError(
            f"Cannot open {file}. Make sure it exists in the current directory."
        )

    return articles


def load_trnews():
    if not os.path.exists("trnews-64.test.raw"):
        try:
            print("Downloading trnews-64.test.raw...")
            url = "https://github.com/tdd-ai/trnews-64/releases/download/v1.0/trnews-64.test.raw.gz"
            r = requests.get(url, allow_redirects=True)
            open("trnews-64.test.raw.gz", "wb").write(r.content)
            os.system("gunzip trnews-64.test.raw.gz")
        except Exception as e:
            raise Exception(f"Cannot download trnews-64.test.raw. {e}")

    try:
        with open("trnews-64.test.raw") as fi:
            articles = re.split("\n\n", fi.read())
            articles = [a for a in articles if len(a) < max_article_length]
    except FileNotFoundError:
        raise FileNotFoundError(
            f"Cannot open trnews-64.test.raw. Make sure it exists in the current directory."
        )
    
    nchars = sum(len(a) for a in articles)
    assert nchars == 6939922, f"Expected 6939922 chars, got {nchars} chars."
    return articles


def load_wikitext():
    if not os.path.exists("wikitext-2.test.raw"):
        try:
            print("Downloading wikitext-2.test.raw...")
            url = "https://github.com/tdd-ai/trnews-64/releases/download/v1.0/wikitext-2.test.raw.gz"
            r = requests.get(url, allow_redirects=True)
            open("wikitext-2.test.raw.gz", "wb").write(r.content)
            os.system("gunzip wikitext-2.test.raw.gz")
        except Exception as e:
            raise Exception(f"Cannot download wikitext-2.test.raw. {e}")

    try:
        with open("wikitext-2.test.raw") as fi:
            articles = re.split("\n\n", fi.read())
            articles = [a for a in articles if len(a) < max_article_length]
    except FileNotFoundError:
        raise FileNotFoundError(
            f"Cannot open wikitext-2.test.raw. Make sure it exists in the current directory."
        )
    
    nchars = sum(len(a) for a in articles)
    assert nchars == 1205161, f"Expected 1205161 chars, got {nchars} chars."
    return articles


def pad_inputs_and_targets(tokenized, pad_token_id, ignore_index, max_length):
    input_ids = [t[:-1] for t in tokenized]
    labels = [t[1:] for t in tokenized]

    # pad inputs and targets
    for i in range(len(input_ids)):
        input_ids[i] = input_ids[i] + [pad_token_id] * (max_length - len(input_ids[i]))
        labels[i] = labels[i] + [ignore_index] * (max_length - len(labels[i]))

    return np.array(input_ids), np.array(labels)


def batch(tokens, pad_token_id, batch_size):
    # sort by length to minimize padding
    tokens.sort(key=len, reverse=True)
    max_length = len(tokens[0])
    batches = []
    for i in range(0, len(tokens), batch_size):
        batch = tokens[i : i + batch_size]
        inputs, targets = pad_inputs_and_targets(
            batch, pad_token_id, IGNORE_INDEX, max_length
        )
        batches.append((inputs, targets))
    return batches


def tokenize(model_id, articles, batch_size=1, bos_id=None, context_size=None):
    tokenizer = AutoTokenizer.from_pretrained(model_id, use_fast=False)

    # check for eos, bos, pad tokens
    if tokenizer.bos_token_id is None and bos_id is None:
        raise ValueError(
            f"{model_id} does not have a bos token. Please provide one using --bos_id."
        )
    elif tokenizer.bos_token_id is None:
        tokenizer.bos_token_id = bos_id

    if tokenizer.pad_token_id is None:
        if tokenizer.unk_token_id is not None:
            tokenizer.pad_token_id = tokenizer.unk_token_id
        else:
            tokenizer.pad_token_id = DEFAULT_PAD_TOKEN_ID

    tokenized, nchars = [], 0

    for article in tqdm(articles, desc="Tokenizing", unit="line"):
        # we always add bos token to predict the first token
        tokens = tokenizer.encode(article, add_special_tokens=False)
        tokens = [tokenizer.bos_token_id] + tokens

        if context_size is not None and len(tokens) > context_size:
            # split into chunks of equal size. last chunk may be smaller than the rest
            chunk_size = math.ceil(len(tokens) / math.ceil(len(tokens) / context_size))
            chunks = [
                tokens[i : i + chunk_size + 1]
                for i in range(0, len(tokens), chunk_size)
            ]  # +1 for shifting
            tokenized.extend(chunks)
        else:
            tokenized.append(tokens)

        nchars += len(article)

    batched = batch(tokenized, tokenizer.pad_token_id, batch_size)
    return batched, tokenized, nchars
