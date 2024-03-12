#!/usr/bin/env python
# coding: utf-8
import flax
import optax
import jax.numpy as jnp

from tqdm import tqdm
from transformers import FlaxAutoModelForCausalLM


def load_model(model):
    llm = FlaxAutoModelForCausalLM.from_pretrained(model, dtype=jnp.bfloat16)

    # convert weights to bfloat16
    flat_params = flax.traverse_util.flatten_dict(llm.params)
    mask = {
        path: not (path[-2][:2] == "ln" and path[-1] in ("bias", "scale"))
        for path in flat_params
    }
    mask = flax.traverse_util.unflatten_dict(mask)
    llm.params = llm.to_bf16(llm.params, mask)
    return llm


def loss_fn(llm, inputs, targets):
    logits = llm(inputs).logits  # [B,T,V]
    logits = logits.reshape(-1, logits.shape[-1])  # [B*T,V]
    targets = targets.reshape(-1)  # [B*T]
    loss = optax.softmax_cross_entropy_with_integer_labels(logits, targets)
    # mask padding tokens
    mask = targets != -100 
    loss = loss * mask
    return loss.sum(), mask.sum()


def nll(llm, batches):

    total_nll = 0.0
    token_cnt = 0
    indv_losses = []
    for x, y in tqdm(batches, desc="Calculating NLL", unit="batch"):
        loss, mask = loss_fn(llm, x, y)
        total_nll += loss.item()
        token_cnt += mask.item()
        indv_losses.append(loss.item())

    return (total_nll, token_cnt, indv_losses)
