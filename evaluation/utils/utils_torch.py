#!/usr/bin/env python
# coding: utf-8
import torch

from torch.nn.functional import cross_entropy
from tqdm import tqdm
from transformers import AutoModelForCausalLM

torch.manual_seed(0)


def load_model(model):
    llm = AutoModelForCausalLM.from_pretrained(
        model, torch_dtype=torch.bfloat16, trust_remote_code=True, device_map="auto"
    )
    llm.eval()
    return llm


def nll(llm, batches):
    total_nll = 0
    token_cnt = 0
    indv_losses = []
    for x, y in tqdm(batches, desc="Calculating NLL", unit="batch"):
        with torch.no_grad():
            x, y = torch.tensor(x), torch.tensor(y)
            logits = llm(x.to(llm.device)).logits  # [B,T,V]

            # check for nans in logits
            if torch.isnan(logits).any():
                raise ValueError("NaNs in logits")

            logits = logits.view(-1, logits.size(-1)).float()  # [B*T,V]
            targets = y.view(-1).to(llm.device)  # [B*T]
            loss = cross_entropy(logits, targets, reduction="sum")
            total_nll += loss.item()
            token_cnt += (targets != -100).sum().item()
            indv_losses.append(loss.item())

    return (total_nll, token_cnt, indv_losses)
