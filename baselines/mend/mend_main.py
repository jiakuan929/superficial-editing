import os
from copy import deepcopy
from typing import Dict, List

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from ..util.globals import *

from ..trainer.algs import MEND


class MendRewriteExecutor:
    method_name = "MEND"

    def __init__(self):
        self.is_init = False

    def init_model(self, model, tok, params):
        assert params.archive is not None or print(f"Training weights Needed...")

        def add_padding(tokenizer, model):
            tokenizer.add_special_tokens({"pad_token": "[PAD]"})
            model.resize_token_embeddings(len(tokenizer))
            model.transformer.wte.weight.data[
                -1
            ] = model.transformer.wte.weight.data.mean(0)

        # Customize the gpt2xl and tokenizer
        self.model = model
        self.tokenizer = tok

        # Load the trained MEND model
        self.alg = MEND(self.model, params, lambda: deepcopy(self.model))
        d = torch.load(params.archive)
        self.alg.load_state_dict(
            {k.replace("gtn.", "mend."): v for k, v in d["model"].items()}
        )
        self.alg.cuda()

        # Disable unneeded gradients
        for n, p in self.model.named_parameters():
            if n not in params.inner_params:
                p.requires_grad = False
        self.is_init = True

    def reset_model(self):
        self.is_init = False
        del self.model, self.tokenizer, self.alg

    def apply_to_model(
        self,
        model: AutoModelForCausalLM,
        tok: AutoTokenizer,
        requests: List[Dict],
        hparams,
        copy=False,
        return_orig_weights=False,
    ):
        """
        Given a request, for example
        {'prompt': '{} has the position of',
         'subject': 'Charles Herman Helmsing',
         'relation_id': 'P39',
         'target_new': {'str': 'President', 'id': 'Q11696'},
         'target_true': {'str': 'bishop', 'id': 'Q29182'}}
        Returns a dictionary of numpy arrays that specifies
        how mend will change the weights of the model.
        """

        if not self.is_init:
            self.init_model(model, tok, hparams)

        weights_copy = {}
        model = deepcopy(self.model) if copy else self.model

        # Define i/o
        targets = [
            (" " if request["target_new"]["str"][0] != " " else "")
            + request["target_new"]["str"]
            for request in requests
        ]
        sentences = [
            request["prompt"].format(request["subject"]) + targets[i]
            for i, request in enumerate(requests)
        ]

        # Tokenize
        sent_tok = self.tokenizer(sentences, padding=True, return_tensors="pt").to(
            "cuda"
        )
        target_tok = self.tokenizer(targets, padding=True, return_tensors="pt").to(
            "cuda"
        )

        # Define labels
        label_tok = deepcopy(sent_tok["input_ids"])
        for i in range(label_tok.size(0)):
            target_len = target_tok["attention_mask"][i].sum()
            padding_len = (
                sent_tok["input_ids"].size(1) - sent_tok["attention_mask"][i].sum()
            )
            label_tok[i][: -target_len - padding_len] = -100
            label_tok[i][label_tok[i] == self.tokenizer.pad_token_id] = -100

        # Run MEND
        edit_inner = dict(
            input_ids=sent_tok["input_ids"],
            attention_mask=sent_tok["attention_mask"],
            labels=target_tok['input_ids'],
        )
        cond = {k: sent_tok[k] for k in ["input_ids", "attention_mask"]}
        _, model_info = self.alg.edit(edit_inner, cond, return_factors=True)
        factors = {
            k + "." + n: v.detach().cpu().numpy()
            for k, pair in model_info["factors"].items()
            for n, v in zip("uv", pair)
        }
        # Also keep these learned LRs.
        factors["edit_lrs"] = self.alg.edit_lrs.detach().cpu().numpy()

        # Edit!
        d = factors
        torch_factors = {k: torch.tensor(v) for k, v in d.items()}
        eli = 0
        edit_lrs = torch_factors["edit_lrs"]

        with torch.no_grad():
            for n, p in model.named_parameters():
                uname, vname = f"{n}.u", f"{n}.v"
                if uname in torch_factors:
                    if return_orig_weights and n not in weights_copy:
                        weights_copy[n] = p.detach().clone()

                    if "gpt2" in hparams.model_name.lower():
                        delta = torch_factors[uname].t() @ torch_factors[vname]
                    elif "gpt-j" in hparams.model_name.lower():
                        delta = torch_factors[vname].t() @ torch_factors[uname]
                    elif "llama" in hparams.model_name.lower():
                        delta = torch_factors[vname].t() @ torch_factors[uname]
                    elif 'baichuan' in hparams.model_name.lower():
                        delta = torch_factors[vname].t() @ torch_factors[uname]
                    elif 't5' in hparams.model_name.lower():
                        delta = torch_factors[vname].t() @ torch_factors[uname]
                    elif 'chatglm2' in hparams.model_name.lower():
                        delta = torch_factors[vname].t() @ torch_factors[uname]
                    elif 'internlm' in hparams.model_name.lower():
                        delta = torch_factors[vname].t() @ torch_factors[uname]
                    elif 'qwen' in hparams.model_name.lower():
                        delta = torch_factors[vname].t() @ torch_factors[uname]
                    else:
                        raise ValueError("Unknown model")
                    p.add_((delta * edit_lrs[eli] * hparams.lr_scale).to(p.device))
                    eli += 1

        return model, weights_copy