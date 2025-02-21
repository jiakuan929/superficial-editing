import json
import math
import os
from typing import Literal, Tuple, List, Dict, Union

import torch
import numpy as np
from transformers import AutoModelForCausalLM, AutoTokenizer

from baselines.util import nethook


def init_model_tokenizer(
    model_name: str,
    tok_pad_side: Literal["left", "right"] = "right",
    add_bos_token: bool = False,
) -> Tuple[AutoModelForCausalLM, AutoTokenizer]:
    # Customization for OpenPAI platform.
    if not os.path.exists(model_name):
        model_name = model_name.replace("publiccache", "usercache")
    print(f"Loading model: {model_name}")
    model = AutoModelForCausalLM.from_pretrained(model_name, device_map="auto")
    model.eval()
    tok = AutoTokenizer.from_pretrained(model_name)
    tok.padding_side = tok_pad_side
    tok.add_bos_token = add_bos_token
    tok.pad_token = tok.eos_token
    return model, tok


def restore_model(model: AutoModelForCausalLM, weights_copy: Dict):
    with torch.no_grad():
        for k, v in weights_copy.items():
            w = nethook.get_parameter(model, k)
            w[...] = v.to(w.device)
    print("Model restored.")
    return model


def get_main_heads(path: str, editor: str):
    """Reads main_layers and main_heads in the specified file path.

    Returns:
        Tuple[main_layers, main_heads]
    """
    with open(path, 'r', encoding='utf-8') as f:
        data = json.load(f)
    main_layers = data[editor]["main_layers"]
    main_heads = data[editor]["main_heads"]
    main_layers: List[int] = [int(x) for x in main_layers]
    main_heads = {
        int(k): [int(x) for x in v]
        for k, v in main_heads.items()
    }
    return main_layers, main_heads


def generate_target_tokens_argmax(
    model: AutoModelForCausalLM, tok: AutoTokenizer, prefix: str, n_steps: int = 1
):
    generated_tokens = {
        "id": [],
        "str": "",
    }
    for i in range(n_steps):
        input_toks = tok(prefix, return_tensors="pt").to("cuda")
        with torch.no_grad():
            logits = model(**input_toks).logits
        probs = torch.softmax(logits[:, -1, :], dim=-1)
        next_id = probs.argmax(dim=-1).item()
        next_str = tok.decode(next_id)
        prefix += next_str
        generated_tokens["id"].append(next_id)
        generated_tokens["str"] += next_str
    return generated_tokens


def get_last_token_probability_distribution(
    model: AutoModelForCausalLM,
    tok: AutoTokenizer,
    prefix: str,
) -> torch.Tensor:
    input_toks = tok(prefix, return_tensors="pt").to("cuda")
    with torch.no_grad():
        logits = model(**input_toks).logits
    probs = torch.softmax(logits[:, -1, :], dim=-1)  # [1, vocab_size]
    return probs


def get_target_probability(
    model: AutoModelForCausalLM, tok: AutoTokenizer, prefix: str, target: str
):
    if target[0] != " ":
        target = " " + target
    target_ids = tok.encode(target)
    target_prob = []
    for i, tgt_id in enumerate(target_ids):
        input_toks = tok(prefix, return_tensors="pt").to("cuda")
        with torch.no_grad():
            logits = model(**input_toks).logits
        probs = torch.softmax(logits[:, -1, :], dim=-1)
        target_prob.append(probs[0, tgt_id].item())
        completion_str = tok.decode(tgt_id)
        prefix += completion_str
    return math.prod(target_prob)


def get_target_probability_2(model, tok, prefixes: Union[List[str], str], targets: Union[List[str], str]):
    if isinstance(prefixes, str):
        prefixes = [prefixes]
    if isinstance(targets, str):
        targets = [targets]

    assert len(prefixes) == len(targets)
    new_targets = []
    for tgt in targets:
        new_targets.append(tgt if tgt[0] == " " else " " + tgt)
    targets = new_targets
    target_ids = [tok(tgt, return_tensors="pt")["input_ids"][0] for tgt in targets]

    prefix_lens = [len(tok.encode(x)) for x in prefixes]
    sentence_toks = tok(
        [p + t for p, t in zip(prefixes, targets)], return_tensors="pt", padding=True
    ).to("cuda")
    with torch.no_grad():
        logits = model(**sentence_toks).logits
    probs = torch.softmax(logits, dim=-1)
    ret = []
    for i in range(len(prefixes)):
        ps = []
        for j, cur_tok in enumerate(target_ids[i]):
            ps.append(probs[i, prefix_lens[i] - 1 + j, cur_tok].item())
        p = np.prod(ps).item()
        ret.append(p)
    return ret


def get_mlp_output_at_layer(model: AutoModelForCausalLM, tok: AutoTokenizer, prefix: str, layer_idx: int):
    mlp_outputs = []
    def _mlp_hook(mod, mod_in, mod_out):
        # print(type(mod_out), len(mod_out))
        mlp_outputs.append(mod_out[:, -1, :])
    mlp_module_str = f"model.layers.{layer_idx}.mlp.down_proj"
    mlp_module = nethook.get_module(model, mlp_module_str)
    hook = mlp_module.register_forward_hook(_mlp_hook)

    input_toks = tok(prefix, return_tensors="pt").to("cuda")
    with torch.no_grad():
        model(**input_toks)
    
    hook.remove()
    return mlp_outputs[0]


def calculate_superficial_score(p_edit: float, p_paras: List[float]) -> float:
    return 2 * p_edit / (p_edit + np.mean(p_paras)) - 1
