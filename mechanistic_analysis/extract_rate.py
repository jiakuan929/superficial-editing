import argparse
from datetime import datetime
import math
import os
import sys
import json
import random
from typing import Literal, List, Tuple

from matplotlib import pyplot as plt
import tqdm
import numpy as np
from omegaconf import OmegaConf

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

sys.path.append(".")
from baselines.util import nethook
from utils import restore_model, init_model_tokenizer, get_main_heads
from name_dict import *
from mi_tools import apply_logit_lens


def get_arguments():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--model",
        type=str,
        choices=[
            "gpt2-xl",
            "llama3-8b-it",
            "qwen2.5-7b-it",
            "qwen2.5-14b-it",
        ],
        default="llama3-8b-it",
    )
    parser.add_argument(
        "--editor", type=str, choices=["rome", "memit", "ft", "mend"], default="rome"
    )
    parser.add_argument(
        "--dataset", type=str, choices=["counterfact", "zsre"], default="counterfact"
    )
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--locate_top", type=float, choices=[0.05, 0.1], default=0.1)
    parser.add_argument("--decode_top", type=int, default=15)
    args = parser.parse_args()
    return args


def seed_everything(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True


def get_attn_input_output(
    model: AutoModelForCausalLM,
    tok: AutoTokenizer,
    prompts: list,
    module_str: str,
    return_last_pos: bool = False,
    getter: Literal["in", "out", "io"] = "io",
):
    module_output = []
    module_input = []

    def _get_module_output(mod, mod_in, mod_out):
        if isinstance(mod_in, tuple):
            # print(len(mod_in), type(mod_in[0]), mod_in[0].shape)
            module_input.append(mod_in[0])
        elif isinstance(mod_in, torch.Tensor):
            module_input.append(mod_in)
        else:
            raise TypeError

        if isinstance(mod_out, torch.Tensor):
            module_output.append(mod_out)
        elif isinstance(mod_out, tuple):
            print(type(mod_out[0]), type(mod_out[1]), mod_out[1].shape, mod_out[2])
            module_output.append(mod_out[0])
        else:
            raise TypeError

    module = nethook.get_module(model, module_str)
    hook = module.register_forward_hook(_get_module_output)
    # Forward the prompt
    encoding = tok(prompts, return_tensors="pt", padding=True).to("cuda")
    with torch.no_grad():
        model(**encoding)

    # real_lens = encoding["attention_mask"].sum(1).cpu().numpy().tolist()
    real_lens = [len(tok.encode(x)) for x in prompts]

    module_output = module_output[0]
    module_input = module_input[0]
    hook.remove()
    if return_last_pos:
        new_module_input, new_module_output = [], []
        for i, _length in enumerate(real_lens):
            new_module_input.append(module_input[i, _length - 1, :].reshape(1, -1))
            new_module_output.append(module_output[i, _length - 1, :].reshape(1, -1))
        module_input = torch.cat(new_module_input, dim=0)
        module_output = torch.cat(new_module_output, dim=0)

    if getter == "in":
        return module_input
    if getter == "out":
        return module_output
    return module_input, module_output


def get_last_pos_repr(
    batch_tensors: torch.Tensor, real_lens: List[int]
) -> torch.Tensor:
    """
    Extract the vector of the last token position.
    batch_tensors: [bsz, seq_len, hidden_size]
    returns a tensor: [bsz, hidden_size]
    """
    new_tensors = []
    for i in range(batch_tensors.shape[0]):
        new_tensors.append(batch_tensors[i, real_lens[i] - 1, :].reshape(1, -1))
    return torch.cat(new_tensors, dim=0)


def locate_left_singular_vectors(
    model: AutoModelForCausalLM,
    tok: AutoTokenizer,
    prompts: list,
    module_str: str,
    target_head: int,
    target_true_id: int,
    locate_top: float,
    decode_top: int,
) -> Tuple[torch.Tensor, List[List[str]]]:
    """
    This function does 2 tasks:
    1. Locate some specific left singular vectors of Wo. The identified vectors are associated with original object.
    2. Use the identified vectors to compute a new output that will replaces the original attn output. Then compute topK decoded tokens based on them.

    returns tuple(located_ids, top_decoded_tokens)
    """

    # Variables to save result.
    # p_orig = []
    # p_new = []
    top_predicted_tokens = []
    identified_vector_ids = []

    head_dim = model.config.hidden_size // model.config.num_attention_heads
    real_lens = [len(tok.encode(x)) for x in prompts]

    layer_str = module_str[: module_str.rfind(".self_attn")]
    layer = nethook.get_module(model, layer_str)

    def _get_module_output(mod, mod_in, mod_out):
        wo = mod.weight.clone().detach()
        wo_heads = torch.split(wo, head_dim, dim=1)

        if isinstance(mod_in, tuple):
            # print(len(mod_in), type(mod_in[0]), mod_in[0].shape)
            x_in = mod_in[0]
        elif isinstance(mod_in, torch.Tensor):
            x_in = mod_in  # [batch, seq_len, hidden_size]
        else:
            raise TypeError

        if isinstance(mod_out, torch.Tensor):
            attn_out = mod_out
        elif isinstance(mod_out, tuple):
            # print(type(mod_out[0]), type(mod_out[1]), mod_out[1].shape, mod_out[2])
            attn_out = mod_out[0]
        else:
            raise TypeError

        bsz, seq_len = x_in.shape[0], x_in.shape[1]
        # Split input to different heads.
        x_in = x_in.reshape(bsz, seq_len, -1, head_dim).transpose(
            1, 2
        )  # [bsz, h, seq_len, head_dim]

        inp = x_in[:, target_head, :, :]  # [bsz, seq_len, head_dim]
        # Unablated head output.
        unabl_head_out = inp @ wo_heads[target_head].T  # [bsz, seq_len, hidden_size]

        unabl_logits = apply_logit_lens(model, unabl_head_out, softmax=True)
        unabl_logits = get_last_pos_repr(unabl_logits, real_lens)  # [bsz, V]
        unabl_logits = unabl_logits[:, target_true_id].reshape(-1)  # [bsz,]

        inp = get_last_pos_repr(inp, real_lens).reshape(head_dim, -1)  # [head_dim, bsz]
        h_U, h_S, h_Vh = torch.linalg.svd(
            wo_heads[target_head], full_matrices=False
        )  # [hidden_size, head_dim], [head_dim], [head_dim, head_dim]

        coef_vector = torch.diag_embed(h_S) @ h_Vh @ inp  # [head_dim, bsz]
        I = (
            torch.eye(n=coef_vector.shape[0], device=coef_vector.device)
            .unsqueeze(0)
            .repeat(coef_vector.shape[1], 1, 1)
        )  # Identity matrix [bsz, head_dim, head_dim]
        coef_vector = coef_vector.reshape(bsz, I.shape[1], -1)  # [bsz, head_dim, 1]
        abl_coef = coef_vector * (1 - I)  # [bsz, head_dim, head_dim]

        abl_head_out = (h_U @ abl_coef).transpose(1, 2)  # [bsz, head_dim, hidden_size]

        abl_logit = apply_logit_lens(model, abl_head_out, softmax=True)[
            :, :, target_true_id
        ].reshape(
            bsz, -1
        )  # [bsz, head_dim]

        logit_diff = unabl_logits.mean() - abl_logit.mean(dim=0).reshape(
            -1
        )  # [head_dim,]

        # lower_bound = logit_diff.min() + (logit_diff.max() - logit_diff.min()) * 0.7
        # count = torch.gt(logit_diff, lower_bound).sum().item()
        count = math.floor(head_dim * locate_top)
        print(f"Computed dynamic count: {count}")

        selected_indices = torch.topk(logit_diff, k=count).indices
        identified_vector_ids.append(selected_indices)
        print(f"selected: {selected_indices}")

        coef_vector = coef_vector.reshape(bsz, -1)  # [bsz, head_dim]
        latent_probs: torch.Tensor = get_subcomponent_product(
            model,
            h_U,
            coef_vector,
            selected_indices,
            logit_lens=True,
            softmax=True,
        )  # [bsz, seq_len, V]
        assert latent_probs.shape[1] == 1
        latent_probs = latent_probs[:, 0, :]

        top_token_ids = torch.topk(
            latent_probs,
            k=decode_top,
        ).indices
        _top_tokens = [[tok.decode(x) for x in ttids] for ttids in top_token_ids]
        top_predicted_tokens.append(_top_tokens)
        print("Attack decoded top tokens:")
        print(_top_tokens)

    module = nethook.get_module(model, module_str)
    hook = module.register_forward_hook(_get_module_output)
    # Forward the prompt
    encoding = tok(prompts, return_tensors="pt", padding=True).to("cuda")
    with torch.no_grad():
        model(**encoding)

    identified_vector_ids = identified_vector_ids[0]
    # p_orig = p_orig[0]
    # p_new = p_new[0]
    top_predicted_tokens = top_predicted_tokens[0]

    hook.remove()
    return identified_vector_ids, top_predicted_tokens


def attn_head_svd_using_edit(
    model: AutoModelForCausalLM,
    tok: AutoTokenizer,
    prompts: list,
    module_str: str,
    target_head: int,
    top_indices: torch.Tensor,
    original_next_id: int,
    new_next_id: int,
    decode_top: int,
) -> List[List[str]]:

    # p_origs = []
    # p_news = []
    top_tokens = []
    head_dim = model.config.hidden_size // model.config.num_attention_heads
    real_lens = [len(tok.encode(x)) for x in prompts]

    layer_str = module_str[: module_str.rfind(".self_attn")]
    layer = nethook.get_module(model, layer_str)

    def _get_module_output(mod, mod_in, mod_out):
        wo = mod.weight.clone().detach()
        wo_heads = torch.split(wo, head_dim, dim=1)

        if isinstance(mod_in, tuple):
            # print(len(mod_in), type(mod_in[0]), mod_in[0].shape)
            x_in = mod_in[0]
        elif isinstance(mod_in, torch.Tensor):
            x_in = mod_in  # [batch, seq_len, hidden_size]
        else:
            raise TypeError

        if isinstance(mod_out, torch.Tensor):
            attn_out = mod_out
        elif isinstance(mod_out, tuple):
            # print(type(mod_out[0]), type(mod_out[1]), mod_out[1].shape, mod_out[2])
            attn_out = mod_out[0]
        else:
            raise TypeError

        bsz, seq_len = x_in.shape[0], x_in.shape[1]
        x_in = x_in.reshape(bsz, seq_len, -1, head_dim).transpose(
            1, 2
        )  # [bsz, h, seq_len, head_dim]

        inp = x_in[:, target_head, :, :]  # [bsz, seq_len, head_dim]
        inp = get_last_pos_repr(inp, real_lens).reshape(head_dim, -1)  # [head_dim, bsz]
        h_U, h_S, h_Vh = torch.linalg.svd(
            wo_heads[target_head], full_matrices=False
        )  # [hidden_size, head_dim], [head_dim], [head_dim, head_dim]

        coef_vector = torch.diag_embed(h_S) @ h_Vh @ inp  # [head_dim, bsz]
        coef_vector = coef_vector.reshape(bsz, -1)
        latent_probs: torch.Tensor = get_subcomponent_product(
            model,
            h_U,
            coef_vector,
            top_indices,
            logit_lens=True,
            softmax=True,
        )  # [bsz, 1, V]
        assert latent_probs.shape[1] == 1
        latent_probs = latent_probs[:, 0, :]

        # p_orig = latent_probs[:, original_next_id].mean().item()
        # p_new = latent_probs[:, new_next_id].mean().item()
        # p_origs.append(p_orig)
        # p_news.append(p_new)

        top_token_ids = torch.topk(
            latent_probs,
            k=decode_top,
        ).indices
        tmp = [[tok.decode(x) for x in ttids] for ttids in top_token_ids]
        top_tokens.append(tmp)
        print("Decoded top tokens with edit_prompt as input:")
        print(tmp)

    module = nethook.get_module(model, module_str)
    hook = module.register_forward_hook(_get_module_output)
    # Forward the prompt
    encoding = tok(prompts, return_tensors="pt", padding=True).to("cuda")
    with torch.no_grad():
        model(**encoding)

    # p_origs = p_origs[0]
    # p_news = p_news[0]
    top_tokens = top_tokens[0]
    hook.remove()
    return top_tokens


def get_subcomponent_product(
    model: AutoModelForCausalLM,
    left_singular_matrix: torch.Tensor,
    lambda_vector: torch.Tensor,
    indices: torch.Tensor,
    logit_lens: bool = False,
    softmax: bool = False,
) -> torch.Tensor:
    # lambda_vector = lambda_vector.reshape(-1)
    bsz, head_dim = lambda_vector.shape
    sub_lambda_v = lambda_vector[:, indices].reshape(bsz, -1)  # [bsz, cnt]
    sub_singular_m = left_singular_matrix[:, indices]  # Sub matrix [hidden_size, cnt]
    sub_output = sub_singular_m @ sub_lambda_v.T  # [hidden_size, bsz]

    if logit_lens:
        sub_output = sub_output.reshape(bsz, 1, -1)
        sub_output = apply_logit_lens(model, sub_output, softmax=softmax)  # [bsz, 1, V]
    return sub_output


def main(
    model_name: str,
    editor: str,
    dataset_name: str,
    locate_top: float,
    decode_top: int,
):
    model, tok = init_model_tokenizer(MODEL_NAME_DICT[model_name])
    # Dataset
    datapath = ANALYSIS_DATASET_DICT[dataset_name].format(
        model=model_name, editor=editor
    )
    with open(datapath, "r", encoding="utf-8") as f:
        dataset = json.load(f)
    assert all(len(x["attack_probes_em"]) != 0 for x in dataset)

    hparams = OmegaConf.load(f"./hparams/{editor}/{model_name}.yaml")

    # if model_name == "llama3-8b-it":
    #     # Use attack as input: {24: [3], 27: [20], 31: [7]}
    #     # Use edit_prompt as input: {24: [3], }
    #     # main_layers = [24, 27, 30, 31]
    #     # main_heads = {
    #     #     # 23: [27],
    #     #     24: [3],
    #     #     27: [20],
    #     #     30: [29],
    #     #     31: [6, 7],
    #     # }
    #     main_layers = [23, 24, 27, 30, 31]
    #     main_heads = {
    #         23: [27],
    #         24: [3],
    #         27: [20],
    #         30: [29],
    #         31: [6, 7],
    #     }
    # elif model_name == "qwen2.5-7b-it":
    #     # main_layers = [23, 27]
    #     # main_heads = {23: [11], 27: [3, 15]}
    #     main_layers = [23, 24, 26, 27]
    #     main_heads = {
    #         23: [4, 6, 11],
    #         24: [27],
    #         26: [0],
    #         27: [2, 3, 15]
    #     }
    # elif model_name == "qwen2.5-14b-it":
    #     # main_layers = [36, 40, 41, 42, 43, 45, 46]
    #     # main_heads = {
    #     #     36: [10, 14, 30],
    #     #     40: [22, 23],
    #     #     41: [14],
    #     #     42: [21],
    #     #     43: [36],
    #     #     45: [27, 37],
    #     #     46: [4, 28]
    #     # }
    #     main_layers = [36, 40, 41, 42, 43, 45, 46]
    #     main_heads = {
    #         36: [10],
    #         40: [22, 23],
    #         41: [14],
    #         42: [21],
    #         43: [36],
    #         45: [27, 37],
    #         46: [4, 28],
    #     }
    # else:
    #     raise NotImplementedError
    main_layers, main_heads = get_main_heads(
        path=f"./hparams/nonorm_attn_heads/{model_name}.json", editor=editor
    )
    # main_layers, main_heads = get_key_head_config(model_name, editor, threshold=0.2)
    print(f"main_layers: {main_layers}")
    print(f"main_heads: {main_heads}")

    apply_editing = ALG_DICT[editor]

    results = []
    attack_results = []

    for record in tqdm.tqdm(dataset):
        if "requested_rewrite" in record:
            # For CounterFact
            request = record["requested_rewrite"]
        else:
            # For ZsRE
            request = {
                "prompt": record["src"].replace(record["subject"], "{}"),
                "subject": record["subject"],
                "target_new": {"str": record["alt"]},
                "target_true": {"str": record["answers"][0]},
            }
        target_true_str = request["target_true"]["str"]
        target_new_str = request["target_new"]["str"]
        target_new_ids = tok.encode(f" {request['target_new']['str']}")
        target_true_ids = tok.encode(f" {request['target_true']['str']}")
        edit_prompt = request["prompt"].format(request["subject"])

        edited_model, weights_copy = apply_editing(
            model,
            tok,
            [request],
            hparams=hparams,
            return_orig_weights=True,
        )

        record_ret = {"target_true": target_true_str, "target_new": target_new_str}
        record_ret_a = {"target_true": target_true_str, "target_new": target_new_str}

        inp_prompts = record["attack_probes_em"]
        for layer in main_layers:
            attn_str = f"model.layers.{layer}.self_attn.o_proj"
            for target_head in main_heads[layer]:
                print("---" * 20 + f" Layer {layer} Head {target_head} " + "---" * 20)

                top_indices, top_tokens_a = locate_left_singular_vectors(
                    edited_model,
                    tok,
                    inp_prompts,
                    attn_str,
                    target_head=target_head,
                    target_true_id=target_true_ids[0],
                    locate_top=locate_top,
                    decode_top=decode_top,
                )
                normed_top_tokens_a = [
                    [s.strip() for s in group] for group in top_tokens_a
                ]
                print(f"normed attack: {normed_top_tokens_a}")
                record_ret_a[f"L{layer}_H{target_head}"] = [
                    [target_true_str in group for group in normed_top_tokens_a],
                    [target_new_str in group for group in normed_top_tokens_a],
                ]
                record_ret[f"L{layer}_H{target_head}_vector_ids"] = (
                    top_indices.cpu().tolist()
                )

                print(f"Localized left singular vectors: {top_indices.cpu().tolist()}")

                top_tokens = attn_head_svd_using_edit(
                    edited_model,
                    tok,
                    [edit_prompt],
                    attn_str,
                    target_head=target_head,
                    top_indices=top_indices,
                    original_next_id=target_true_ids[0],
                    new_next_id=target_new_ids[0],
                    decode_top=decode_top,
                )
                normed_top_tokens = [[s.strip() for s in group] for group in top_tokens]
                print(f"normed edit: {normed_top_tokens}")
                record_ret[f"L{layer}_H{target_head}"] = [
                    [target_true_str in group for group in normed_top_tokens],
                    [target_new_str in group for group in normed_top_tokens],
                ]
                record_ret[f"L{layer}_H{target_head}_vector_ids"] = (
                    top_indices.cpu().tolist()
                )

        results.append(record_ret)
        attack_results.append(record_ret_a)
        # Restore
        model = restore_model(edited_model, weights_copy)

    this_time = datetime.now().strftime("%Y-%m-%d-%H-%M")
    output_dir = f"./complete_results/analysis_full_nonorm/extraction_rate/{model_name}/{editor}_u{locate_top}_dec{decode_top}_{dataset_name}_{this_time}"
    os.makedirs(output_dir, exist_ok=True)

    with open(f"{output_dir}/config.json", "w", encoding="utf-8") as f:
        json.dump(
            {
                "model": model_name,
                "editor": editor,
                "dataset": dataset_name,
                "main_layers": main_layers,
                "main_heads": main_heads,
                "locate_top": locate_top,
                "decode_top": decode_top,
            },
            f,
            indent=4,
        )

    with open(
        f"{output_dir}/numeric_results_for_attack.json", "w", encoding="utf-8"
    ) as f:
        json.dump(
            attack_results,
            f,
            indent=4,
        )

    with open(
        f"{output_dir}/numeric_results_for_edit_prompt.json", "w", encoding="utf-8"
    ) as f:
        json.dump(
            results,
            f,
            indent=4,
        )

    summarized_res = {}
    for idx, setting in enumerate(["orig", "new"]):
        for layer in main_layers:
            for head in main_heads[layer]:
                summarized_res[f"L{layer}_H{head}_{setting}"] = [
                    x[f"L{layer}_H{head}"][idx] for x in results
                ]

    for k, v in summarized_res.items():
        summarized_res[k] = [item for x in v for item in x]
        summarized_res[k] = np.mean(summarized_res[k])
    print(f"Summarized result of edit_prompt:\n{summarized_res}")

    summarized_res_a = {}
    for idx, setting in enumerate(["orig", "new"]):
        for layer in main_layers:
            for head in main_heads[layer]:
                summarized_res_a[f"L{layer}_H{head}_{setting}"] = [
                    x[f"L{layer}_H{head}"][idx] for x in attack_results
                ]
    for k, v in summarized_res_a.items():
        summarized_res_a[k] = [item for x in v for item in x]
        summarized_res_a[k] = np.mean(summarized_res_a[k])
    print(f"Summarized result of attack_probes:\n{summarized_res_a}")

    with open(f"{output_dir}/overall_edit_prompt.json", "w", encoding="utf-8") as f:
        json.dump(summarized_res, f, indent=4)
    with open(f"{output_dir}/overall_attack.json", "w", encoding="utf-8") as f:
        json.dump(summarized_res_a, f, indent=4)


if __name__ == "__main__":

    args = get_arguments()

    seed_everything(args.seed)

    main(
        args.model,
        args.editor,
        args.dataset,
        args.locate_top,
        args.decode_top,
    )
