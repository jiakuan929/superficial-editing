import argparse
from datetime import datetime
import os
import sys
import json
import random
from typing import Literal, List

from matplotlib import pyplot as plt
import seaborn as sns
import tqdm
import numpy as np
from omegaconf import OmegaConf

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

sys.path.append(".")
from baselines.util import nethook
from utils import restore_model, init_model_tokenizer
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
    parser.add_argument("--input", type=str, choices=["edit", "attack"], default="attack")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--expname", type=str, default="")
    args = parser.parse_args()
    return args


def seed_everything(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True


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

# def get_attn_input_output(
#     model: AutoModelForCausalLM,
#     tok: AutoTokenizer,
#     prompt: str,
#     module_str: str,
#     return_last_pos: bool = False,
# ):
#     module_output = []
#     module_input = []

#     def _get_module_output(mod, mod_in, mod_out):
#         if isinstance(mod_in, tuple):
#             module_input.append(mod_in[0])
#         elif isinstance(mod_in, torch.Tensor):
#             module_input.append(mod_in)
#         else:
#             raise TypeError

#         if isinstance(mod_out, torch.Tensor):
#             module_output.append(mod_out)
#         elif isinstance(mod_out, tuple):
#             print(type(mod_out[0]), type(mod_out[1]), mod_out[1].shape, mod_out[2])
#             module_output.append(mod_out[0])
#         else:
#             raise TypeError

#     module = nethook.get_module(model, module_str)
#     hook = module.register_forward_hook(_get_module_output)
#     # Forward the prompt
#     encoding = tok(prompt, return_tensors="pt").to("cuda")
#     with torch.no_grad():
#         model(**encoding)

#     module_output = module_output[0]
#     module_input = module_input[0]
#     hook.remove()
#     if return_last_pos:
#         module_output = module_output[:, -1, :]
#         module_input = module_input[:, -1, :]
#     return module_input, module_output


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

    real_lens = encoding["attention_mask"].sum(1).cpu().numpy().tolist()

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


def get_attn_head_output(
    model: AutoModelForCausalLM,
    tok: AutoTokenizer,
    prompts: list,
    module_str: str,
    return_last_pos: bool = False,
) -> List[torch.Tensor]:
    
    head_outputs = []
    head_dim = model.config.hidden_size // model.config.num_attention_heads

    def _get_module_output(mod, mod_in, mod_out):
        wo = mod.weight.clone().detach()
        wo_heads = torch.split(wo, head_dim, dim=1)

        if isinstance(mod_in, tuple):
            # print(len(mod_in), type(mod_in[0]), mod_in[0].shape)
            x_in = mod_in[0].clone().detach()
        elif isinstance(mod_in, torch.Tensor):
            x_in = mod_in.clone().detach()    # [batch, seq_len, hidden_size]
        else:
            raise TypeError
        
        bsz, seq_len = x_in.shape[0], x_in.shape[1]
        x_in = x_in.reshape(bsz, seq_len, -1, head_dim).transpose(1, 2)   # [bsz, h, seq_len, head_dim]

        for head_idx, h_wo in enumerate(wo_heads):
            inp = x_in[:, head_idx, :, :]
            _output = inp @ h_wo.T    # [bsz, seq_len, hidden_size]
            head_outputs.append(_output)

    module = nethook.get_module(model, module_str)
    hook = module.register_forward_hook(_get_module_output)
    # Forward the prompt
    encoding = tok(prompts, return_tensors="pt", padding=True).to("cuda")
    with torch.no_grad():
        model(**encoding)
    
    real_lens = encoding["attention_mask"].sum(1).cpu().numpy().tolist()
    print(f"Batch = {len(prompts)} | Individual length = {real_lens}")

    
    if return_last_pos:
        new_head_outputs = []
        for h in range(model.config.num_attention_heads):
            _h_output = []
            for i in range(len(prompts)):
                _h_output.append(head_outputs[h][i, real_lens[i] - 1, :].reshape(1, -1))
            _h_output = torch.cat(_h_output, dim=0)   # [bsz, hidden_size]
            new_head_outputs.append(_h_output)
        head_outputs = new_head_outputs

    hook.remove()    
    return head_outputs


def sort_attention_heads(results, n_head: int):
    vals = np.array(results).mean(axis=0)   # [n_layer, n_head]
    vals_flatten = torch.from_numpy(vals).reshape(-1)
    total_n = len(vals_flatten)
    indices = torch.topk(vals_flatten, k=total_n).indices.reshape(-1).tolist()
    indices = [(x // n_head, x % n_head, vals[x // n_head, x % n_head]) for x in indices]
    return indices


def main(
    model_name: str,
    editor: str,
    dataset_name: str,
    input_type: str,
    exp_name: str,
):
    model, tok = init_model_tokenizer(MODEL_NAME_DICT[model_name])
    # Dataset
    datapath = ANALYSIS_DATASET_DICT[dataset_name].format(model=model_name, editor=editor)
    with open(datapath, "r", encoding="utf-8") as f:
        dataset = json.load(f)
    assert all(len(x["attack_probes_em"]) != 0 for x in dataset)

    hparams = OmegaConf.load(f"./hparams/{editor}/{model_name}.yaml")

    apply_editing = ALG_DICT[editor]

    results = []

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
        target_true_ids = tok.encode(f" {request['target_true']['str']}")

        edited_model, weights_copy = apply_editing(
            model,
            tok,
            [request],
            hparams=hparams,
            return_orig_weights=True,
        )

        record_ret = []

        attack_prompts = record["attack_probes_em"]
        edit_prompt = request['prompt'].format(request['subject'])

        inp_prompts = attack_prompts if input_type == 'attack' else [edit_prompt]
        real_lens = [len(tok.encode(x)) for x in inp_prompts]

        for layer in range(model.config.num_hidden_layers):
            attn_str = f"model.layers.{layer}.self_attn.o_proj"
            head_outputs = get_attn_head_output(
                edited_model,
                tok,
                inp_prompts,
                attn_str,
                return_last_pos=False,
            )  # n_head * [bsz, seq_len, hidden_size]

            original_head_probs = []    # n_heads * [p] 
            for head_idx, head_out in enumerate(head_outputs):
                head_probs = apply_logit_lens(edited_model, head_out, softmax=True)  # [bsz, V]
                head_probs = get_last_pos_repr(head_probs, real_lens)
                p_orig = head_probs[:, target_true_ids[0]].mean().item()
                original_head_probs.append(p_orig)
            record_ret.append(original_head_probs)

        results.append(record_ret)
        # Restore
        model = restore_model(edited_model, weights_copy)

    np_results = np.stack(results).mean(axis=0)   # [n_layer, n_head]
    
    this_time = datetime.now().strftime("%Y-%m-%d-%H-%M")
    output_dir = f"./complete_results/analysis_full_nonorm/attn_layer_head_op/{model_name}/{exp_name}_{editor}_{dataset_name}_{input_type}_{this_time}"
    os.makedirs(output_dir, exist_ok=True)

    with open(f"{output_dir}/config.json", "w", encoding="utf-8") as f:
        json.dump(
            {
                "model": model_name,
                "editor": editor,
                "dataset": dataset_name,
                "input_type": input_type,
            },
            f,
            indent=4,
        )

    with open(f"{output_dir}/numeric_results.json", "w", encoding="utf-8") as f:
        json.dump(
            results,
            f,
            indent=4,
        )
    
    sort_heads = sort_attention_heads(results, n_head=model.config.num_attention_heads)
    with open(f"{output_dir}/sorted_heads.json", "w", encoding="utf-8") as f:
        json.dump(sort_heads, f, indent=4)
    
    # Plot figures
    plt.figure()
    sns.heatmap(np_results, cmap='Purples', cbar_kws={'label': 'Original Prob.'})
    plt.legend()
    plt.xlabel('Attention heads')
    plt.ylabel('Layers')
    plt.savefig(f"{output_dir}/layer{layer}.pdf")


if __name__ == "__main__":

    args = get_arguments()

    seed_everything(args.seed)

    main(
        args.model,
        args.editor,
        args.dataset,
        args.input,
        args.expname,
    )
