import os, re
import sys
import torch, numpy
import tqdm
import importlib, copy
import transformers
from collections import defaultdict

from ..util import nethook
from matplotlib import pyplot as plt
from .main import (
    ModelAndTokenizer,
    make_inputs,
    predict_from_input,
    decode_tokens,
    layername,
    find_token_range,
    trace_with_patch,
    plot_trace_heatmap,
    collect_embedding_std,
)
from ..util.globals import DATA_DIR
from .knowns import KnownsDataset
from name_dict import MODEL_NAME_DICT


def trace_with_repatch(
    model,  # The model
    inp,  # A set of inputs
    states_to_patch,  # A list of (token index, layername) triples to restore
    states_to_unpatch,  # A list of (token index, layername) triples to re-randomize
    answers_t,  # Answer probabilities to collect
    tokens_to_mix,  # Range of tokens to corrupt (begin, end)
    noise=0.1,  # Level of noise to add
):
    prng = numpy.random.RandomState(1)  # pseudorandom noise generator
    patch_spec = defaultdict(list)
    for t, l in states_to_patch:
        patch_spec[l].append(t)  # k: layer, v: token_position
    unpatch_spec = defaultdict(list)
    for t, l in states_to_unpatch:
        unpatch_spec[l].append(t)

    def untuple(x):
        return x[0] if isinstance(x, tuple) else x

    if "gpt" in model.config._name_or_path.lower():
        word_embedding_key = "transformer.wte"
    elif "llama" in model.config._name_or_path.lower():
        word_embedding_key = "model.embed_tokens"
    elif 'qwen' in model.config._name_or_path.lower():
        word_embedding_key = "model.embed_tokens"
    else:
        raise NotImplementedError

    # Define the model-patching rule.
    def patch_rep(x, layer):
        if layer == word_embedding_key:
            # If requested, we corrupt a range of token embeddings on batch items x[1:]
            if tokens_to_mix is not None:
                b, e = tokens_to_mix
                x[1:, b:e] += noise * torch.from_numpy(
                    prng.randn(x.shape[0] - 1, e - b, x.shape[2])
                ).to(x.device)
            return x
        if first_pass or (layer not in patch_spec and layer not in unpatch_spec):
            return x
        # If this layer is in the patch_spec, restore the uncorrupted hidden state
        # for selected tokens.
        h = untuple(x)
        for t in patch_spec.get(layer, []):
            h[1:, t] = h[0, t]
        for t in unpatch_spec.get(layer, []):
            h[1:, t] = untuple(first_pass_trace[layer].output)[1:, t]
        return x

    # With the patching rules defined, run the patched model in inference.
    for first_pass in [True, False] if states_to_unpatch else [False]:
        with torch.no_grad(), nethook.TraceDict(
            model,
            [word_embedding_key] + list(patch_spec.keys()) + list(unpatch_spec.keys()),
            edit_output=patch_rep,
        ) as td:
            outputs_exp = model(**inp)
            if first_pass:
                first_pass_trace = td

    # We report softmax probabilities for the answers_t token predictions of interest.
    probs = torch.softmax(outputs_exp.logits[1:, -1, :], dim=1).mean(dim=0)[answers_t]

    return probs


def calculate_hidden_flow_3(
    mt,
    prompt,
    subject,
    token_range=None,
    samples=10,
    noise=0.1,
    window=10,
    extra_token=0,
    disable_mlp=False,
    disable_attn=False,
):
    """
    Runs causal tracing over every token/layer combination in the network
    and returns a dictionary numerically summarizing the results.
    """
    inp = make_inputs(mt.tokenizer, [prompt] * (samples + 1))
    with torch.no_grad():
        answer_t, base_score = [d[0] for d in predict_from_input(mt.model, inp)]
    [answer] = decode_tokens(mt.tokenizer, [answer_t])
    e_range = find_token_range(mt.tokenizer, inp["input_ids"][0], subject)
    if token_range == "last_subject":
        token_range = [e_range[1] - 1]
    e_range = (e_range[0], e_range[1] + extra_token)
    low_score = trace_with_patch(
        mt.model, inp, [], answer_t, e_range, noise=noise
    ).item()

    differences = trace_important_states_3(
        mt.model,
        mt.num_layers,
        inp,
        e_range,
        answer_t,
        noise=noise,
        disable_mlp=disable_mlp,
        disable_attn=disable_attn,
        token_range=token_range,
    )
    differences = differences.detach().cpu()
    return dict(
        scores=differences,
        low_score=low_score,
        high_score=base_score,
        input_ids=inp["input_ids"][0],
        input_tokens=decode_tokens(mt.tokenizer, inp["input_ids"][0]),
        subject_range=e_range,
        answer=answer,
        window=window,
        kind="",
    )


def trace_important_states_3(
    model,
    num_layers,
    inp,
    e_range,
    answer_t,
    noise=0.1,
    disable_mlp=False,
    disable_attn=False,
    token_range=None,
):
    ntoks = inp["input_ids"].shape[1]
    table = []
    zero_mlps = []
    if token_range is None:
        token_range = range(ntoks)
    for tnum in token_range:
        zero_mlps = []
        if disable_mlp:
            zero_mlps = [
                (tnum, layername(model, L, "mlp")) for L in range(0, num_layers)
            ]
        if disable_attn:
            zero_mlps += [
                (tnum, layername(model, L, "attn")) for L in range(0, num_layers)
            ]
        row = []
        for layer in range(0, num_layers):
            r = trace_with_repatch(
                model,
                inp,
                [(tnum, layername(model, layer))],
                zero_mlps,  # states_to_unpatch
                answer_t,
                tokens_to_mix=e_range,
                noise=noise,
            )
            row.append(r)
        table.append(torch.stack(row))
    return torch.stack(table)


def calculate_last_subject(
    mt, prefix, entity, noise_level, cache=None, token_range="last_subject"
):
    def load_from_cache(filename):
        try:
            dat = numpy.load(f"{cache}/{filename}")
            return {
                k: (
                    v
                    if not isinstance(v, numpy.ndarray)
                    else str(v) if v.dtype.type is numpy.str_ else torch.from_numpy(v)
                )
                for k, v in dat.items()
            }
        except FileNotFoundError as e:
            return None

    no_attn_r = load_from_cache("no_attn_r.npz")
    uncached_no_attn_r = no_attn_r is None
    no_mlp_r = load_from_cache("no_mlp_r.npz")
    uncached_no_mlp_r = no_mlp_r is None
    ordinary_r = load_from_cache("ordinary.npz")
    uncached_ordinary_r = ordinary_r is None
    if uncached_no_attn_r:
        no_attn_r = calculate_hidden_flow_3(
            mt,
            prefix,
            entity,
            disable_attn=True,
            token_range=token_range,
            noise=noise_level,
        )
    if uncached_no_mlp_r:
        no_mlp_r = calculate_hidden_flow_3(
            mt,
            prefix,
            entity,
            disable_mlp=True,
            token_range=token_range,
            noise=noise_level,
        )
    if uncached_ordinary_r:
        ordinary_r = calculate_hidden_flow_3(
            mt, prefix, entity, token_range=token_range, noise=noise_level
        )
    if cache is not None:
        os.makedirs(cache, exist_ok=True)
        for u, r, filename in [
            (uncached_no_attn_r, no_attn_r, "no_attn_r.npz"),
            (uncached_no_mlp_r, no_mlp_r, "no_mlp_r.npz"),
            (uncached_ordinary_r, ordinary_r, "ordinary.npz"),
        ]:
            if u:
                numpy.savez(
                    f"{cache}/{filename}",
                    **{
                        k: v.cpu().numpy() if torch.is_tensor(v) else v
                        for k, v in r.items()
                    },
                )
    if False:
        return (ordinary_r["scores"][0], no_attn_r["scores"][0], no_mlp_r["scores"][0])
    return (
        ordinary_r["scores"][0] - ordinary_r["low_score"],
        no_attn_r["scores"][0] - ordinary_r["low_score"],
        no_mlp_r["scores"][0] - ordinary_r["low_score"],
    )


def plot_last_subject(mt, prefix, entity, token_range="last_subject", savepdf=None):
    ordinary, no_attn, no_mlp = calculate_last_subject(
        mt, prefix, entity, token_range=token_range
    )
    plot_comparison(ordinary, no_attn, no_mlp, prefix, savepdf=savepdf)


def plot_comparison(ordinary, no_attn, no_mlp, title, savepdf=None):
    with plt.rc_context(rc={"font.family": "Times New Roman"}):
        import matplotlib.ticker as mtick

        fig, ax = plt.subplots(1, figsize=(6, 1.5), dpi=300)
        ax.bar(
            [i - 0.3 for i in range(len(ordinary))],
            ordinary,
            width=0.3,
            color="#7261ab",
            label="Impact of single state on P",
        )
        ax.bar(
            [i for i in range(len(no_attn))],
            no_attn,
            width=0.3,
            color="#f3201b",
            label="Impact with Attn severed",
        )
        ax.bar(
            [i + 0.3 for i in range(len(no_mlp))],
            no_mlp,
            width=0.3,
            color="#20b020",
            label="Impact with MLP severed",
        )
        ax.set_title(
            title
        )  #'Impact of individual hidden state at last subject token with MLP disabled')
        ax.set_ylabel("Indirect Effect")
        # ax.set_xlabel('Layer at which the single hidden state is restored')
        ax.yaxis.set_major_formatter(mtick.PercentFormatter(1.0))
        ax.set_ylim(None, max(0.025, ordinary.max() * 1.05))
        ax.legend()
        if savepdf:
            os.makedirs(os.path.dirname(savepdf), exist_ok=True)
            plt.savefig(savepdf, bbox_inches="tight")
            plt.close()
        else:
            plt.show()


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--model",
        type=str,
        choices=["gpt2-xl", "llama3-8b-it", "qwen2.5-3b-it", "qwen2.5-14b-it", "qwen2.5-7b-it"],
        default="llama3-8b-it",
    )
    args = parser.parse_args()

    model_name = MODEL_NAME_DICT[args.model]
    mt = ModelAndTokenizer(
        model_name,
        low_cpu_mem_usage=False,
        torch_dtype=(torch.float16 if "20b" in model_name else None),
    )
    knowns = KnownsDataset(
        f"./dataset/knowns/{args.model}.json"
    )  # Dataset of known facts
    noise_level = 3 * collect_embedding_std(mt, [k["subject"] for k in knowns])
    print(f"Using noise level {noise_level}")

    all_ordinary = []
    all_no_attn = []
    all_no_mlp = []

    for i, knowledge in enumerate(tqdm.tqdm(knowns)):
        ordinary, no_attn, no_mlp = calculate_last_subject(
            mt,
            knowledge["prompt"],
            knowledge["subject"],
            noise_level=noise_level,
            cache=f"{DATA_DIR}/causal_trace_results/{args.model}/case_{i}",
        )
        all_ordinary.append(ordinary)
        all_no_attn.append(no_attn)
        all_no_mlp.append(no_mlp)

    title = "Causal effect of states at the early site with Attn or MLP modules severd"

    avg_ordinary = torch.stack(all_ordinary).mean(dim=0)
    avg_no_attn = torch.stack(all_no_attn).mean(dim=0)
    avg_no_mlp = torch.stack(all_no_mlp).mean(dim=0)

    import matplotlib.ticker as mtick

    with plt.rc_context(rc={"font.family": "Times New Roman"}):
        fig, ax = plt.subplots(1, figsize=(10, 3.5), dpi=300)
        ax.bar(
            [i - 0.3 for i in range(len(avg_ordinary))],
            avg_ordinary,
            width=0.3,
            color="#7261ab",
            label="Effect of single state on P",
        )
        ax.bar(
            [i for i in range(len(avg_no_attn))],
            avg_no_attn,
            width=0.3,
            color="#f3201b",
            label="Effect with Attn severed",
        )
        ax.bar(
            [i + 0.3 for i in range(len(avg_no_mlp))],
            avg_no_mlp,
            width=0.3,
            color="#20b020",
            label="Effect with MLP severed",
        )
        ax.set_title(
            title
        )  #'Impact of individual hidden state at last subject token with MLP disabled')
        ax.set_ylabel("Average Indirect Effect")
        ax.set_xlabel("Layer at which the single hidden state is restored")
        ax.yaxis.set_major_formatter(mtick.PercentFormatter(1.0))
        # ax.set_ylim(None, max(0.025, 0.125))
        ax.legend(frameon=False)
    fig.savefig(
        f"./figures/causal_trace/no-attn-mlp-{args.model}.pdf", bbox_inches="tight"
    )
    print([d[20] - d[10] for d in [avg_ordinary, avg_no_attn, avg_no_mlp]])
    print(avg_ordinary[15], avg_no_attn[15], avg_no_mlp[15])
    
    print(f'avg_ordinary.shape: {avg_ordinary.shape}')
    print(f'avg_no_mlp.shape: {avg_no_mlp.shape}')
    abl_mlp = avg_ordinary - avg_no_mlp
    print(abl_mlp)
    print("MLP severed topk:", abl_mlp.topk(k=10).indices)
    print("Single state topk:", avg_ordinary.topk(k=10).indices)
