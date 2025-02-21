import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from baselines.util import nethook


def get_repr_at_layer(
    model: AutoModelForCausalLM,
    tok: AutoTokenizer,
    prompt: str,
    layer: int = -1,
    return_last_pos: bool = True,
):
    """Get layer output given `prompt` as input.

    Args:
        model (AutoModelForCausalLM): Language model
        tok (AutoTokenizer): tokenizer
        prompt (str): input prompt
        layer (int, optional): target layer. Defaults to -1.
        return_last_pos (bool, optional): Whether return the vector at last position or not. Defaults to True.

    Returns:
        Tensor: specific layer output
    """
    layer_output = list()

    # Define hook operations at target layer.
    def _get_layer_output(layer, layer_in, layer_out):
        # Llama3-8b-it layer_out: Tuple length: 2
        layer_output.append(layer_out[0])

    layer_module = model.model.layers[layer]
    hook = layer_module.register_forward_hook(_get_layer_output)

    # Forward the prompt
    encoding = tok(prompt, return_tensors="pt").to("cuda")
    with torch.no_grad():
        model(**encoding)

    layer_output = layer_output[0]  # [1, seq_len, hidden_size]
    if return_last_pos:
        layer_output = layer_output[:, -1, :]
    hook.remove()
    return layer_output


def get_input_output_at_module(
    model: AutoModelForCausalLM,
    tok: AutoTokenizer,
    prompt: str,
    module_str: str,
    return_last_pos: bool = False,
):
    assert "mlp" in module_str or "self_attn" in module_str
    module_output = []
    module_input = []

    def _get_module_output(mod, mod_in, mod_out):
        if isinstance(mod_out, torch.Tensor):
            module_output.append(mod_out)
        elif isinstance(mod_out, tuple):
            module_output.append(mod_out[0])
        else:
            raise ValueError
        # if isinstance(mod_in, tuple):
        #     print(f"mod_in: {type(mod_in)}, len: {len(mod_in)}")
        #     module_input.append(mod_in[0])
        # elif isinstance(mod_in, torch.Tensor):
        #     module_input.append(mod_in)
        # else:
        #     raise ValueError

    module = nethook.get_module(model, module_str)
    hook = module.register_forward_hook(_get_module_output)
    # Forward the prompt
    encoding = tok(prompt, return_tensors="pt").to("cuda")
    with torch.no_grad():
        model(**encoding)

    module_output = module_output[0]
    hook.remove()
    if return_last_pos:
        module_output = module_output[:, -1, :]
    return module_output


def get_mlp_input_output(
    model: AutoModelForCausalLM,
    tok: AutoTokenizer,
    prompt: str,
    module_str: str,
    return_last_pos: bool = False,
):
    mlp_output = []
    mlp_input = []

    def _hook_fn(mod, mod_in, mod_out):
        # For down_proj, its input is a tuple with ONE tensor [batch, seq_len, intermediate_size]
        # its output is a tensor [batch, seq_len, hidden_size]
        mlp_output.append(mod_out.clone().detach())
        mlp_input.append(mod_in[0].clone().detach())

    mlp = nethook.get_module(model, module_str)
    hook = mlp.register_forward_hook(_hook_fn)

    encoding = tok(prompt, return_tensors="pt").to("cuda")
    with torch.no_grad():
        model(**encoding)

    hook.remove()

    mlp_input = mlp_input[0]
    mlp_output = mlp_output[0]

    if return_last_pos:
        mlp_input = mlp_input[:, -1, :]
        mlp_output = mlp_output[:, -1, :]
    return mlp_input, mlp_output


def get_attn_input_output(
    model: AutoModelForCausalLM,
    tok: AutoTokenizer,
    prompt: str,
    module_str: str,
    return_last_pos: bool = False,
):
    module_output = []
    module_input = []

    def _get_module_output(mod, mod_in, mod_out):
        if isinstance(mod_in, tuple):
            print(len(mod_in), type(mod_in[0]), mod_in[0].shape)
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
    encoding = tok(prompt, return_tensors="pt").to("cuda")
    with torch.no_grad():
        model(**encoding)

    module_output = module_output[0]
    module_input = module_input[0]
    hook.remove()
    if return_last_pos:
        module_output = module_output[:, -1, :]
        module_input = module_input[:, -1, :]
    return module_input, module_output


def apply_logit_lens(
    model: AutoModelForCausalLM,
    repr: torch.Tensor,
    softmax: bool = False,
) -> torch.Tensor:
    # Obtain the output embedding.
    _norm = model.model.norm
    _lm_head = model.lm_head
    ret = _lm_head(_norm(repr))
    if softmax:
        ret = torch.softmax(ret, dim=-1)
    return ret


def get_all_layer_outputs(
    model: AutoModelForCausalLM,
    tok: AutoTokenizer,
    sentence: str,
    return_last_pos: bool = False,
):
    layer_activations = []
    for layer in range(model.config.num_hidden_layers):
        layer_repr: torch.Tensor = get_repr_at_layer(
            model,
            tok,
            sentence,
            layer=layer,
            return_last_pos=return_last_pos,
        )
        layer_activations.append(layer_repr.cpu())

    layer_activations = torch.cat(layer_activations, dim=0)
    return layer_activations


def get_topk_tokens_from_logits(
    model: AutoModelForCausalLM,
    tok: AutoTokenizer,
    logits: torch.Tensor,
    k: int = 10,
    return_str: bool = True,
):
    prob_distributions: torch.Tensor = apply_logit_lens(model, logits)
    prob_distributions = torch.softmax(prob_distributions, dim=-1)
    topk_values, topk_indices = prob_distributions.topk(k=k, dim=-1)
    topk_indices = topk_indices.reshape(-1)
    topk_values = topk_values.reshape(-1)
    if return_str:
        return [
            {
                "token": t,
                "prob": p.item(),
            }
            for p, t in zip(topk_values, [tok.decode(x) for x in topk_indices])
        ]
    return [
        {
            "token": t.item(),
            "prob": p.item(),
        }
        for p, t in zip(topk_values, topk_indices)
    ]


def get_target_rank(logits: torch.Tensor, token_id: int):
    assert logits.shape[0] == 1
    if logits.dim() == 3:
        token_logit = logits[0, :, token_id]
    else:
        token_logit = logits[0, token_id]
    flat_logits = logits.reshape(-1)
    sorted_logits, sorted_indices = torch.sort(flat_logits, descending=True)
    return torch.where(sorted_logits == token_logit)[0].item()
