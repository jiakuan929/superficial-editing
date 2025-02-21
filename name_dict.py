from baselines import apply_rome_to_model
from baselines import apply_memit_to_model
from baselines import apply_ft_to_model
from baselines import apply_r_rome_to_model
from baselines import apply_emmet_to_model
from baselines import apply_pmet_to_model
from baselines import apply_jeep_to_model
from baselines import apply_AlphaEdit_to_model
from baselines import MendRewriteExecutor


# Replace the following model paths to yours!
# We have modify the original code for the changed path in globals.py, layer_stats.py
MODEL_NAME_DICT = {
    "llama3-8b": "/mnt/publiccache/huggingface/Meta-Llama-3-8B/",
    "llama3-8b-it": "/mnt/publiccache/huggingface/Meta-Llama-3-8B-Instruct/",
    "gpt2-xl": "/mnt/publiccache/huggingface/gpt2-xl/",
    "llama3.2-3b-it": "/mnt/publiccache/huggingface/Llama-3.2-3B-Instruct/",
    "qwen2.5-3b-it": "/mnt/publiccache/huggingface/Qwen2.5-3B-Instruct/",
    "qwen2.5-7b-it": "/mnt/publiccache/huggingface/Qwen2.5-7B-Instruct/",
    "qwen2.5-14b-it": "/mnt/publiccache/huggingface/Qwen2.5-14B-Instruct/",
    "qwen2.5-32b-it": "/mnt/publiccache/huggingface/Qwen2.5-32B-Instruct/",
}


DATASET_DICT = {
    "counterfact_wiki": "./exp_datas/{model}/counterfact/wiki.json",
    "counterfact_rep": "./exp_datas/{model}/counterfact/rep.json",
    "counterfact_puz": "./exp_datas/{model}/counterfact/puz.json",
    "zsre_wiki": "./exp_datas/{model}/zsre/wiki.json",
    "zsre_rep": "./exp_datas/{model}/zsre/rep.json",
    "zsre_puz": "./exp_datas/{model}/zsre/puz.json",
}

ANALYSIS_DATASET_DICT = {
    "counterfact": "./exp_datas/analysis_full/{model}/{editor}_counterfact.json",
}

# ANALYSIS_DATA2 = {
#     "counterfact": "./exp_datas/analysis_full_2/{model}/{editor}_counterfact.json",
# }

# ANALYSIS_CZ = {
#     "llama3-8b-it_rome": "./exp_datas/analysis_cz/llama3-8b-it/rome.json",
#     "llama3-8b-it_memit": "./exp_datas/analysis_cz/llama3-8b-it/memit.json",
#     "qwen2.5-7b-it_rome": "./exp_datas/analysis_cz/qwen2.5-7b-it/rome.json",
#     "qwen2.5-7b-it_memit": "./exp_datas/analysis_cz/qwen2.5-7b-it/memit.json",
#     "qwen2.5-14b-it_rome": "./exp_datas/analysis_cz/qwen2.5-14b-it/rome.json",
#     "qwen2.5-14b-it_memit": "./exp_datas/analysis_cz/qwen2.5-14b-it/memit.json"
# }

ALG_DICT = {
    "rome": apply_rome_to_model,
    "r_rome": apply_r_rome_to_model,
    "memit": apply_memit_to_model,
    "ft": apply_ft_to_model,
    "emmet": apply_emmet_to_model,
    "pmet": apply_pmet_to_model,
    "jeep": apply_jeep_to_model,
    "alphaedit": apply_AlphaEdit_to_model,
    "mend": MendRewriteExecutor().apply_to_model
}
