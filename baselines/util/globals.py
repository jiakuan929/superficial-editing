import os
from pathlib import Path

data =   {
    "RESULTS_DIR": "results",

    # Data files
    "DATA_DIR": "/mnt/publiccache/xiejiakuan/data",
    "STATS_DIR": "/mnt/publiccache/xiejiakuan/data/stats",
    "KV_DIR": "/share/projects/rewriting-knowledge/kvs",
    "HF_DATASET_CACHE_DIR": "/mnt/publiccache/huggingface/datasets/",

    # Hyperparameters
    "HPARAMS_DIR": "hparams",

    # Remote URLs
    "REMOTE_ROOT_URL": "https://memit.baulab.info"
}

# Customization for openpai platform
for key in ["DATA_DIR", "STATS_DIR", "HF_DATASET_CACHE_DIR"]:
    if not os.path.exists(data[key]):
        data[key] = data[key].replace("publiccache", "usercache")

(RESULTS_DIR, DATA_DIR, STATS_DIR, HPARAMS_DIR, KV_DIR, HF_DATASET_CACHE_DIR) = (
    Path(z)
    for z in [
        data["RESULTS_DIR"],
        data["DATA_DIR"],
        data["STATS_DIR"],
        data["HPARAMS_DIR"],
        data["KV_DIR"],
        data["HF_DATASET_CACHE_DIR"],
    ]
)

REMOTE_ROOT_URL = data["REMOTE_ROOT_URL"]
