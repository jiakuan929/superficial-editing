import json
import os
from pathlib import Path

import torch
from torch.utils.data import Dataset

from ..util.globals import *

REMOTE_URL = f"{REMOTE_ROOT_URL}/data/dsets/known_1000.json"


class KnownsDataset(Dataset):
    def __init__(self, data_dir: str, *args, **kwargs):
        if os.path.isdir(data_dir):
            data_dir = f"{data_dir}/known_1000.json"
        with open(data_dir, "r") as f:
            self.data = json.load(f)

        print(f"Loaded dataset with {len(self)} elements")

    def __len__(self):
        return len(self.data)

    def __getitem__(self, item):
        return self.data[item]