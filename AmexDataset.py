import os

import pandas as pd
import numpy as np
from torch.utils.data import Dataset

from config import *


class AmexDataset(Dataset):

    def __init__(self, data_dir: str, target_path: str = None):

        if target_path:
            self.target = pd.read_csv(target_path)
        else:
            self.target = None
        self.data_dir = data_dir
        self.data_files = os.listdir(data_dir)
        self.data_files = [file for file in self.data_files if file.split("/")[-1].split(".")[0].isnumeric()]
        self.data_files.sort(key=lambda x: int(x.split("/")[-1].split(".")[0]))
        self.len = len(self.data_files)

    def __len__(self):
        return self.len

    def __getitem__(self, idx: int):
        path = os.path.join(self.data_dir, self.data_files[idx])
        df = pd.read_csv(path)
        features = list(df.columns.difference(non_features))
        mask = df[features] == -np.infty
        msk = [0]
        for i, row in mask.iterrows():
            if all(row):
                msk.append(True)
            else:
                msk.append(False)
        df.replace(to_replace=-np.infty, value=0, inplace=True)

        if self.target is not None:
            return torch.Tensor(df[features].to_numpy()), torch.Tensor([self.target.loc[idx, "target"]]), torch.Tensor(
                msk)
        else:
            return torch.Tensor(df[features].to_numpy()), torch.Tensor(msk)
