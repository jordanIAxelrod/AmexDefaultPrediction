import datetime

from AmexDataset import AmexDataset
import json
import torch

"""
 test training function
"""

from transformer import *
from utils import *
from config import *

if __name__ == "__main__":
    print("Create Dataset")

    data = AmexDataset("inputs", "inputs/train_labels.csv")

    now = datetime.datetime.now()
    print(data[10000])
    print(datetime.datetime.now() - now)
    print(categorical_vars)
    train_dataset, test_dataset = data_utils.random_split(data, [len(data) * 9 // 10, len(data) // 10 + 1])
    print("Data Loaded")

    train_loader = torch.utils.data.DataLoader(data, batch_size=batch_size,
                                               shuffle=True, num_workers=1)

    test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=batch_size,
                                              shuffle=True, num_workers=5)
    model = Transformer(in_features=in_features,
                        enc_features=132,
                        n_heads=11,
                        qkv_bias=True,
                        mlp_ratio=8,
                        p=.1,
                        attn_p=.1,
                        max_len=13,
                        n_classes=1,
                        depth=15)
    model.to(device)
    print(device)
    for epoch in range(25):
        train_model(train_loader=train_loader,
                    model_=model, epoch=epoch)

        eval_model(test_loader=test_loader,
                   model_=model)
