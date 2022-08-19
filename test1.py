import datetime

from AmexDataset import AmexDataset
import torch
import matplotlib.pyplot as plt

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
    train_len = len(data) * 9 // 10
    test_len = len(data) // 10 + 1
    train_dataset, test_dataset = data_utils.random_split(data, [train_len, test_len])
    print("Data Loaded")

    train_loader = torch.utils.data.DataLoader(data, batch_size=batch_size,
                                               shuffle=True, num_workers=5)

    test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=150,
                                              shuffle=True, num_workers=5)
    try:
        model = torch.load("Models/08.18.2022 Transformer Encoder Only LARGE 0.pt")
        raise FileNotFoundError
    except FileNotFoundError:
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

    optimizer = optim.Adam(model.parameters(), lr=0.000001)
    model.to(device)
    print(device)
    train_loss = []
    amex = []
    for epoch in range(25):

        train_loss.append(train_model(train_loader=train_loader,
                                      model_=model, optimizer=optimizer, epoch=epoch, length=train_len))
        amex.append(eval_model(test_loader=test_loader,
                               model_=model, length=test_len))
    plt.plot(list(range(len(amex))), amex, color="red", label="Amex Metric")
    plt.plot(list(range(len(train_loss))), train_loss, color="blue", label="Training Loss")
    plt.legend()
    plt.title("Amex Metric and Training loss")
    plt.savefig("Chart/Loss chart.png")
    plt.show()

