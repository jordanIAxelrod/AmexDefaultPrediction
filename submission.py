import os

from config import *
from Clean import clean, make_csv, start_end_dict
import json
from tqdm import tqdm
import AmexDataset
import pandas as pd


def main():
    try:
        with open("inputs/submission/start_end_info.json") as fp:
            start_end = json.load(fp)
    except FileNotFoundError:
        start_end_dict("inputs/submission/test_data.csv", 1000000, "test")
        with open("inputs/submission/start_end_info.json") as fp:
            start_end = json.load(fp)
    submissions = os.listdir("inputs/submission")
    if len(submissions) < 924621:
        clean("inputs/test_data.csv", "inputs/test_data_cat.csv", 1000000)

        keys = list(start_end.keys())
        sub = []
        for idx in tqdm(keys):
            sub.append(idx)
            if len(sub) == 200000 or idx == keys[-1]:
                make_csv("inputs/test_data_cat.csv", start_end, sub, "submission")
                sub = []

    data = AmexDataset.AmexDataset("inputs/submission")

    dataloader = torch.utils.data.dataloader(data, batch_size=150, shuffle=False, num_workers=8)

    max_model = max(os.listdir("Models"), key=lambda x: int(x.split(' ')[-1].split('.')[0]))
    model = torch.load(f'Model/{max_model}')
    model.to(device)
    model.eval()
    all_predictions = []
    with torch.no_grad():
        for features in dataloader:
            features, mask = features[0].to(device), features[1].to(device)
            all_predictions.append(model(features, mask).cpu())
    all_predictions = torch.cat(all_predictions, dim=0)
    cust_IDs = [v["cust_ID"] for v in start_end.values()]
    df = pd.DataFrame([cust_IDs, all_predictions], columns=["Customer_ID", "prediction"])
    df.to_csv("output/predictions.csv")


if __name__ == "__main__":
    main()
