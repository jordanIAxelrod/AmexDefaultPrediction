from config import *
from tqdm import tqdm
import pandas as pd
import json
import numpy as np


def get_categories(data_path, chunksize):
    length = 5540000 // chunksize + 1
    data = pd.read_csv(data_path, chunksize=chunksize)
    for chunk in tqdm(data, total=length):
        for cat in categorical_vars.keys():
            vals = pd.unique(chunk[cat])
            for val in vals:
                value_condition = not pd.isna(val) and val not in categorical_vars[cat]
                if value_condition:
                    categorical_vars[cat].append(val)
    with open("inputs/categorical variables.json", "w") as fp:
        json.dump(categorical_vars, fp)


# Add categorical data to amex_data, remove categorical data and fill na save it out in chunks

def clean(data_url, out_url, chunksize):
    data = pd.read_csv(data_url, chunksize=chunksize)
    header = True
    count = 0
    length = 5540000 // chunksize + 1
    for chunk in tqdm(data, total=length):
        print(count)
        for cat in categorical_vars.keys():
            new_rows = chunk[cat].apply(lambda
                                            x: [1 if x == j else 0 for j in categorical_vars[cat]])

            new_cols = pd.DataFrame(new_rows, columns=[f"{cat}_{j}" for j in categorical_vars[cat]])
            chunk = pd.concat((chunk, new_cols), axis=1)
            chunk = chunk.drop([cat], axis=1)  # drop cat col

        # d) fill nas and make numeric
        chunk = chunk.fillna(0)
        features = list(chunk.columns.difference(non_features))  # remove target, old time id
        chunk[features] = chunk[features].apply(pd.to_numeric)  # convert features to numeric
        chunk.to_csv(out_url, header=header, mode='a')
        header = False
        count += 1


def make_csv(file_path: str, start_end_info: dict, idx: list, kind: str):
    start = start_end_info[str(idx[0])]["start"]
    n_rows = sum([start_end_info[str(i)]["n_rows"] for i in idx])
    df = pd.read_csv(
        file_path,
        skiprows=[i for i in range(1, start + 1)],
        nrows=n_rows,
        index_col="Unnamed: 0"
    )
    print(df.index)
    for i in idx:
        n_rows = start_end_info[str(i)]["n_rows"]
        start = start_end_info[str(i)]["start"]
        local = df.loc[start: start + n_rows - 1, :]
        new_col = list(range(13 - n_rows, 13))
        if len(pd.unique(local['customer_ID'])) > 1:
            raise ValueError(
                f"Not all the same customer, index is {start} customers are {pd.unique(local['customer_ID'])}")
        try:
            local.insert(2, "time_id", new_col, True)
        except Exception as e:
            print(start, n_rows, len(local))
            print(df.loc[start - 1: start + n_rows + 2])
            raise e

        if n_rows < 13:
            new_rows = [pd.DataFrame(-np.infty, columns=local.columns, index=[1]) for i in range(13 - n_rows)]

            new_rows.append(local)
            local = pd.concat(new_rows)
        features = list(local.columns.difference(non_features))
        if kind == "train":
            local[features].to_csv(f"inputs/{i}.csv", index=False)
        else:
            local[features].to_csv(f"inputs/submission/i.csv", index=False)


def start_end_dict(file_path: str, chunksize: int, kind: str):
    df = pd.read_csv(file_path, chunksize=chunksize)
    start_end_dicts = {}
    count = 0
    for chunk in df:
        group = chunk.groupby(["customer_ID"], as_index=False)
        n_rows = group["customer_ID"].count()
        idxs = group.first().index
        cust_IDs = pd.unique(chunk["customer_ID"])
        for i in range(len(n_rows)):
            start_end_dicts[count] = {
                "start": idxs[i],
                "n_rows": n_rows[i],
                "cust_ID": cust_IDs[i]
            }
    if kind == "train":
        out_url = "inputs/start_end_info.json"
    else:
        out_url = "inputs/submission/start_end_info.json"
    with open(out_url, "w") as file:
        json.dump(start_end_dicts, file)


if __name__ == "__main__":
    # get_categories("inputs/train_data.csv", 120000)
    # clean("inputs/train_data.csv", "inputs/train_data_cat.csv", 1000000)
    with open("inputs/start_end_info.json") as fp:
        start_end = json.load(fp)
    sub = []
    keys = list(start_end.keys())
    for idx in tqdm(keys):
        sub.append(idx)
        if len(sub) == 100000 or idx == keys[-1]:
            make_csv("inputs/train_data_cat.csv", start_end, sub, "train")
            sub = []
