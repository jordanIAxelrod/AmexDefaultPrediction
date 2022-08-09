"""
 test training function
"""

from transformer import *
from utils import *
from config import *

if __name__ == "__main__":
    print("getting dat")
    df_raw = load_data("Data/Sample/amex-sample.parquet")
    print("Data Loaded")
    df = df_raw.head(5000*13)  # obtain first 5000 / 19549 ids
    features = list(df.columns.difference(non_features))
    in_features = len(features)

    dfloaded = LoadedData(df)  # prep data

    model = Transformer(in_features=in_features,
                        enc_features=132,
                        n_heads=11,
                        qkv_bias=True,
                        mlp_ratio=9,
                        p=.1,
                        attn_p=.1,
                        max_len=13,
                        n_classes=1,
                        depth=14)
    # model = torch.load("Models/08.08.2022 Transformer Encoder Only 6.pt")
    amex = AmexMetric()
    for epoch in range(25):
        train_model(dfLoaded_=dfloaded,
                    model_=model, epoch=epoch)

        eval_model(dfLoaded_=dfloaded,
                   model_=model)
