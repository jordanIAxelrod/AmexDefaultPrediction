"""
 test training function
"""

from transformer import *
from utils import *
from config import *

if __name__ == "__main__":

    df_raw = load_data("/Users/Aex/Desktop/Kaggle/Amex/data/amex-sample.parquet")
    df = df_raw.head(13*10) #obtain first 5000 / 19549 ids
    features = list(df.columns.difference(non_features))
    in_features = len(features)

    dfloaded = LoadedData(df) #prep data

    model  = Transformer(in_features=in_features,
                     enc_features= 100,
                     n_heads= 10,
                     qkv_bias= True,
                     mlp_ratio = 10,
                     p = .1,
                     attn_p =.1,
                     max_len =13,
                     n_classes =1,
                     depth =8)

    train_model(dfLoaded_ = dfloaded,
            model_  = model)

    eval_model(dfLoaded_ = dfloaded,
            model_  = model)

