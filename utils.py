"""
helper functions, including functions that load data and train model
"""

from config import categorical_vars, non_features

from torchmetrics import Metric
from torchmetrics.utilities import rank_zero_warn
from typing import Optional
import torch.utils.data as data_utils
import torch
from sklearn.model_selection import train_test_split
import torch.optim as optim

import datetime as dt
import pandas as pd
import numpy as np
import copy


def load_data(file_path):
    """
    loads data from file path. Pads data for customers with less than 13 periods of data
    Args:
        file_path: file path
    Output:
        pandas data frame of cleaned data
    """
    
    parquet_file = file_path
    df = pd.read_parquet(parquet_file,engine="auto")
    df["obs_count"] = df[['customer_ID','S_2']].groupby(['customer_ID']).transform('count')

    # a) add time time index
    df13 = df[df['obs_count']==13]
    new_col =  [i for i in range(13)]*int(len(df13)/13)
    df13.insert(2, "time_id",new_col, True)

    # b) padd incomplete series
    
    #store features
    features = list(df13.columns.difference(non_features))
    #gen new df list - we'll eventually merge them all
    padded_dfs =[df13]

    for i in range(1,13):
        # first we prep a chunk of good data
        df_temp = df[df['obs_count']==i]
        new_col =  [i for i in range(13-i,13)]*int(len(df_temp)/(i))
        df_temp.insert(2, "time_id",new_col, True)
        
        #next we'll pad with -1000
        pad_value = -10000
        row_template = df_temp[df_temp["time_id"]==12]

        for feature in features:
            row_template[feature].values[:] = pad_value
        row_template["S_2"] = pad_value #force s_2 to minus 1000, this won't actually matter though
        row_template["time_id"] = 0 #force to 0


        dfs = [df_temp]
        for j in range(0,13-i):
            df_temp_ = copy.deepcopy(row_template)
            df_temp_["time_id"] = j #add padded value by time
            dfs.append(df_temp_)

        df_concat = pd.concat(dfs)

        # append df
        padded_dfs.append(df_concat)


    df_padded = pd.concat(padded_dfs)  

    # c) turn categorical to binary variables
    for cat in categorical_vars:
        new_cols = pd.get_dummies(df_padded[cat],prefix=cat)
        df_padded = pd.concat((df_padded, new_cols), axis=1)
        df_padded = df_padded.drop([cat], axis=1) #drop cat col
    
    # d) fill nas and make numeric
    df_padded = df_padded.fillna(df_padded.mean()) 
    features =  list(df_padded.columns.difference(non_features)) #remove target, old time id
    df_padded[features] = df_padded[features].apply(pd.to_numeric) #convert features to numeric

    df_padded.sort_values(by=['customer_ID','Age'], ascending=False) #sort values by id and time sequence

    return(df_padded)

# amex metric


class AmexMetric(Metric):
    is_differentiable: Optional[bool] = False

    # Set to True if the metric reaches it optimal value when the metric is maximized.
    # Set to False if it when the metric is minimized.
    higher_is_better: Optional[bool] = True

    # Set to True if the metric during 'update' requires access to the global metric
    # state for its calculations. If not, setting this to False indicates that all
    # batch states are independent and we will optimize the runtime of 'forward'
    full_state_update: bool = True

    def __init__(self):
        super().__init__()

        self.add_state("all_true", default=[], dist_reduce_fx="cat")
        self.add_state("all_pred", default=[], dist_reduce_fx="cat")

        rank_zero_warn(
            "Metric `Amex` will save all targets and predictions in buffer."
            " For large datasets this may lead to large memory footprint."
        )

    def update(self, y_pred: torch.Tensor, y_true: torch.Tensor):
        y_true = y_true.double()
        y_pred = y_pred.double()

        self.all_true.append(y_true)
        self.all_pred.append(y_pred)

    def compute(self):
        y_true = torch.cat(self.all_true)
        y_pred = torch.cat(self.all_pred)
        # count of positives and negatives
        n_pos = y_true.sum()
        n_neg = y_pred.shape[0] - n_pos

        # sorting by descring prediction values
        indices = torch.argsort(y_pred, dim=0, descending=True)
        preds, target = y_pred[indices], y_true[indices]

        # filter the top 4% by cumulative row weights
        weight = 20.0 - target * 19.0
        cum_norm_weight = (weight / weight.sum()).cumsum(dim=0)
        four_pct_filter = cum_norm_weight <= 0.04

        # default rate captured at 4%
        d = target[four_pct_filter].sum() / n_pos
        print(d)
        # weighted gini coefficient
        lorentz = (target / n_pos).cumsum(dim=0)
        gini = ((lorentz - cum_norm_weight) * weight).sum()

        # max weighted gini coefficient
        gini_max = 10 * n_neg * (1 - 19 / (n_pos + 20 * n_neg))

        # normalized weighted gini coefficient
        g = gini / gini_max

        return 0.5 * (g + d)


def amex_metric(y_true: pd.DataFrame, y_pred: pd.DataFrame) -> float:
    def top_four_percent_captured(y_true: pd.DataFrame, y_pred: pd.DataFrame) -> float:
        df = (pd.concat([y_true, y_pred], axis='columns')
              .sort_values('prediction', ascending=False))
        df['weight'] = df['target'].apply(lambda x: 20 if x == 0 else 1)
        four_pct_cutoff = int(0.04 * df['weight'].sum())
        df['weight_cumsum'] = df['weight'].cumsum()
        df_cutoff = df.loc[df['weight_cumsum'] <= four_pct_cutoff]
        return (df_cutoff['target'] == 1).sum() / (df['target'] == 1).sum()

    def weighted_gini(y_true: pd.DataFrame, y_pred: pd.DataFrame) -> float:
        df = (pd.concat([y_true, y_pred], axis='columns')
              .sort_values('prediction', ascending=False))
        df['weight'] = df['target'].apply(lambda x: 20 if x == 0 else 1)
        df['random'] = (df['weight'] / df['weight'].sum()).cumsum()
        total_pos = (df['target'] * df['weight']).sum()
        df['cum_pos_found'] = (df['target'] * df['weight']).cumsum()
        df['lorentz'] = df['cum_pos_found'] / total_pos
        df['gini'] = (df['lorentz'] - df['random']) * df['weight']
        return df['gini'].sum()

    def normalized_weighted_gini(y_true: pd.DataFrame, y_pred: pd.DataFrame) -> float:
        y_true_pred = y_true.rename(columns={'target': 'prediction'})
        return weighted_gini(y_true, y_pred) / weighted_gini(y_true, y_true_pred)

    g = normalized_weighted_gini(y_true, y_pred)
    d = top_four_percent_captured(y_true, y_pred)

    return 0.5 * (g + d)


class LoadedData:
    def __init__(self, df):
        # extract feature info
        features = list(df.columns.difference(non_features))
        in_features = len(features)

        # gen tensors
        all_tensor_x = torch.reshape(torch.tensor(df[features].to_numpy()), (-1, 13, in_features)).float()
        all_tensor_y = torch.tensor(df.groupby('customer_ID').first()['target'].to_numpy()).float()

        # split
        X_trainval, X_test, y_trainval, y_test = train_test_split(all_tensor_x, all_tensor_y, test_size=0.1,
                                                                  random_state=1)

        # training_set = Dataset(X_trainval, y_trainval)
        # validation_set = Dataset(X_test, y_test)

        training_set = torch.utils.data.TensorDataset(X_trainval, y_trainval)
        validation_set = torch.utils.data.TensorDataset(X_test, y_test)

        print(y_trainval.shape[0], y_test.shape[0])

        # initialise data loaders
        batch_size = 50
        trainloader = torch.utils.data.DataLoader(training_set, batch_size=batch_size,
                                                  shuffle=True, num_workers=1)
        testloader = torch.utils.data.DataLoader(validation_set, batch_size=batch_size,
                                                 shuffle=False, num_workers=1)

        # add attributes
        self.loadedTrain = trainloader
        self.loadedTest = testloader


def train_model(dfLoaded_, model_, epoch):
    criterion = torch.nn.BCELoss()
    optimizer = optim.Adam(model_.parameters(), lr=0.00005)

    running_loss = 0.0
    print("-" * 25 + str(epoch) + "-" * 25)
    for i, data in enumerate(dfLoaded_.loadedTrain, 0):
        inputs, labels = data

        labels = torch.reshape(labels, (labels.shape[0], 1))
        optimizer.zero_grad()
        # print(inputs.shape)
        outputs = model_(inputs)
        # print(outputs)
        # print(labels)

        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
        # print statistics
        running_loss += loss.item()
        fmt = "%m.%d.%Y"
        torch.save(
            model_,
            "Models/" + f"{dt.datetime.now().strftime(fmt)} Transformer Encoder Only LARGE{epoch}.pt"
        )
        if i % 20 == 0:
            print(loss.detach().item())
        # pprint for each epoch
    print('loss:' + str(round(running_loss / (i + 1), 4)))


def eval_model(dfLoaded_, model_):
    correct = 0
    total = 0

    correct_positives = 0
    total_positives = 0

    all_labels = []
    all_predictions = []
    # since we're not training, we don't need to calculate the gradients for our outputs
    with torch.no_grad():
        for i, data in enumerate(dfLoaded_.loadedTest, 0):
            inputs, labels = data
            labels = torch.reshape(labels, (labels.shape[0], 1))
            labels_pred = [1 if i > 0.5 else 0 for i in labels]
            # calculate outputs by running inputs through the network
            outputs = model_(inputs)
            predicted = [1 if i > 0.5 else 0 for i in outputs]
            # add to counts
            total += labels.size(0)
            correct += sum([predicted[i] == labels_pred[i] for i in range(len(predicted))])

            total_positives += sum([labels_pred[i] for i in range(len(labels_pred))])
            correct_positives += sum(
                [labels_pred[i] * (predicted[i] == labels_pred[i]) for i in range(len(labels_pred))])
            all_labels.append(labels)
            all_predictions.append(outputs)
    all_labels = torch.cat(all_labels, dim=0)
    all_predictions = torch.cat(all_predictions, dim=0)
    amexMetric_ = amex_metric(
        pd.DataFrame(all_labels.detach().numpy(), columns=["target"]),
        pd.DataFrame(all_predictions.detach().numpy(), columns=["prediction"]))

    print("Accuracy of the network on the %d test datapoints: %d percent " % (total, 100 * correct // total))
    print("Accuracy of the network on the %d positive test datapoints: %d percent " % (
        total_positives, 100 * correct_positives // total_positives))
    print("The Amex metric is : ", amexMetric_)
