import sys
import torch
from Network.lstm import LSTM
from Network.conditional_net import ConditionalNet
from Network.feedforward_net import FeedforwardNet
from torch import nn
import pandas as pd


def get_macro_by_date(macro_df, begin_date, end_date):
    sub_df = macro_df[(macro_df["sasdate"] >= begin_date)
                      & (macro_df["sasdate"] < end_date)]
   
    sub_df = sub_df.drop("sasdate", axis=1)
    out = sub_df.values
    return out


def get_feature_data_by_date(feature_df, date):
    next_month = (date + 1) if (date % 100 < 12) else (date + 89)
    ret = feature_df[feature_df["time_avail_m"] == next_month][["permno","RET"]]
    ret_cp = ret.copy()
    feature = feature_df[feature_df["time_avail_m"] == date][feature_df.columns[0:29]]
    feature_cp = feature.copy()
    merge_df = pd.merge(feature_cp, ret_cp, left_on=["permno"],right_on=["permno"])

    merge_df.drop(merge_df.columns[[0, 1]], axis=1, inplace=True)
    
    feature_list = []
    ret_list = []
    for index, row in merge_df.iterrows():
        array = row.values
        feature = array[0:27]
        ret = array[27]
        feature_list.append(feature)
        
        ret_list.append(ret)
        
    feature_array = np.array(feature_list)
    ret_array = np.array(ret_list)
    return feature_array,ret_array

MACRO_SIZE = 124
LSTM_HIDDEN = 128
LSTM_LAYER = 1
CHAR_SIZE = 27
FF_HIDDEN = [32, 64, 32]
FF_LAYER = 3

conditional_net = ConditionalNet(
    MACRO_SIZE, CHAR_SIZE, LSTM_HIDDEN, LSTM_LAYER, FF_HIDDEN, FF_LAYER)


macro_df = pd.read_csv("./data/124_macro_data.csv")
macro_x = get_macro_by_date(macro_df, 199901, 201701)

feature_df = pd.read_csv("./data/27_features_rets_normalized.csv")
feature_x = get_feature_data_by_date(feature_df, 201701)

g_hat = conditional_net(feature_x[0][0], macro_x)
print(g_hat)
