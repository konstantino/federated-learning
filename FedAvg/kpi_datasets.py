#!/usr/bin/env python
# -*- coding: utf-8 -*-
# Python version: 3.6

import torch
import pandas as pd
import numpy as np
from torch.utils.data import Dataset

class KPICSV(Dataset):
    def __init__(self, mode="train"):
        self.mode = mode
        if ( mode is "full" ):
            file = '../data/selims/outLags_Latency_QC1_50.csv'
            
        if ( mode is "train" ):
            file = '../data/selims/outLags_Latency_QC1_50_train.csv'

        if ( mode is "test" ):
            file = '../data/selims/outLags_Latency_QC1_50_test.csv'

        self.csv_file = pd.read_csv(file).drop(
        ["enodeB","year","month","day","time","cell","Unnamed: 0",
                "datetime","prev_datetime","year_1","month_1","day_1","time_1",
                "datetime_1","datetime_2","prev_datetime_1","year_2","month_2","day_2","time_2",
                "datetime_3","prev_datetime_2","prev_datetime_3","year_3","month_3","day_3","time_3",
                "pred_datetime","datetime_y","DLLatecny_y"], axis=1
        )

        #print self.csv_file.info()

    # data set size
    def __len__(self):
        return len(self.csv_file)

    def __getitem__(self,idx):
        row = self.csv_file.iloc[idx]
        #floats = row.drop(["cell","LatencyTarget"]).values.astype(np.float32)
        floats = row.drop(["LatencyTarget"]).values.astype(np.float32)
        data = torch.tensor(floats)

        y_val = self.csv_file.iloc[idx]["LatencyTarget"]
        y = torch.tensor(y_val, dtype=torch.long)
        return data, y
