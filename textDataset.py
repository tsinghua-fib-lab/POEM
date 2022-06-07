from os import error
import numpy as np
import pandas as pd

from sklearn.preprocessing import LabelEncoder
import math
from torch.utils.data import Dataset
from sklearn.utils import shuffle

class textDataset(Dataset):
    def __init__(self, data_path, data_type):
        self.label_encoder = LabelEncoder()
        self.data_type = data_type
        if data_type == 'train':
            self.data, self.labels, self.field_dims = self.load_fm_dataset(data_path, data_type)
        elif data_type == 'valid':
            self.data, self.labels = self.load_fm_dataset(data_path, data_type)
        elif data_type == 'test':
            self.data, self.labels = self.load_fm_dataset(data_path, data_type)
        else:
            raise NotImplementedError
    def __getitem__(self, index):
            # return self.data[index], self.labels[index]
            return self.data[index], self.labels[index]
    def __len__(self):
        return len(self.data)
    def load_fm_dataset(self, data_path, data_type):
        if data_type == 'train':
            print('Reading train Datasets...')
            if data_path == 'data_a':
                fac_data = pd.read_csv('./data_a/train_fac_code.csv')
            elif data_path == 'data_b':
                fac_data = pd.read_csv('./data_b/train_fac_code.csv')
            else:
                raise NotImplementedError
            fac_data = fac_data[['value', 'user_id', 'item_id','married', 'age', 'sex', 'job', 'kid_tag', 'car_owner', 'client', 'edu', 'user_location_type', 
            'second_cate_id','third_cate_id','is_click']]
            fac_data = fac_data.rename(columns={'is_click':'label'})
            if data_path == 'data_a':
                cf_data = pd.read_csv('./data_a/cf_data_code.csv')
            elif data_path == 'data_b':
                cf_data = pd.read_csv('./data_b/cf_data_code.csv')
            else:
                raise NotImplementedError
            cf_data = cf_data[['value', 'user_id', 'item_id','married', 'age', 'sex', 'job', 'kid_tag', 'car_owner', 'client', 'edu', 'user_location_type', 
            'second_cate_id','third_cate_id','is_click']]
            cf_data = cf_data.rename(columns={'is_click':'label'})
            cf_data = cf_data.dropna(axis=0, how='any')
            
            all_data = pd.concat([cf_data, fac_data], axis=0)
            # all_data = fac_data
            all_data = all_data.dropna(axis=0, how='any')
            all_data = all_data.drop_duplicates()
            train_data = all_data.drop("label", axis=1)
            if data_path == 'data_a':
                valid_data = pd.read_csv('./data_a/valid_fac_code.csv')
                test_data = pd.read_csv('./data_a/test_fac_code.csv')
            elif data_path == 'data_b':
                valid_data = pd.read_csv('./data_b/valid_fac_code.csv')
                test_data = pd.read_csv('./data_b/test_fac_code.csv')
            else:
                raise NotImplementedError
            valid_data = valid_data[['value', 'user_id', 'item_id','married', 'age', 'sex', 'job', 'kid_tag', 'car_owner', 'client', 'edu', 'user_location_type', 
            'second_cate_id','third_cate_id','is_click']]
            test_data = test_data[['value', 'user_id', 'item_id','married', 'age', 'sex', 'job', 'kid_tag', 'car_owner', 'client', 'edu', 'user_location_type', 
            'second_cate_id','third_cate_id','is_click']]
            valid_data = valid_data.rename(columns={'is_click':'label'})
            test_data = test_data.rename(columns={'is_click':'label'})
            valid_data = valid_data.drop_duplicates()
            test_data = test_data.drop_duplicates()
            valid_feat = valid_data.drop('label', axis=1)
            test_feat = test_data.drop('label', axis=1)
            full = pd.concat([train_data, valid_feat, test_feat], axis=0)
            field_dims = full.nunique()
            return train_data.values, all_data["label"].values, field_dims.values
        elif data_type == 'valid':
            print('Reading valid datasets...')
            if data_path == 'data_a':
                valid_data = pd.read_csv('./data_a/valid_fac_code.csv')
            elif data_path == 'data_b':
                valid_data = pd.read_csv('./data_b/valid_fac_code.csv')
            else:
                raise NotImplementedError
            valid_data = valid_data[[ 'value', 'user_id', 'item_id','married', 'age', 'sex', 'job', 'kid_tag', 'car_owner', 'client', 'edu', 'user_location_type', 
            'second_cate_id','third_cate_id','is_click']]
            valid_data = valid_data.rename(columns={'is_click':'label'})
            valid_data = valid_data.drop_duplicates()
            feat = valid_data.drop('label', axis=1)
            return feat.values, valid_data["label"].values
        elif data_type == 'test':
            print('Reading test datasets...')
            if data_path == 'data_a':
                test_data = pd.read_csv('./data_a/test_fac_code.csv')
            elif data_path == 'data_b':
                test_data = pd.read_csv('./data_b/test_fac_code.csv')
            else:
                raise NotImplementedError
            test_data = test_data[[ 'value', 'user_id', 'item_id','married', 'age', 'sex', 'job', 'kid_tag', 'car_owner', 'client', 'edu', 'user_location_type', 
            'second_cate_id','third_cate_id','is_click']]
            test_data = test_data.rename(columns={'is_click':'label'})
            test_data = test_data.drop_duplicates()
            feat = test_data.drop('label', axis=1)
            return feat.values, test_data["label"].values
        else:
            raise NotImplementedError
