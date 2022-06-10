import numpy as np
import pandas as pd
import numpy as np
import torch
import os
import seaborn as sns
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
import operator
from typing import Union
from sklearn.preprocessing import MinMaxScaler
import re

def load_train(data_dir:str) -> pd.DataFrame():
    train_transaction_df = pd.read_csv(os.path.join(data_dir, "train_transaction.csv"))
    train_identity_df = pd.read_csv(os.path.join(data_dir, "train_identity.csv"))
    train_transaction_df = train_transaction_df.merge(train_identity_df,how='left',on='TransactionID')
    return train_transaction_df
train_data=load_train(data_dir="/Users/alexkosa/Desktop/AcLabs/TrainingData")

def reduce_mem_usage(df, verbose=True):
    numerics = ['int16', 'int32', 'int64', 'float16', 'float32', 'float64']
    start_mem = df.memory_usage().sum() / 1024**2
    for col in df.columns:
        col_type = df[col].dtypes
        if col_type in numerics:
            c_min = df[col].min()
            c_max = df[col].max()
            if str(col_type)[:3] == 'int':
                if c_min > np.iinfo(np.int8).min and c_max < np.iinfo(np.int8).max:
                    df[col] = df[col].astype(np.int8)
                elif c_min > np.iinfo(np.int16).min and c_max < np.iinfo(np.int16).max:
                    df[col] = df[col].astype(np.int16)
                elif c_min > np.iinfo(np.int32).min and c_max < np.iinfo(np.int32).max:
                    df[col] = df[col].astype(np.int32)
                elif c_min > np.iinfo(np.int64).min and c_max < np.iinfo(np.int64).max:
                    df[col] = df[col].astype(np.int64)
            else:
                if c_min > np.finfo(np.float16).min and c_max < np.finfo(np.float16).max:
                    df[col] = df[col].astype(np.float16)
                elif c_min > np.finfo(np.float32).min and c_max < np.finfo(np.float32).max:
                    df[col] = df[col].astype(np.float32)
                else:
                    df[col] = df[col].astype(np.float64)

    end_mem = df.memory_usage().sum() / 1024**2
    print('Memory usage after optimization is: {:.2f} MB'.format(end_mem))
    print('Decreased by {:.1f}%'.format(100 * (start_mem - end_mem) / start_mem))

    return df

def delete_features_with_nan(df):
    df.drop(df.columns[df.apply(lambda col: col.isnull().sum() / df.shape[0] > 0.90)], axis=1, inplace=True)
    return df

def adding_new_features(aux):
    aux['TransactionHour'] = aux['TransactionDT'] / 3600 % 24
    aux['TransactionDay'] = aux['TransactionDT'] / (3600 * 24) % 7
    aux.drop(['TransactionDT'], axis=1, inplace=True)
    aux['P_emaildomain'].fillna(value=aux['R_emaildomain'], inplace=True)
    aux['R_emaildomain'].fillna(value=aux['P_emaildomain'], inplace=True)
    aux['P_emaildomain'].fillna(value='missing', inplace=True)
    aux['R_emaildomain'].fillna(value='missing', inplace=True)
    aux['DifferentEmailDomain'] = (aux['P_emaildomain'] != aux['R_emaildomain']).astype(int)
    aux['MissingEmailDomain'] = (aux['P_emaildomain']=='missing').astype(int)
    aux.drop(['P_emaildomain'], axis=1, inplace=True)
    aux.drop(['R_emaildomain'], axis=1, inplace=True)

    return aux

def resolve_string_features(aux):
    remain_string = aux.select_dtypes(include=["object"]).columns.tolist()
    for col in remain_string:
        if len(aux[col].unique()) > 10:
            aux.drop(col, axis=1, inplace=True)

    remain_string = aux.select_dtypes(include=["object"]).columns.tolist()
    for col in remain_string:
        aux[col].fillna(value="missing", inplace=True)
        aux = pd.get_dummies(aux, columns=[col])
    return aux

def fill_nans(df: pd.DataFrame, operation: Union[str, float, int] = "mean") -> pd.DataFrame:
    if isinstance(operation, str):
        assert operation in ("mean", "median")

    is_na_columns = df.isnull().any(axis=0)
    columns_with_na = is_na_columns[is_na_columns == True].index
    df_with_nans = df[columns_with_na]
    df_with_nans = df_with_nans.astype("float32")
    if isinstance(operation, str):
        if operation == "mean":
            fill_values = df_with_nans.mean(axis=0, skipna=True)
        else:
            fill_values = df_with_nans.median(axis=0, skipna=True)

        for column in columns_with_na:
            if 1.*df[column].value_counts().max()/df[column].count() < 0.5:
                df[column] = df[column].fillna(fill_values[column])
        for column in columns_with_na:
            df[column] = df[column].fillna(8*df[column].max())
    else:
        fill_value = operation
        df[columns_with_na] = df[columns_with_na].fillna(fill_value)

    return df

def remove_correlated_features(df):
    corr_mat = df.corr().abs()
    upper_tri = corr_mat.where(np.triu(np.ones(corr_mat.shape), k=1).astype(bool))
    to_drop = [column for column in upper_tri.columns if any(upper_tri[column] > 0.90)]
    df.drop(df[to_drop], axis=1, inplace=True)
    return df
def feature_scaling(df):
    scaler = MinMaxScaler()
    aux3 = scaler.fit_transform(df)
    df = pd.DataFrame(aux3, index=df.index, columns=df.columns)
    return df

def resolve_features_name(df):
    df['id_34_match_status:minus1'] = df['id_34_match_status:-1']
    df.drop('id_34_match_status:-1', axis=1, inplace=True)

    df = df.rename(columns=lambda x: re.sub('[^A-Za-z0-9_]+', '', x))
    return df

def balance_dataset(df):
    aux_isFraud = df.copy()
    aux_isFraud = aux_isFraud.loc[aux_isFraud['isFraud'] == True]
    for i in range(20):
        df = pd.concat([df, aux_isFraud], ignore_index=True)
    df = df.sample(frac=1).reset_index(drop=True)

    return df




train_data = load_train(data_dir="/Users/alexkosa/Desktop/AcLabs/TrainingData")
train_data.drop(['TransactionID'], axis=1, inplace=True)
#train_data=reduce_mem_usage(train_data)
#rows=train_data.shape[0]
#columns=train_data.shape[1]

train_data = delete_features_with_nan(train_data)

train_data = adding_new_features(train_data)

train_data_balanced=balance_dataset(train_data)

train_data = resolve_string_features(train_data)
train_data_balanced=resolve_string_features(train_data_balanced)

train_data = fill_nans(train_data,'mean')
train_data_balanced=fill_nans(train_data_balanced,'mean')

train_data = remove_correlated_features(train_data)
train_data_balanced=remove_correlated_features(train_data_balanced)

train_data  = feature_scaling(train_data)
train_data_balanced=feature_scaling(train_data_balanced)

train_data = resolve_features_name(train_data)
train_data_balanced=resolve_features_name(train_data_balanced)

#torch.save(train_data,"ProcessedTrainingData1.csv")

X_train_unbalanced, X_test_unbalanced = train_test_split(train_data, test_size=0.2)
y_train_unbalanced = X_train_unbalanced['isFraud']
y_test_unbalanced = X_test_unbalanced['isFraud']
X_train_unbalanced.drop('isFraud',axis=1,inplace=True)
X_test_unbalanced.drop('isFraud',axis=1,inplace=True)



X_train_balanced, X_test_balanced = train_test_split(train_data_balanced, test_size=0.2)
y_train_balanced = X_train_balanced['isFraud']
y_test_balanced = X_test_balanced['isFraud']
X_train_balanced.drop('isFraud',axis=1,inplace=True)
X_test_balanced.drop('isFraud',axis=1,inplace=True)



X_train_balanced.to_csv("X_train_balanced.csv",index=False)
X_train_unbalanced.to_csv("X_train_unbalanced.csv",index=False)
X_test_balanced.to_csv("X_test_balanced.csv",index=False)
X_test_unbalanced.to_csv("X_test_unbalanced.csv",index=False)

y_train_balanced.to_csv("y_train_balanced.csv",index=False)
y_train_unbalanced.to_csv("y_train_unbalanced.csv",index=False)
y_test_balanced.to_csv("y_test_balanced.csv",index=False)
y_test_unbalanced.to_csv("y_test_unbalanced.csv",index=False)