import pandas as pd
import numpy as np
import torch
from sklearn.metrics import roc_auc_score
import os
from catboost import CatBoostClassifier

data_dir="/Users/alexkosa/Desktop/AcLabs"
X_train_unbalanced = pd.read_csv(os.path.join(data_dir, "X_train_unbalanced.csv"))
X_test_unbalanced = pd.read_csv(os.path.join(data_dir, "X_test_unbalanced.csv"))
X_train_balanced = pd.read_csv(os.path.join(data_dir, "X_train_balanced.csv"))
X_test_balanced = pd.read_csv(os.path.join(data_dir, "X_test_balanced.csv"))

y_train_unbalanced = pd.read_csv(os.path.join(data_dir, "y_train_unbalanced.csv"))
y_test_unbalanced = pd.read_csv(os.path.join(data_dir, "y_test_unbalanced.csv"))
y_train_balanced = pd.read_csv(os.path.join(data_dir, "y_train_balanced.csv"))
y_test_balanced = pd.read_csv(os.path.join(data_dir, "y_test_balanced.csv"))

y_train_unbalanced = y_train_unbalanced.squeeze()
y_test_unbalanced = y_test_unbalanced.squeeze()
y_train_balanced = y_train_balanced.squeeze()
y_test_balanced = y_test_balanced.squeeze()

catboostbalanced=CatBoostClassifier()
catboostbalanced.fit(X_train_balanced,y_train_balanced)

torch.save(catboostbalanced,"CatBoostOnBalancedModel")

Y_predict=catboostbalanced.predict_proba(X_test_balanced)

print("For balanced data : "+str(roc_auc_score(y_test_balanced,Y_predict[:,1])))

catboostunbalanced=CatBoostClassifier()
catboostunbalanced.fit(X_train_unbalanced,y_train_unbalanced)

torch.save(catboostunbalanced,"CatBoostOnUnbalancedModel")

Y_predict=catboostunbalanced.predict_proba(X_test_unbalanced)

print("For unbalanced data : "+str(roc_auc_score(y_test_unbalanced,Y_predict[:,1])))
