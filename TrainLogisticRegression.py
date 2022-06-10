import pandas as pd
import numpy as np
import torch
from sklearn.metrics import roc_auc_score
import os
from sklearn.linear_model import LogisticRegression

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

print("data loaded")

logB=LogisticRegression(max_iter=3000)
logU=LogisticRegression(max_iter=3000)
logB.fit(X_train_balanced,y_train_balanced)
logU.fit(X_train_unbalanced,y_train_unbalanced)

torch.save(logB,"LogisticRegressionOnBalanced")
torch.save(logU,"LogisticRegressionOnUnbalanced")

print("Balanced :"+str(roc_auc_score(y_test_balanced,logB.predict_proba(X_test_balanced)[:,1])))
print("Unbalanced :"+str(roc_auc_score(y_test_unbalanced,logU.predict_proba(X_test_unbalanced)[:,1])))