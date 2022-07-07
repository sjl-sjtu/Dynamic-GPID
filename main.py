import torch
from torch import nn
import torch.nn.functional as F
from torchvision import datasets
from torchvision import transforms
import matplotlib.pyplot as plt
from torch import optim
import torch.utils.data as Data
import numpy as np
import pandas as pd
from data_import import get_data,data_loader
from model import Net
from train import train_model,test_model
from loss_func import VAEloss, RNNloss, predloss
from plotpy import plot_vae, plot_pred, plot_rnn, plot_acc, plot_loss

print(torch.cuda.is_available())
device = 'cuda' if torch.cuda.is_available() else 'cpu'
print(device)

df_geno = pd.read_csv("data//df_geno4.csv")
df_pheno = pd.read_csv("data//df_pheno4.csv")
df_geno_train_vali, df_geno_test, df_pheno_train_vali, df_pheno_test = get_data(df_geno,df_pheno,0.2)
df_geno_train, df_geno_vali, df_pheno_train, df_pheno_vali = get_data(df_geno_train_vali,df_pheno_train_vali,0.25)

geno_dim = df_geno_train.shape[1]-2
latent_dim = 10
pheno_dim = df_pheno_train.shape[1]-2
seq_len = 5
extract_dim = 10
out_dim = 2
model = Net(geno_dim,latent_dim,pheno_dim,seq_len,extract_dim,out_dim,device).to(device)

epoch_num = 50
batch_size = 128
lr = 0.01
# train_model(device, df_geno_train, df_pheno_train, df_geno_vali, df_pheno_vali, model, seq_len, epoch_num, batch_size, lr)

model = Net(geno_dim,latent_dim,pheno_dim,seq_len,extract_dim,out_dim,device).to(device)
train_model(device, df_geno_train_vali, df_pheno_train_vali, df_geno_test, df_pheno_test, model, seq_len, epoch_num, batch_size, lr)

# 对比其他算法
df_train_vali = pd.merge(df_geno_train_vali,df_pheno_train_vali.loc[df_pheno_train_vali["times"]==0,:].drop(columns="times"),on="IID")
df_test = pd.merge(df_geno_test,df_pheno_test.loc[df_pheno_test["times"]==0,:].drop(columns="times"),on="IID")
df_train = pd.merge(df_geno_train,df_pheno_train.loc[df_pheno_train["times"]==0,:].drop(columns="times"),on="IID")
df_vali = pd.merge(df_geno_vali,df_pheno_vali.loc[df_pheno_vali["times"]==0,:].drop(columns="times"),on="IID")
X_train, y_train = np.array(df_train.drop(columns=["IID","label"])), np.array(df_train["label"])
X_vali, y_vali = np.array(df_vali.drop(columns=["IID","label"])), np.array(df_vali["label"])
X_train_vali, y_train_vali = np.array(df_train_vali.drop(columns=["IID","label"])), np.array(df_train_vali["label"])
X_test, y_test = np.array(df_test.drop(columns=["IID","label"])), np.array(df_test["label"])

from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import accuracy_score
parameters = {"penalty":["l1","l2"], "C":[0.01,0.1,1,10]}
gs = GridSearchCV(LogisticRegression(),param_grid=parameters,scoring="accuracy",cv=5)
gs.fit(X_train_vali,y_train_vali)
print("best parameters:", gs.best_params_)
print("best score:", gs.best_score_)
y_hat = gs.predict(X_train_vali)
print("performance in training sets",accuracy_score(y_train_vali,y_hat))
y_hat = gs.predict(X_test)
print("performance in testing sets",accuracy_score(y_test,y_hat))

from sklearn.svm import SVC
parameters = {"kernel":["linear", "poly", "rbf"], "C":[0.01,0.1,1,10]}
gs = GridSearchCV(SVC(),param_grid=parameters,scoring="accuracy",cv=5)
gs.fit(X_train_vali,y_train_vali)
print("best parameters:", gs.best_params_)
print("best score:", gs.best_score_)
y_hat = gs.predict(X_train_vali)
print("performance in training sets",accuracy_score(y_train_vali,y_hat))
y_hat = gs.predict(X_test)
print("performance in testing sets",accuracy_score(y_test,y_hat))

from sklearn.ensemble import RandomForestClassifier
parameters = {"n_estimators":[20,50,100], "criterion":["gini", "entropy"]}
gs = GridSearchCV(RandomForestClassifier(),param_grid=parameters,scoring="accuracy",cv=5)
gs.fit(X_train_vali,y_train_vali)
print("best parameters:", gs.best_params_)
print("best score:", gs.best_score_)
y_hat = gs.predict(X_train_vali)
print("performance in training sets",accuracy_score(y_train_vali,y_hat))
y_hat = gs.predict(X_test)
print("performance in testing sets",accuracy_score(y_test,y_hat))