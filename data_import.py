import torch
import numpy as np
import pandas as pd
import torch.utils.data as Data
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import torch.nn.functional as F

def get_data(df_geno,df_pheno,test_size):   
    scaler=StandardScaler().fit(df_pheno.iloc[:,2:])
    df_pheno.iloc[:,2:] = scaler.transform(df_pheno.iloc[:,2:])  
    df_geno_train, df_geno_test = train_test_split(df_geno,test_size=test_size,stratify=df_geno["label"])
    df_geno_train, df_geno_test = df_geno_train.sort_values(by="IID"), df_geno_test.sort_values(by="IID")
    df_pheno_train = df_pheno[df_pheno["IID"].isin(df_geno_train["IID"])].sort_values(by=["IID","times"])
    df_pheno_test = df_pheno[df_pheno["IID"].isin(df_geno_test["IID"])].sort_values(by=["IID","times"])
    return df_geno_train, df_geno_test, df_pheno_train,df_pheno_test

def data_loader(device, geno, pheno, batch_size, seq_len):
    X_pheno = np.array(pheno.drop(columns=["IID","times"]))
    X_pheno = X_pheno.reshape(X_pheno.shape[0]//seq_len,seq_len,X_pheno.shape[1])
    X_geno = np.array(geno.drop(columns=["IID","label"]))
    y = np.array(geno["label"])
    X_pheno = torch.tensor(X_pheno,requires_grad=True).float().to(device)
    X_geno = torch.tensor(X_geno).to(device)
    X_geno = F.one_hot(X_geno, num_classes=3)
    X_geno = X_geno.transpose(1,2).float().requires_grad_()
    y = torch.tensor(y).to(torch.long).to(device)
    dataset = Data.TensorDataset(X_geno,X_pheno,y)   
    loader = Data.DataLoader(dataset,batch_size,shuffle=True)
    return loader
