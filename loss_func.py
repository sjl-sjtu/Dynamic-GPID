import torch
from torch import nn

def VAEloss(x_hat, x, mu, log_var):
    BCEloss = nn.BCELoss()
    BCE = BCEloss(x_hat, x) # the reconstruction loss.   
    KLD = 0.5 * torch.sum(torch.exp(log_var) + torch.pow(mu, 2) - 1. - log_var) # KL-divergence  
    loss = BCE + KLD # total loss
    return loss, BCE, KLD

def RNNloss(pred,input):
    MSEloss = nn.MSELoss()
    MSE = MSEloss(pred,input)
    return MSE

def predloss(pred,label):
    loss_func = nn.CrossEntropyLoss()
    loss = loss_func(pred,label)
    return loss


