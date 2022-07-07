import torch
from torch import nn
from torch import optim
from model import Net
from loss_func import VAEloss, RNNloss, predloss
from plotpy import plot_vae, plot_pred, plot_rnn, plot_acc, plot_loss
from data_import import data_loader
import numpy as np


def train_model(device, train_geno, train_pheno, test_geno, test_pheno, model, seq_len, epoch_num, batch_size, lr):
    # load data
    train_data = data_loader(device, train_geno, train_pheno, batch_size, seq_len)
    
    optimizer_CDVAE = optim.Adam(model.CDVAE.parameters(), lr=lr)
    optimizer_RNN = optim.Adam(model.RNNextractor.parameters(), lr=lr)
    optimizer_class = optim.Adam(model.classifier.parameters(), lr=lr)
    optimizer = optim.Adam(model.parameters(), lr=lr)

    model.train()

    # stage I: pre train
    vae_epoch, BCE_epoch, KLD_epoch  = [], [], []
    for epoch in range(epoch_num):
        vae_batch, BCE_batch, KLD_batch = 0,0,0
        for _, (geno,pheno,label) in enumerate(train_data):
            geno_hat, mu, log_var, _ = model.CDVAE(geno)
            vae_loss, BCE, KLD = VAEloss(geno_hat, geno, mu, log_var)
            optimizer_CDVAE.zero_grad()  
            vae_loss.backward()  
            optimizer_CDVAE.step() 

            vae_batch+=vae_loss.item()
            BCE_batch+=BCE.item()
            KLD_batch+=KLD.item()
    
        vae_epoch.append(vae_batch/len(train_data))
        BCE_epoch.append(BCE_batch/len(train_data))
        KLD_epoch.append(KLD_batch/len(train_data))

        if (epoch + 1) % 10 == 0:
            print("epoch {}: BCE = {:.4f}, KLD = {:.4f}, loss for CDVAE = {:.4f}".format(
                epoch+1,BCE_epoch[-1],KLD_epoch[-1],vae_epoch[-1]))

    plot_vae(epoch_num,vae_epoch,BCE_epoch,KLD_epoch)

    MSE_epoch = []
    for epoch in range(epoch_num):
        MSE_batch = 0
        for _, (geno,pheno,label) in enumerate(train_data):
            pheno_pred, _ = model.RNNextractor(pheno)
            MSE = RNNloss(pheno_pred, pheno)
            optimizer_RNN.zero_grad()  
            MSE.backward()  
            optimizer_RNN.step() 

            MSE_batch+=MSE.item()
    
        MSE_epoch.append(MSE_batch/len(train_data))

        if (epoch + 1) % 10 == 0:
            print("epoch {}: MSE = {:.4f}".format(epoch+1,MSE_epoch[-1]))

    plot_rnn(epoch_num, MSE_epoch)


    # stage II: classification
    loss_epoch, vae_epoch, pred_epoch, BCE_epoch, KLD_epoch, MSE_epoch, acc_epoch  = [], [], [], [], [], [], []
    for epoch in range(epoch_num):
        # pred_batch, acc_batch, size = 0,0,0
        loss_batch, vae_batch, pred_batch, BCE_batch, KLD_batch, MSE_batch, acc_batch, size = 0,0,0,0,0,0,0,0
        for _, (geno,pheno,label) in enumerate(train_data):
            geno_hat, mu, log_var, pheno_pred, pred = model(geno,pheno)
            vae_loss, BCE, KLD = VAEloss(geno_hat, geno, mu, log_var)
            MSE = RNNloss(pheno_pred, pheno)
            pred_loss = predloss(pred,label)
            loss = vae_loss + MSE + pred_loss
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
            pred_label = torch.argmax(pred,1).cpu().numpy()
            acc_batch += np.sum(pred_label==label.cpu().numpy())
            size += len(pred_label)
            loss_batch+=loss.item()
            pred_batch+=pred_loss.item()
            vae_batch+=vae_loss.item()
            BCE_batch+=BCE.item()
            KLD_batch+=KLD.item()
            MSE_batch+=MSE.item()

        acc_epoch.append(acc_batch/size)
        loss_epoch.append(loss_batch/len(train_data))
        pred_epoch.append(pred_batch/len(train_data))
        vae_epoch.append(vae_batch/len(train_data))
        BCE_epoch.append(BCE_batch/len(train_data))
        KLD_epoch.append(KLD_batch/len(train_data))
        MSE_epoch.append(MSE_batch/len(train_data))

        if (epoch + 1) % 10 == 0:
            print("epoch {}: BCE = {:.4f}, KLD = {:.4f}, loss for CDVAE = {:.4f}, loss for RNN = {:.4f}, loss for classification = {:.4f}, total loss = {:.4f}, accuracy = {:.4f}".format(
                epoch+1,BCE_epoch[-1],KLD_epoch[-1],vae_epoch[-1],MSE_epoch[-1],pred_epoch[-1],loss_epoch[-1],acc_epoch[-1]))
    
    plot_loss(epoch_num,loss_epoch,vae_epoch,pred_epoch,BCE_epoch,KLD_epoch,MSE_epoch)


    # stage III
    pred_epoch = []
    for epoch in range(epoch_num):
        pred_batch, acc_batch, size = 0,0,0
        for _, (geno,pheno,label) in enumerate(train_data):
            _, _, _, _, pred = model(geno,pheno)
            pred_loss = predloss(pred,label)
            optimizer_class.zero_grad() 
            pred_loss.backward()  
            optimizer_class.step() 

            pred_label = torch.argmax(pred,1).cpu().numpy()
            acc_batch += np.sum(pred_label==label.cpu().numpy())
            size += len(pred_label)
            pred_batch+=pred_loss.item()
        acc_epoch.append(acc_batch/size)
        pred_epoch.append(pred_batch/len(train_data))
        if (epoch + 1) % 10 == 0:
            print("epoch {}: loss for classification = {:.4f}, accuracy = {:.4f}".format(
                epoch+1,pred_epoch[-1],acc_epoch[-1]))
    plot_pred(epoch_num,pred_epoch)

    plot_acc(2*epoch_num, acc_epoch)

    test_model(device, test_geno, test_pheno, model, seq_len, batch_size)


def test_model(device, geno, pheno, model, seq_len, batch_size):
    model.eval()
    test_data = data_loader(device, geno, pheno, batch_size, seq_len)
    acc_batch,size = 0,0
    for _, (geno,pheno,label) in enumerate(test_data):
        _, _, _, _, pred = model(geno,pheno)
        pred_label = torch.argmax(pred,1).cpu().numpy()
        acc_batch += np.sum(pred_label==label.cpu().numpy()) 
        size += len(pred_label)
    acc = acc_batch/size
    print("test accuracy = {:.4f}".format(acc))









