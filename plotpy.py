import matplotlib.pyplot as plt

def plot_loss(epoch_num,loss,vae_loss,pred_loss,BCE,KLD,MSE):
    plt.plot(list(range(epoch_num)),loss,label="total loss")
    plt.plot(list(range(epoch_num)),vae_loss,label="loss for CDVAE")
    plt.plot(list(range(epoch_num)),BCE,label="BCE loss for CDVAE")
    plt.plot(list(range(epoch_num)),KLD,label="KLD loss for CDVAE")
    plt.plot(list(range(epoch_num)),MSE,label="loss for RNN")
    plt.plot(list(range(epoch_num)),pred_loss,label="loss for classification")
    plt.legend()
    plt.title("training loss in stage II")
    plt.xlabel('epoch')
    plt.ylabel('loss')
    plt.grid()
    plt.show()

def plot_vae(epoch_num,vae_loss,BCE,KLD):
    plt.plot(list(range(epoch_num)),vae_loss,label="loss for CDVAE")
    plt.plot(list(range(epoch_num)),BCE,label="BCE loss for CDVAE")
    plt.plot(list(range(epoch_num)),KLD,label="KLD loss for CDVAE")
    plt.legend()
    plt.title("CDVAE loss in stage I")
    plt.xlabel('epoch')
    plt.ylabel('loss')
    plt.grid()
    plt.show()

def plot_rnn(epoch_num,MSE):
    plt.plot(list(range(epoch_num)),MSE,label="loss for RNN")
    plt.legend()
    plt.title("RNN loss in stage I")
    plt.xlabel('epoch')
    plt.ylabel('loss')
    plt.grid()
    plt.show()

def plot_pred(epoch_num,pred_loss):
    plt.plot(list(range(epoch_num)),pred_loss,label="loss for classification")
    plt.legend()
    plt.title("training loss in stage III")
    plt.xlabel('epoch')
    plt.ylabel('loss')
    plt.grid()
    plt.show()

def plot_acc(epoch_num,acc):
    plt.plot(list(range(epoch_num)),acc,label="training accuracy")
    plt.legend()
    plt.title("training accuracy in stage II & III")
    plt.xlabel('epoch')
    plt.ylabel('accuracy')
    plt.grid()
    plt.show()
