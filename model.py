import torch
from torch import nn
import torch.nn.functional as F

class CDVAE(nn.Module):
    def __init__(self, input_dim=3241, z_dim=30):
        super(CDVAE, self).__init__()
        self.h_dim = ((input_dim-1)//24-1)*6
        self.z_dim = z_dim
        #encoder: [b, input_dim] => [b, z_dim]
        self.convEncoder = nn.Sequential(
            nn.Conv1d(3, 6, kernel_size=5, stride=2, padding=1),  # in => 0.5(in-5+2)+1 = 0.5in-0.5  3241 -> 1620
            nn.ReLU(),
            nn.MaxPool1d(3,stride=3), # 0.5in-0.5 => (0.5in-0.5-3)/3+1 = (in-1)/6  1620 -> 540
            nn.Conv1d(6, 6, kernel_size=6, stride=2, padding=1),  # (in-1)/6 => 0.5[(in-1)/6—6+2]+1 = (in-1)/12-1  540 -> 269
            nn.ReLU(),
            nn.MaxPool1d(3, stride=2),  # (in-1)/12-1 => 0.5[(in-1)/12-1-3]+1 = (in-1)/24-1  269 -> 134
            # nn.AdaptiveAvgPool2d((1,1)),
            nn.Flatten()
        )
        self.mean = nn.Linear(self.h_dim, z_dim)  # mu
        self.logvar = nn.Linear(self.h_dim, z_dim)  # log_var
        #decoder: [b, z_dim] => [b, input_dim]
        self.linear = nn.Linear(z_dim, self.h_dim)
        self.decoder = nn.Sequential(
            nn.ConvTranspose1d(6, 6, kernel_size=3, stride=2),
            nn.ReLU(),
            nn.ConvTranspose1d(6, 6, kernel_size=6,stride=2, padding=1),
            nn.ReLU(),
            nn.ConvTranspose1d(6, 6, kernel_size=3, stride=3),
            nn.ReLU(),
            nn.ConvTranspose1d(6, 3, kernel_size=5,stride=2, padding=1)
        )

    def forward(self, x): 
        x = self.add_noise(x)
        mu, log_var = self.encode(x)  # encoder
        sampled_z = self.reparameterization(mu, log_var)  # reparameterization trick    
        x_hat = self.decode(sampled_z)  # decoder
        return x_hat, mu, log_var, sampled_z

    def encode(self, x):
        h = self.convEncoder(x)  
        mu = self.mean(h)
        log_var = self.logvar(h)
        return mu, log_var

    def reparameterization(self, mu, log_var):
        sigma = torch.exp(log_var * 0.5)
        eps = torch.randn_like(sigma)
        return mu + sigma * eps

    def decode(self, z):
        x = F.relu(self.linear(z))
        x = x.view(-1,6,self.h_dim//6)
        x_hat = torch.sigmoid(self.decoder(x))
        return x_hat
    
    def add_noise(self, inputs, noise_factor=0.2):
        noisy = inputs+torch.randn_like(inputs) * noise_factor
        noisy = torch.clip(noisy,0.,1.)
        return noisy


class RNNextractor(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers, num_out, seq_len, device):
        super(RNNextractor, self).__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.gru = nn.GRU(input_size, hidden_size, num_layers, batch_first=True)
        self.linear = nn.Linear(hidden_size*num_layers, num_out)
        self.device = device
    
    def forward(self, x):
        batch_size = x.shape[0]
        h0 = torch.zeros(self.num_layers, batch_size, self.hidden_size).to(self.device)
        # c0 = torch.zeros(self.num_layers, batch_size, self.hidden_size).to(self.device)
        y, h = self.gru(x, h0)
        h = h.transpose(0,1)
        h = h.reshape(h.shape[0], -1)
        h = self.linear(h)
        return y, h
        

class Classifier(nn.Module):
    def __init__(self, input_size, hidden_size1, hidden_size2, output_size):
        super(Classifier, self).__init__()
        self.classify = nn.Sequential(
            nn.Linear(input_size,hidden_size1),
            nn.ReLU(),
            nn.BatchNorm1d(hidden_size1),
            nn.Dropout(p=0.5),
            nn.Linear(hidden_size1,hidden_size2),
            nn.ReLU(),
            nn.BatchNorm1d(hidden_size2),
            nn.Dropout(p=0.5),
            nn.Linear(hidden_size2,output_size)
        )
    
    def forward(self, x):
        y = self.classify(x)
        return y


class Net(nn.Module):
    def __init__(self, geno_dim, latent_dim, pheno_dim, seq_len, extract_dim, out_dim, device):
        super(Net, self).__init__()
        self.CDVAE = CDVAE(geno_dim,latent_dim)
        self.RNNextractor = RNNextractor(pheno_dim,pheno_dim,3,extract_dim,seq_len,device)
        self.classifier = Classifier(latent_dim+extract_dim+pheno_dim,20,10,out_dim)
    
    def forward(self,geno,pheno):
        geno_hat, mu, log_var, geno_latent = self.CDVAE(geno)
        pheno_pred, pheno_extract = self.RNNextractor(pheno)
        feature = torch.cat((geno_latent,pheno_extract),dim=1)
        # baseline = pheno[:,0,:].view(pheno.shape[0],-1)
        # feature = torch.cat((feature,baseline),dim=1)
        pred = self.classifier(feature)
        return geno_hat, mu, log_var, pheno_pred, pred
