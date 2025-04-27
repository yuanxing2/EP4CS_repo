from torch import nn
import torch
import torch.nn.functional as F


class VAE(nn.Module):

    def __init__(self, input_dim=784, h_dim=400, z_dim=20):
        
        super(VAE, self).__init__()

        self.input_dim = input_dim
        self.h_dim = h_dim
        self.z_dim = z_dim

        self.fc1 = nn.Linear(input_dim, h_dim)  
        self.fc2 = nn.Linear(h_dim, z_dim)  
        self.fc3 = nn.Linear(h_dim, z_dim)  

        self.fc4 = nn.Linear(z_dim, h_dim)
        self.fc5 = nn.Linear(h_dim, input_dim)

    def forward(self, x):

        batch_size = x.shape[0] 
        x = x.contiguous()
        x = x.view(batch_size, self.input_dim)  
        mu, log_var = self.encode(x)
        sampled_z = self.reparameterization(mu, log_var)
        x_hat = self.decode(sampled_z)

        x_hat = x_hat.view(batch_size, 64, 768)
        return x_hat, mu, log_var

    def encode(self, x):
        """
        encoding part
        :param x: input image
        :return: mu and log_var
        """
        h = F.relu(self.fc1(x))
        mu = self.fc2(h)
        log_var = self.fc3(h)

        return mu, log_var

    def reparameterization(self, mu, log_var):
        """
        Given a standard gaussian distribution epsilon ~ N(0,1),
        we can sample the random variable z as per z = mu + sigma * epsilon
        :param mu:
        :param log_var:
        :return: sampled z
        """
        sigma = torch.exp(log_var * 0.5)
        eps = torch.randn_like(sigma)
        return mu + sigma * eps  

    def decode(self, z):
        """
        Given a sampled z, decode it back to image
        :param z:
        :return:
        """
        h = F.relu(self.fc4(z))
        x_hat = torch.sigmoid(self.fc5(h))  
        return x_hat
