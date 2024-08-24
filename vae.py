import torch
import torch.nn as nn
import torch.nn.functional as F


"""
The Whole WorkFlow
Input Image -> Hidden Dim -> Mean, Std_dev -> Parametrization trick -> Decoder -> Output Image

"""

class VariationalAutoEncoder(nn.Module):
    def __init__(self, input_dim, h_dim=200, z_dim=20):
        super().__init__()

        # encoder
        # encode the input image into a hidden dimension
        # extract the mean and standard deviation from that hidden space 
        # call it the z space 
        self.img_to_hid = nn.Linear(input_dim, h_dim)
        self.hid_to_mean = nn.Linear(h_dim, z_dim)
        self.hid_to_std = nn.Linear(h_dim, z_dim)

        # decoder
        # extrapolate the hidden state from the z space
        # further extrapolate the generated image from the hidden state.
        self.z_to_hid = nn.Linear(z_dim, h_dim)
        self.hid_to_img = nn.Linear(h_dim, input_dim)

        self.relu = nn.ReLU()


    def encode(self, x):
        # q_phi(z|x)
        
        h = self.relu(self.img_to_hid(x))
        mu, sigma = self.hid_to_mean(h), self.hid_to_std(h)
        return mu, sigma

    def decode(self, z):
        h = self.relu(self.z_to_hid(z))
        return torch.sigmoid(self.hid_to_img(h))

    def forward(self, x):
        # p_theta(x|z)
        
        mu, sigma = self.encode(x)
        epsilon = torch.rand_like(sigma)
        z_reparametrized = mu + sigma*epsilon
        x_reconstructed = self.decode(z_reparametrized)

        return x_reconstructed, mu, sigma
    

