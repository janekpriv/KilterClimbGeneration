import torch
import torch.nn as nn
import torch.nn.functional as F
import matplotlib.pyplot as plt


class KilterAE(nn.Module):
    def __init__(self, latent_dim=128):
        super(KilterAE, self).__init__()

        self.encoder = KilterEncoder(latent_dim=128)
        self.decoder = KilterDecoder(latent_dim=128)

    
    def forward(self, x):

        latent_space = self.encoder(x)

        reconstruction = self.decoder(latent_space)

        return reconstruction, latent_space

class KilterEncoder(nn.Module):
    def __init__(self, latent_dim = 128):
        super(KilterEncoder, self).__init__()

        self.feature_extractor = nn.Sequential(
            
            nn.Conv2d(in_channels=4, out_channels=32, kernel_size=4, stride=2, padding=1),
            nn.LeakyReLU(0.3),

            nn.Conv2d(in_channels=32, out_channels=64, kernel_size=4, stride=2, padding=1),
            nn.LeakyReLU(0.3),

            nn.Conv2d(in_channels=64, out_channels=128, kernel_size=4, stride=2, padding=1),
            nn.LeakyReLU(0.3),

            nn.Conv2d(in_channels=128, out_channels=256, kernel_size=4, stride=2, padding=1),
            nn.LeakyReLU(0.3),

            nn.Flatten()
        )

        self.flattened_size = 256*10*11

        self.fc = nn.Linear(self.flattened_size, latent_dim)

    def forward(self, x):

        features = self.feature_extractor(x)

        latent_vector = self.fc(features)

        return latent_vector

class KilterDecoder(nn.Module):
    def __init__(self, latent_dim=128):
        super(KilterDecoder, self).__init__()
        

        self.flattened_size = 256 * 10 * 11
        

        self.fc = nn.Linear(latent_dim, self.flattened_size)
        

        self.decoder_cnn = nn.Sequential(

            nn.Unflatten(1, (256, 10, 11)),

            nn.ConvTranspose2d(in_channels=256, out_channels=128, kernel_size=4, stride=2, padding=1),
            nn.ReLU(),
            

            nn.ConvTranspose2d(in_channels=128, out_channels=64, kernel_size=4, stride=2, padding=1),
            nn.ReLU(),
            
            nn.ConvTranspose2d(in_channels=64, out_channels=32, kernel_size=4, stride=2, padding=1),
            nn.ReLU(),
            
            nn.ConvTranspose2d(in_channels=32, out_channels=4, kernel_size=4, stride=2, padding=1),
            
            
            nn.Sigmoid() 
        )

    def forward(self, x):
       
        x = self.fc(x)
        
       
        x = self.decoder_cnn(x)
        
    
        if x.shape[-2:] != (173, 185):
            x = F.interpolate(x, size=(173, 185), mode='bilinear', align_corners=False)
            
        return x