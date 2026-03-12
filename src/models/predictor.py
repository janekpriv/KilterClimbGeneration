import torch 
import torch.nn as nn
from models.autoencoder import KilterAE

class KilterGradePredictor(nn.Module):
    def __init__(self, weights_path, latent_dim=256):
        super.__init__()

        autoencoder = KilterAE(latent_dim=latent_dim)

        autoencoder.load_state_dict(torch.load(weights_path, weights_only=True))

        self.encoder = autoencoder.encoder

        for param in self.encoder.parameters():
            param.requires_grad = False

        self.grade_head = nn.Sequential(
            nn.Linear(latent_dim, 128),
            nn.ReLU(),
            nn.Dropout(0.2), 
            nn.Linear(128, 32),
            nn.ReLU(),
            nn.Linear(32, 1)
        )
    
    def forward(self, x):
       
        with torch.no_grad():
            latent_vector = self.encoder(x)      
        
        predicted_grade = self.grade_head(latent_vector)
        
        return predicted_grade