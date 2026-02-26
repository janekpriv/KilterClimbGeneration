import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim


from data.dataset import get_data_loaders
from models.autoencoder import KilterAE

def main():
    db_path = r'../../data/raw/db.sqlite3'

    training_data_loader, val_data_loader = get_data_loaders(db_path)
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model = KilterAE(latent_dim=128).to(device)

    criterion = nn.BCELoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)

    epochs = 10

    for epochs in range(epochs):
        
        for idx, real_routes in enumerate(training_data_loader):
            pass

if __name__ == '__main__':
    main()