import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import time
from pathlib import Path

from data.dataset import get_data_loaders
from models.autoencoder import KilterAE


def focal_loss(preds, targets, alpha_pos=50.0, alpha_neg=1.0, gamma=2.0):

    preds = torch.clamp(preds, min=1e-7, max=1-1e-7)
    

    bce = F.binary_cross_entropy(preds, targets, reduction='none')
    

    pt = torch.exp(-bce)
    
    alpha_tensor = torch.where(targets == 1.0, alpha_pos, alpha_neg)
    
    focal_weight = alpha_tensor * (1 - pt)**gamma
    
    return (focal_weight * bce).mean()

def main():
    db_path = r'../../data/raw/db.sqlite3'

    training_data_loader, val_data_loader = get_data_loaders(db_path)
    device = torch.device("cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu")
    print(device)
    real_routes = enumerate(val_data_loader)
    model = KilterAE(latent_dim=256).to(device)

    criterion = nn.BCELoss()
    optimizer = optim.Adam(model.parameters(), lr=0.0005)

    EPOCHS = 20

    for epoch in range(EPOCHS):

        start_time = time.time()
        
        end_time = time.time()
        epoch_mins = (end_time - start_time) / 60

        model.train()
        train_loss=0.0
        
        for idx, (real_routes, grades) in enumerate(training_data_loader):
            optimizer.zero_grad()


            real_routes = real_routes.to(torch.float32).to(device)
            reconstructed_routes, _ = model(real_routes)


            loss = focal_loss(reconstructed_routes, real_routes)

            loss.backward()
            optimizer.step()

            train_loss += loss.item()

        avg_train_loss = train_loss/len(training_data_loader)
        
        model.eval()
        val_loss = 0.0
        with torch.no_grad():
            for test_routes, _ in val_data_loader:
                
                test_routes = test_routes.to(torch.float32).to(device)
                
                reconstructed_routes, _ = model(test_routes)
                
                loss = focal_loss(reconstructed_routes, test_routes)

                val_loss += loss.item()
                
            avg_val_loss = val_loss/len(val_data_loader)
        
        print(f"epoch: [{epoch+1}/{EPOCHS}] | Train Loss: {avg_train_loss:.4f} | Val Loss: {avg_val_loss:.4f} | EpochTime: {epoch_mins}")

    project_root = Path(__file__).resolve().parent.parent.parent
    models_dir = project_root / "models"
    save_path = models_dir / "kilter_ae_weights.pth"

    torch.save(model.state_dict(), save_path)

if __name__ == '__main__':
    main()