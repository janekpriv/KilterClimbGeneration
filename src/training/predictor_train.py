import torch
import torch.nn as nn
import torch.optim as optim
import time
from pathlib import Path

from data.dataset import get_data_loaders 
from models.predictor import KilterGradePredictor

def main():

    project_root = Path(__file__).resolve().parent.parent.parent
    db_path = project_root / "data" / "raw" / "db.sqlite3"
    models_dir = project_root / "models"
    ae_weights_path = models_dir / "kilter_ae_weights.pth"
    predictor_save_path = models_dir / "kilter_predictor_weights.pth"


    device = torch.device("cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu")


    training_data_loader, val_data_loader = get_data_loaders(str(db_path))


    model = KilterGradePredictor(weights_path=str(ae_weights_path), latent_dim=128).to(device)


    criterion = nn.MSELoss()
    

    optimizer = optim.Adam(model.grade_head.parameters(), lr=0.001)

    EPOCHS = 20

    for epoch in range(EPOCHS):
        start_time = time.time

        model.train()
        train_loss = 0.0
        train_mae = 0.0 
        

        for idx, (routes, grades) in enumerate(training_data_loader):
            optimizer.zero_grad()

            routes = routes.to(torch.float32).to(device)

            grades = grades.to(torch.float32).to(device)

            predictions = model(routes)


            loss = criterion(predictions, grades)
            loss.backward()
            optimizer.step()

            train_loss += loss.item()

            train_mae += torch.abs(predictions - grades).mean().item()

        avg_train_loss = train_loss / len(training_data_loader)
        avg_train_mae = train_mae / len(training_data_loader)
        



        model.eval()
        val_loss = 0.0
        val_mae = 0.0
        
        with torch.no_grad():
            for routes, grades in val_data_loader:
                routes = routes.to(torch.float32).to(device)
                grades = grades.to(torch.float32).to(device)
                
                predictions = model(routes)
                loss = criterion(predictions, grades)

                val_loss += loss.item()
                val_mae += torch.abs(predictions - grades).mean().item()
                
        avg_val_loss = val_loss / len(val_data_loader)
        avg_val_mae = val_mae / len(val_data_loader)
        

        

        print(f"Epoch: [{epoch+1}/{EPOCHS}]")
        print(f"  Train   -> MSE Loss: {avg_train_loss:.4f} | {avg_train_mae:.2f} pkt")
        print(f"  Validate -> MSE Loss: {avg_val_loss:.4f} | {avg_val_mae:.2f} pkt")
        print("-" * 60)


    torch.save(model.state_dict(), predictor_save_path)

if __name__ == '__main__':
    main()