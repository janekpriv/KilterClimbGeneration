from pathlib import Path
import torch
import matplotlib.pyplot as plt
import numpy as np

from data.dataset import get_data_loaders
from models.autoencoder import KilterAE

def main():

    db_path = r'../../data/raw/db.sqlite3'

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    project_root = Path(__file__).resolve().parent.parent.parent
    models_dir = project_root / "models"
    model_path = models_dir / "kilter_ae_weights.pth"

    model = KilterAE(latent_dim=128)
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.eval()

    _, test_loader = get_data_loaders(db_path, batch_size=32)

    batch, _ = next(iter(test_loader)) 
    single_route = batch[0].unsqueeze(0).to(device)

    #print(torch.nonzero(single_route, as_tuple=False))

    with torch.no_grad():
        reconstructed_route, _ = model(single_route)

    orig_tensor = single_route.squeeze().cpu()
    recon_tensor = reconstructed_route.squeeze().cpu()   

    #recon_tensor = torch.where(recon_tensor > 0.8, recon_tensor, torch.tensor(0.0))

    recon_tensor = (recon_tensor > 0.7).float()

    threshold = 0.7
    clean_recon = (recon_tensor > threshold).float()

    print(len(torch.nonzero(recon_tensor, as_tuple=False)))
 

    img_original = orig_tensor.sum(dim=0).numpy()
    img_reconstructed = clean_recon.sum(dim=0).numpy()

    diff_tensor = img_original - img_reconstructed  


    fig, axes = plt.subplots(1, 2, figsize=(12,6))

    axes[0].imshow(img_original, cmap='grey')
    axes[0].set_title("original_route")

    im = axes[1].imshow(img_reconstructed, cmap='viridis')
    axes[1].set_title("reconstructed_route")

    fig.colorbar(im, ax=axes[1], fraction=0.046, pad=0.04)

    height, width = img_original.shape
    
    # 2. Ustawiamy krok siatki (np. co 10 pikseli)
    step = 10 

    for ax in axes:

        ax.set_xticks(np.arange(0, width, step))
        ax.set_yticks(np.arange(0, height, step))
        
        ax.grid(True, color='white', linestyle='--', linewidth=0.5, alpha=0.3)

    plt.tight_layout()
    plt.show()

if __name__ == '__main__':
    main()