from pathlib import Path
import torch
import matplotlib.pyplot as plt
import numpy as np
import math

from data.dataset import get_data_loaders
from models.autoencoder import KilterAE
from models.predictor import KilterGradePredictor
from utils.grade_converter import Grade_converter

def main():

    db_path = r'../../data/raw/db.sqlite3'

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    project_root = Path(__file__).resolve().parent.parent.parent
    models_dir = project_root / "models"
    model_path = models_dir / "kilter_ae_weights.pth"
    predictor_weights_path = models_dir / "kilter_predictor_weights.pth"

    model = KilterAE(latent_dim=128)
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.eval()

    gc = Grade_converter()
    grades = gc.get_grade_dictionary()

    predictor_model = KilterGradePredictor(weights_path=str(model_path), latent_dim=128).to(device)
    predictor_model.load_state_dict(torch.load(predictor_weights_path, map_location=device, weights_only=True))
    predictor_model.eval()

    _, test_loader = get_data_loaders(db_path, batch_size=32)

    batch_routes, batch_grades = next(iter(test_loader)) 
    single_route = batch_routes[0].unsqueeze(0).to(device)
    real_grade = batch_grades[0].item()

    #print(torch.nonzero(single_route, as_tuple=False))

    with torch.no_grad():
        reconstructed_route, _ = model(single_route)
        predicted_grade = predictor_model(single_route).item()

    v_real_grade = grades[math.floor(real_grade)]
    v_predicted_grade = grades[math.floor(predicted_grade)]

    orig_tensor = single_route.squeeze().cpu()
    recon_tensor = reconstructed_route.squeeze().cpu()   

    #recon_tensor = torch.where(recon_tensor > 0.8, recon_tensor, torch.tensor(0.0))

    threshold = 0.7
    clean_recon = (recon_tensor > threshold).float()

    img_original = orig_tensor.sum(dim=0).numpy()
    img_reconstructed = clean_recon.sum(dim=0).numpy()


    fig, axes = plt.subplots(1, 2, figsize=(12, 7)) # Lekko powiększyłem wysokość na tytuł


    error = abs(real_grade - predicted_grade)
    fig.suptitle(f"Kilter DB: {v_real_grade} pt   |   Model: {v_predicted_grade} pt\n(ERROR: {error:.1f} pt)", 
                 fontsize=16, fontweight='bold', color='darkred' if error > 2.0 else 'darkgreen')

    axes[0].imshow(img_original, cmap='grey')
    axes[0].set_title("Original Route")

    im = axes[1].imshow(img_reconstructed, cmap='viridis')
    axes[1].set_title("model reconstruction")

    fig.colorbar(im, ax=axes[1], fraction=0.046, pad=0.04)

    height, width = img_original.shape
    step = 10 

    for ax in axes:
        ax.set_xticks(np.arange(0, width, step))
        ax.set_yticks(np.arange(0, height, step))
        ax.grid(True, color='white', linestyle='--', linewidth=0.5, alpha=0.3)

    plt.tight_layout()
    plt.show()

if __name__ == '__main__':
    main()