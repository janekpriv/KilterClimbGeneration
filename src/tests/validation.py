import torch
import torch.nn.functional as F
from pathlib import Path

from data.dataset import get_data_loaders
from models.autoencoder import KilterAE

def evaluate_model(model, test_loader, device, threshold=0.75):

    model.eval()

    total_routes = 0
    perfect_matches = 0

    total_tp = 0 #
    total_fp = 0 
    total_fn = 0 


    with torch.no_grad():
        for targets in test_loader:
            targets = targets.to(torch.float32).to(device)
            preds, _ = model(targets)


            pooled = F.max_pool2d(preds, kernel_size=3, stride=1, padding=1)
            is_peak = (preds == pooled)
            clean_preds = (is_peak & (preds > threshold)).float()


            for i in range(targets.size(0)):
                pred_route = clean_preds[i]
                target_route = targets[i]

                tp = torch.sum((pred_route == 1) & (target_route == 1)).item()
                fp = torch.sum((pred_route == 1) & (target_route == 0)).item()
                fn = torch.sum((pred_route == 0) & (target_route == 1)).item()

                total_tp += tp
                total_fp += fp
                total_fn += fn
                total_routes += 1

                if fp == 0 and fn == 0:
                    perfect_matches += 1


    precision = total_tp / (total_tp + total_fp) if (total_tp + total_fp) > 0 else 0
    recall = total_tp / (total_tp + total_fn) if (total_tp + total_fn) > 0 else 0
    exact_match_ratio = perfect_matches / total_routes if total_routes > 0 else 0

    print("\n" + "="*30)
    print("Results")
    print("="*30)
    print(f"Routes Tested: {total_routes}")
    print(f"Perfectly reconstructed routes (1:1): {perfect_matches} ({exact_match_ratio*100:.2f}%)")
    print("-" * 30)
    print(f"Sum of real holds: {total_tp + total_fn}")
    print(f"correctly highlited holds: {total_tp}")
    print(f"false positives: {total_fp}")
    print(f"false negatives: {total_fn}")
    print("-" * 30)
    print(f"Precision: {precision*100:.2f}")
    print(f"Recall:     {recall*100:.2f}")
    print("="*30)


def main():
    db_path = r'../../data/raw/db.sqlite3'
    device = torch.device("cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu")


    _, test_loader = get_data_loaders(db_path, batch_size=32)


    project_root = Path(__file__).resolve().parent.parent.parent
    models_dir = project_root / "models"
    model_path = models_dir / "kilter_ae_weights.pth"

    model = KilterAE(latent_dim=128).to(device)
    model.load_state_dict(torch.load(model_path, map_location=device))

    # Odpalamy twardy test!
    evaluate_model(model, test_loader, device, threshold=0.5)

if __name__ == '__main__':
    main()