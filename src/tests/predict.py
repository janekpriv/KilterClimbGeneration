import torch
from pathlib import Path
from torch import tensor


from models.predictor import KilterGradePredictor
from data.dataset import get_data_loaders
from utils.grade_converter import Grade_converter


def predict(route_tensor: tensor):

    project_root = Path(__file__).resolve().parent.parent.parent
    models_dir = project_root / "models"
    ae_weights_path = models_dir / "kilter_ae_weights.pth"
    predictor_weights_path = models_dir / "kilter_predictor_weights.pth"


    device = torch.device("cpu")


    predictor_model = KilterGradePredictor(weights_path=str(ae_weights_path))
    predictor_model.load_state_dict(torch.load(predictor_weights_path, map_location=device, weights_only=True))
    predictor_model.eval()

    gc = Grade_converter()
    grades = gc.get_grade_dictionary()

    single_route = route_tensor[0].unsqueeze(0).to(device)

    with torch.no_grad():
        predicted_grade = predictor_model(single_route).item()

    v_predicted_grade = grades[round(predicted_grade)]

    return v_predicted_grade
    