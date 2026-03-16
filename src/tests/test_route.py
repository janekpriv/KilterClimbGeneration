import torch
import sqlite3
from pathlib import Path
import csv 


from models.predictor import KilterGradePredictor
from data.dataset import get_data_loaders
from utils.grade_converter import Grade_converter


def main():

    project_root = Path(__file__).resolve().parent.parent.parent
    db_path = project_root / "data" / "raw" / "db.sqlite3"
    models_dir = project_root / "models"
    ae_weights_path = models_dir / "kilter_ae_weights.pth"
    predictor_weights_path = models_dir / "kilter_predictor_weights.pth"
    output_csv = project_root / "data" / "csv" / "predictions.csv"

    device = torch.device("cpu")


    _, val_data_loader = get_data_loaders(db_path)
    model = KilterGradePredictor(weights_path=str(ae_weights_path))
    model.load_state_dict(torch.load(predictor_weights_path, map_location=device, weights_only=True))
    model.eval()


    gc = Grade_converter()
    grades_dict = gc.get_grade_dictionary()


    total_routes = 0
    total_mae = 0.0
    total_mse = 0.0
    
    exact_matches = 0       
    within_one_grade = 0    
    max_error = 0.0

    csv_data = []

    print("="*60)
    print(f"{'ROUTE NAME':<30} | {'KILTER DB':<10} | {'AI MODEL':<10}")
    print("="*60)


    with torch.no_grad():
        for routes, grades in val_data_loader:
            
            routes = routes.to(torch.float32).to(device)
            grades = grades.to(torch.float32).to(device)
                

            preds = model(routes)
            

            errors = torch.abs(preds - grades)
            squared_errors = (preds - grades) ** 2
            

            total_routes += routes.size(0)
            total_mae += torch.sum(errors).item()
            total_mse += torch.sum(squared_errors).item()
            

            current_max_error = torch.max(errors).item()
            if current_max_error > max_error:
                max_error = current_max_error
                

            exact_matches += torch.sum(errors <= 0.5).item()
            within_one_grade += torch.sum(errors <= 1.6).item()


            for i in range(routes.size(0)):
                real = grades[i].item()
                predicted = preds[i].item()
                error = abs(real - predicted)
                
                csv_data.append({
                    "Real_Grade": round(real, 2),
                    "Model_Prediction": round(predicted, 2),
                    "Real_Grade(V)": grades_dict[round(real)],
                    "Model prediction(V)": grades_dict[round(predicted)],
                    "Error": round(error, 2)
                })

        avg_mae = total_mae / total_routes if total_routes > 0 else 0
        avg_mse = total_mse / total_routes if total_routes > 0 else 0
        
        exact_match_ratio = exact_matches / total_routes if total_routes > 0 else 0
        within_grade_ratio = within_one_grade / total_routes if total_routes > 0 else 0


        with open(output_csv, mode='w', newline='', encoding='utf-8') as file:
            writer = csv.DictWriter(file, fieldnames=["Real_Grade", "Model_Prediction", "Real_Grade(V)","Model prediction(V)", "Error"])
            writer.writeheader()
            writer.writerows(csv_data)

        print("\n" + "="*40)
        print("Grade Predictor Results")
        print("="*40)
        print(f"Routes Tested: {total_routes}")
        print(f"Perfectly predicted (error < 0.5 pkt): {exact_matches} ({exact_match_ratio*100:.2f}%)")
        print(f"Predicted within 1 V-Grade (< 1.6 pkt): {within_one_grade} ({within_grade_ratio*100:.2f}%)")
        print("-" * 40)
        print(f"Mean Absolute Error (MAE): {avg_mae:.2f} pkt")
        print(f"Mean Squared Error (MSE):  {avg_mse:.2f}")
        print(f"Max Error (Biggest miss):  {max_error:.2f} pkt")
        print("="*40)


if __name__ == '__main__':
    main()