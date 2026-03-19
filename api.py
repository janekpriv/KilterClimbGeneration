import torch


from contextlib import asynccontextmanager
from fastapi import FastAPI, Request
from pydantic import BaseModel



from src.utils.Converter import Converter
from src.tests.predict import load_model




@asynccontextmanager
async def lifespan(app: FastAPI):
    model = load_model
    convert_to_tensor = Converter.convert_to_tensor
    grade_dict = Converter.get_grade_dictionary()

    device = torch.device("cpu")

    app.state.model = model
    app.state.grade_dict = grade_dict
    app.state.convert_to_tensor = convert_to_tensor
    yield

    print("model and utils ready for launch.")




app = FastAPI(lifespan=lifespan)

class Hold(BaseModel):
    x: int
    y: int
    channel: int

class PredictRequest(BaseModel):
    route: list[Hold]

@app.post("/predict")
def predict(body: PredictRequest, request: Request):
    model = request.app.state.model
    convert_to_tensor = request.app.state.convert_to_tensor
    grade_dict = request.app.state.grade_dict
    device = request.app.state.device

    tensor = convert_to_tensor(body)

    single_route = tensor.unsqueeze(0).to(device)

    with torch.no_grad():
        predicted_grade = model(single_route).item()
    
    grade = grade_dict[round(predicted_grade)]

    return {"grade": grade}






