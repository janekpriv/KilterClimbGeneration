import torch
from pydantic import BaseModel

class Hold(BaseModel):
    x: int
    y: int
    channel: int

class Hold_list(BaseModel):
    holds: list[Hold]


def convert_to_tensor(list:Hold_list):
    tensor = torch.zeros(1, 4, 173, 185, dtype=torch.float32)
    for h in list:
        tx = h.x + 20
        ty = h.y - 4    
        tensor[0, h.channel, ty, tx] = 1.0

    return tensor