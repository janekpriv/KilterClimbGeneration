import sqlite3
import torch

from pydantic import BaseModel

class Hold(BaseModel):
    x: int
    y: int
    channel: int

class Hold_list(BaseModel):
    holds: list[Hold]


class Converter:

    def __init__(self):
        self.db_path = r'../../data/raw/db.sqlite3'
        


    def get_grade_dictionary(self):

        conn = sqlite3.connect(str(self.db_path))
        cursor = conn.cursor()

    
        query = """
        SELECT
            difficulty, 
            boulder_name
        FROM difficulty_grades
        """

        cursor.execute(query)
        result = cursor.fetchall()

        grade_dict = {}

        for x in range(len(result)):
            grade_dict[result[x][0]] = result[x][1].split("/")[1]

        return grade_dict

    def convert_to_tensor(list:Hold_list):
        tensor = torch.zeros(1, 4, 173, 185, dtype=torch.float32)
        for h in list:
            tx = h.x + 20
            ty = h.y - 4    
        tensor[0, h.channel, ty, tx] = 1.0
