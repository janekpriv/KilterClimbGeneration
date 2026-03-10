import torch
from torch.utils.data import Dataset, DataLoader
from pathlib import Path
import sqlite3
import numpy as np


class KilterDataSet(Dataset):

    def __init__(self, database_path=r'../../data/raw/db.sqlite3'):
        connection = sqlite3.connect(database_path)
        self.cursor = connection.cursor()
        self.frames = self.get_frames()
        self.positions = self.get_poistions_map()

    def __len__(self):
        return(len(self.frames))

    def __getitem__(self, idx):
        frame = self.frames[idx][0]
        grade = self.frames[idx][1]

        grade_tensor = torch.tensor([grade], dtype=torch.float32)
        route_tensor = self.create_tensor(frame)
        return route_tensor, grade_tensor

    def get_frames(self):
        query="""
        SELECT 
        climbs.frames,
        climb_stats.difficulty_average
        FROM climbs
        JOIN climb_stats ON climb_stats.climb_uuid = climbs.uuid
        WHERE climb_stats.angle = 40    
          AND climbs.layout_id = 1
          AND climbs.frames IS NOT NULL 
          AND climbs.frames NOT LIKE '%x%'
          AND climbs.frames NOT LIKE '%,%'
          AND climbs.frames NOT LIKE '%"%'
          LIMIT 10000;
        """

        self.cursor.execute(query)
        result = self.cursor.fetchall()

        frames_list = []
        for x in range(len(result)):
            frames_list.append((result[x][0], result[x][1]))

        return frames_list

    def create_tensor(self, frame):

        """
        data from notebook
        185 widht 
        173 height
        """
        
        tensor = torch.zeros((4, 173, 185), dtype=torch.float32)



        holds = frame.split('p')
        for hold in holds:
            if not hold: continue
            if "r" not in hold: continue

            channel = -1
            hold_id_str, role_str = hold.split("r")
            hold_id = int(hold_id_str)
            role = int(role_str)

            x, y = self.positions[hold_id]


            if role == 12: channel = 0
            elif role == 13: channel = 1
            elif role == 14: channel = 2
            elif role == 15: channel = 3
            if channel != -1:
                tensor[channel, y, x] = 1.0

        return tensor

    def get_poistions_map(self):
        query = """
        SELECT placements.id, holes.x, holes.y
        FROM placements
        JOIN holes ON placements.hole_id = holes.id
        WHERE placements.layout_id = 1
        """

        # remember to add connection when transfering to py file

        self.cursor.execute(query)

        result = self.cursor.fetchall()

        ### building placement: position dictionary

        positions = {}
        for pid, x, y in result:
            c_x = x + 20 #received in notebook file
            c_y = y - 4
            positions[pid] = (c_x, c_y)

        return positions


def get_data_loaders(db_path, batch_size=32):
    
    kilter_dataset = KilterDataSet(database_path=r'../../data/raw/db.sqlite3')
    train_dataset, test_dataset = torch.utils.data.random_split(kilter_dataset, [0.8, 0.2])

    train_data_loader = DataLoader(kilter_dataset, batch_size=32, shuffle=True)
    val_data_loader = DataLoader(kilter_dataset, batch_size=32, shuffle=True)

    return train_data_loader, val_data_loader