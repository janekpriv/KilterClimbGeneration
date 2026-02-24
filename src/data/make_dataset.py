import torch
from torch.utils.data import Dataset
from pathlib import Path
import sqlite3
import numpy as np


class KilterDataSet(Dataset):

    def __init__(self, root_dir, database_path=r'../../data/raw/db.sqlite3'):
        connection = sqlite3.connect(database_path)
        self.cursor = connection.cursor()
        self.frames = self.get_frames()
        self.positions = self.get_poistions_map()

    def __len__(self):
        return(len(self.frames))

    def __getitem__(self, idx):
        frame = self.frames[idx]
        tensor = self.create_tensor(frame)
        return tensor

    def get_frames(self):
        query="""
        SELECT 
        climbs.frames
        FROM climbs
        WHERE climbs.layout_id = 1
        """

        self.cursor.execute(query)
        result = self.cursor.fetchall()

        frames_list = []
        for x in range(len(result)):
            frames_list.append(result[x][0])

        return frames_list

    def create_tensor(self, frame):

        """
        data from notebook
        185 widht 
        173 height
        """
        
        tensor = torch.zeros((4, 185, 173), dtype=torch.float32)

        holds = frame.split('p')
        print(holds)
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

