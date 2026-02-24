import torch
from pathlib import Path
import sqlite3
import numpy as np

def create_kiter_dataset():
    connection = sqlite3.connect(r'../../data/raw/db.sqlite3')
    cursor = connection.cursor()

    positions = get_poistions_map(cursor)



def get_frames(cursor):
    query="""
    SELECT 
    climbs.frames
    FROM climbs
    WHERE climbs.layout_id = 1
    """

    cursor.execute(query)
    result = cursor.fetchall()

    frames_list = []
    np_r = np.array(result)
    for x in range(np_r.shape[0]):
        frames_list.append(np_r[x][0])

    return frames_list





def create_tensor(positions, frame):

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

        x, y = positions[hold_id]


        if role == 12: channel = 0
        elif role == 13: channel = 1
        elif role == 14: channel = 2
        elif role == 15: channel = 3
        if channel != -1:
            tensor[channel, y, x] = 1.0

    return tensor




def get_poistions_map(cursor):
    query = """
    SELECT placements.id, holes.x, holes.y
    FROM placements
    JOIN holes ON placements.hole_id = holes.id
    WHERE placements.layout_id = 1
    """

    # remember to add connection when transfering to py file

    cursor.execute(query)

    result = cursor.fetchall()

    ### building placement: position dictionary

    positions = {}
    for pid, x, y in result:
        c_x = x + 20 #received in notebook file
        c_y = y - 4
        positions[pid] = (c_x, c_y)

    return positions

