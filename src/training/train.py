from data.dataset import get_data_loaders

def main():
    db_path = r'../../data/raw/db.sqlite3'

    training_data_loader, val_data_loader = get_data_loaders(db_path)