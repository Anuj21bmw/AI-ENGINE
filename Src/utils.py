import pandas as pd

def load_data(user_file, item_file):
    user_df = pd.read_csv(user_file)
    item_df = pd.read_csv(item_file)
    return user_df, item_df
