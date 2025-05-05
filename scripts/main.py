import pandas as pd

KEY_POINTS_PATH = "data/model/hdf"

def execute():
    df = pd.read_hdf(KEY_POINTS_PATH + "/moving_open_left_hand.h5")
    print(df.sample(10))
    

if __name__ == '__main__':
    execute()