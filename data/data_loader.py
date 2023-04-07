import pandas as pd
import numpy as np


def load_data():
    # Read csv file
    data_file = pd.read_csv("Spotify_Youtube.csv")
    print(data_file)

    # Clean data: replace missing or null value (with mode?),
    # remove insignificant columns

    # Preprocess data: categorical to numerical, normalize, etc.

    # Split dataset into train and test data arrays?


if __name__ == "__main__":
    load_data()
