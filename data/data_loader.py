import pandas as pd
import numpy as np


def load_data():
    # Read csv file
    data_file = pd.read_csv("Spotify_Youtube.csv")

    # remove insignificant columns
    processed_data_file = data_file.drop(
        columns=['Album_type', 'Artist', 'Url_spotify', 'Track', 'Album', 'Uri', 'Url_youtube',
                 'Title', 'Channel', 'Description'])

    # rename the id column
    processed_data_file = processed_data_file.rename(columns={'Unnamed: 0': 'TrackID'})

    # drop entries that have empty Stream cells
    processed_data_file = processed_data_file.dropna(subset=['Stream'])

    # replace NaN with 0 and convert boolean values into numerical boolean
    processed_data_file['Licensed'] = processed_data_file['Licensed'].replace(np.nan, 0)
    processed_data_file['Licensed'] = processed_data_file['Licensed'].astype(int)
    processed_data_file['official_video'] = processed_data_file['official_video'].replace(np.nan, 0)
    processed_data_file['official_video'] = processed_data_file['official_video'].astype(int)

    filtered_data = processed_data_file[(processed_data_file['Stream'] >= 50000000.0)]

    print(filtered_data)

    # split data set based on criteria of popular songs
    training_labels = filtered_data[['TrackID', 'Views', 'Likes', 'Comments', 'Stream']]
    training_inputs = filtered_data.drop(columns=['Views', 'Likes', 'Comments', 'Stream'])
    testing_inputs = processed_data_file.drop(training_inputs.index)
    testing_labels = testing_inputs.loc[:, ['TrackID', 'Views', 'Likes', 'Comments', 'Stream']]
    testing_inputs = testing_inputs.drop(columns=['TrackID', 'Views', 'Likes', 'Comments', 'Stream'])

    return training_labels, training_inputs, testing_labels, testing_inputs


def prediction(training_labels, training_inputs, testing_labels, testing_inputs):
    # normalize columns relating to the characteristics of the songs
    key_min = training_inputs['Key'].min()
    key_max = training_labels['Key'].max()
    training_inputs['Key'] = (training_inputs['Key'] - key_min) / (key_max - key_min)

    loudness_min = training_inputs['Loudness'].min()
    loudness_max = training_inputs['Loudness'].max()
    training_inputs['Loudness'] = (training_inputs['Loudness'] - loudness_min) / (loudness_max - loudness_min)

    tempo_min = training_inputs['Tempo'].min()
    tempo_max = training_inputs['Tempo'].max()
    training_inputs['Tempo'] = (training_inputs['Tempo'] - tempo_min) / (tempo_max - tempo_min)

    duration_min = training_inputs['Duration_ms'].min()
    duration_max = training_inputs['Duration_ms'].max()
    training_inputs['Duration_ms'] = (training_inputs['Duration_ms'] - duration_min) / (duration_max - duration_min)


if __name__ == "__main__":
    load_data()
