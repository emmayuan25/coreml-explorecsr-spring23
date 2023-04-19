import pandas as pd
import numpy as np
import torch
from torch import nn
import matplotlib as mpl
import matplotlib.pyplot as plt


class AutoEncoder(nn.Module):
    def __init__(self, output_dim=14):
        super(AutoEncoder, self).__init__()
        self.enc = nn.Sequential(
            nn.Linear(output_dim, round(output_dim * 0.75)),
            nn.Tanh(),
            # nn.Dropout(p = 0.1),
            nn.Linear(round(output_dim * 0.75), round(output_dim * 0.50)),
            nn.Tanh(),
            # nn.Dropout(p = 0.1),
            nn.Linear(round(output_dim * 0.50), round(output_dim * 0.33)),
            nn.Tanh(),
            # nn.Dropout(p = 0.1),
            nn.Linear(round(output_dim * 0.33), round(output_dim * 0.25)),
        )
        self.dec = nn.Sequential(
            nn.Linear(round(output_dim * 0.25), round(output_dim * 0.33)),
            nn.Tanh(),
            # nn.Dropout(p = 0.1),
            nn.Linear(round(output_dim * 0.33), round(output_dim * 0.50)),
            nn.Tanh(),
            # nn.Dropout(p = 0.1),
            nn.Linear(round(output_dim * 0.50), round(output_dim * 0.75)),
            nn.Tanh(),
            # nn.Dropout(p = 0.1),
            nn.Linear(round(output_dim * 0.75), output_dim),
        )

    def forward(self, x):
        encode = self.enc(x)
        decode = self.dec(encode)
        return decode


def train(model, train_data, device, epochs, learning_rate):
    model = model

    model.to(device)
    model.train()

    # train and update
    criterion = nn.MSELoss().to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

    epoch_loss = []
    for epoch in range(epochs):
        batch_loss = []
        for batch_idx, x in enumerate(train_data):
            x = x.to(device).float()
            optimizer.zero_grad()
            decode = model(x)
            e_loss = criterion(decode, x)
            e_loss.backward()
            optimizer.step()

            batch_loss.append(e_loss.item())
        epoch_loss.append(sum(batch_loss) / len(batch_loss))
        # print('epoch loss is ' + str(epoch_loss[epoch]))

    return epoch_loss


def plot_loss(epoch_loss, epoch, lr):
    # plot epoch loss curve
    plt.plot(epoch_loss)
    plt.title("Learning Rate=" + str(lr) + " Epoch=" + str(epoch))
    plt.savefig('epoch_loss.png')


def process_min_max(training_inputs):
    # normalize columns relating to the characteristics of the songs on training data
    danceability_min = training_inputs['Danceability'].min()
    danceability_max = training_inputs['Danceability'].max()
    training_inputs['Danceability'] = (training_inputs['Danceability'] - danceability_min) / (
            danceability_max - danceability_min)

    energy_min = training_inputs['Energy'].min()
    energy_max = training_inputs['Energy'].max()
    training_inputs['Energy'] = (training_inputs['Energy'] - energy_min) / (energy_max - energy_min)

    key_min = training_inputs['Key'].min()
    key_max = training_inputs['Key'].max()
    training_inputs['Key'] = (training_inputs['Key'] - key_min) / (key_max - key_min)

    loudness_min = training_inputs['Loudness'].min()
    loudness_max = training_inputs['Loudness'].max()
    training_inputs['Loudness'] = (training_inputs['Loudness'] - loudness_min) / (loudness_max - loudness_min)

    speechiness_min = training_inputs['Speechiness'].min()
    speechiness_max = training_inputs['Speechiness'].max()
    training_inputs['Speechiness'] = (training_inputs['Speechiness'] - speechiness_min) / (
            speechiness_max - speechiness_min)

    acousticness_min = training_inputs['Acousticness'].min()
    acousticness_max = training_inputs['Acousticness'].max()
    training_inputs['Acousticness'] = (training_inputs['Acousticness'] - acousticness_min) / (
            acousticness_max - acousticness_min)

    instrumentalness_min = training_inputs['Instrumentalness'].min()
    instrumentalness_max = training_inputs['Instrumentalness'].max()
    training_inputs['Instrumentalness'] = (training_inputs['Instrumentalness'] - instrumentalness_min) / (
            instrumentalness_max - instrumentalness_min)

    liveness_min = training_inputs['Liveness'].min()
    liveness_max = training_inputs['Liveness'].max()
    training_inputs['Liveness'] = (training_inputs['Liveness'] - liveness_min) / (liveness_max - liveness_min)

    valence_min = training_inputs['Valence'].min()
    valence_max = training_inputs['Valence'].max()
    training_inputs['Valence'] = (training_inputs['Valence'] - valence_min) / (valence_max - valence_min)

    tempo_min = training_inputs['Tempo'].min()
    tempo_max = training_inputs['Tempo'].max()
    training_inputs['Tempo'] = (training_inputs['Tempo'] - tempo_min) / (tempo_max - tempo_min)

    duration_min = training_inputs['Duration_ms'].min()
    duration_max = training_inputs['Duration_ms'].max()
    training_inputs['Duration_ms'] = (training_inputs['Duration_ms'] - duration_min) / (duration_max - duration_min)

    return training_inputs


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

    # split data by cutoff
    stream_cutoff = 50000000.0
    filtered_data = processed_data_file[(processed_data_file['Stream'] >= stream_cutoff)]

    # split data set based on criteria of popular songs
    loader_training_inputs = filtered_data.drop(columns=['Views', 'Likes', 'Comments', 'Stream'])
    loader_testing_inputs = processed_data_file.drop(loader_training_inputs.index)
    loader_testing_inputs = loader_testing_inputs.drop(columns=['Views', 'Likes', 'Comments', 'Stream'])

    # normalize using min max
    loader_training_inputs = process_min_max(loader_training_inputs)

    # pandas array to numpy
    loader_training_inputs = np.array(loader_training_inputs)
    loader_testing_inputs = np.array(loader_testing_inputs)
    loader_training_inputs[np.isnan(loader_training_inputs)] = 0
    loader_testing_inputs[np.isnan(loader_testing_inputs)] = 0

    # put data into dataloader
    train_data = torch.utils.data.DataLoader(
        loader_training_inputs, batch_size=128, shuffle=False
    )
    test_data = torch.utils.data.DataLoader(
        loader_testing_inputs, batch_size=128, shuffle=False
    )

    return train_data, test_data


if __name__ == "__main__":
    # load data
    training_inputs, testing_inputs = load_data()

    # train data
    autoencoder = AutoEncoder()
    epoch = 100
    lr = 0.1
    loss = train(model=autoencoder, train_data=training_inputs, device="cpu", epochs=epoch, learning_rate=lr)

    plot_loss(loss, epoch, lr)
