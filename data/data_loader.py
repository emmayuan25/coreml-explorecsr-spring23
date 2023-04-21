import pandas as pd
import numpy as np
import torch
from torch import nn
import matplotlib as mpl
import matplotlib.pyplot as plt


def process_min_max(training_inputs, testing_inputs):
    # normalize columns relating to the characteristics of the songs on training data
    train_min = np.min(training_inputs, axis=0)
    train_max = np.max(training_inputs, axis=0)

    training_inputs = training_inputs - train_min / train_max - train_min
    testing_inputs = testing_inputs - train_min / train_max - train_min

    # print(training_inputs)
    # print(testing_inputs)

    return training_inputs, testing_inputs


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
    loader_training_inputs = filtered_data.drop(columns=['Views', 'Likes', 'Comments', 'Stream', 'TrackID', 'Duration_ms'])
    loader_testing_inputs = processed_data_file.drop(loader_training_inputs.index)
    loader_testing_inputs = loader_testing_inputs.drop(columns=['Views', 'Likes', 'Comments', 'Stream', 'TrackID', 'Duration_ms'])

    # pandas array to numpy
    loader_training_inputs = np.array(loader_training_inputs)
    loader_testing_inputs = np.array(loader_testing_inputs)
    loader_training_inputs[np.isnan(loader_training_inputs)] = 0
    loader_testing_inputs[np.isnan(loader_testing_inputs)] = 0

    # normalize using min max
    loader_training_inputs, loader_testing_inputs = process_min_max(loader_training_inputs, loader_testing_inputs)

    # put data into dataloader
    train_data = torch.utils.data.DataLoader(
        loader_training_inputs, batch_size=128, shuffle=True
    )
    eval_data = torch.utils.data.DataLoader(
        loader_training_inputs, batch_size=1, shuffle=False
    )
    test_data = torch.utils.data.DataLoader(
        loader_testing_inputs, batch_size=1, shuffle=False
    )

    return train_data, test_data, eval_data


class AutoEncoder(nn.Module):
    def __init__(self, output_dim=12):
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

    return epoch_loss


def plot_loss(epoch_loss, epoch, lr):
    # plot epoch loss curve
    plt.plot(epoch_loss)
    plt.title("Learning Rate=" + str(lr) + " Epoch=" + str(epoch))
    plt.savefig('epoch_loss.png')


def get_threshold(eval_data, device, model, alpha):
    model.eval()
    model.to(device)

    criterion = nn.MSELoss().to(device)

    eval_loss = []
    for batch_idx, x in enumerate(eval_data):
        x = x.to(device).float()
        decode = model(x)
        e_loss = criterion(decode, x)
        eval_loss.append(e_loss.item())

    mean = np.mean(eval_loss)
    var = np.var(eval_loss)
    tr = mean + alpha * var

    return tr


def test_model(model, device, test_data, tr):
    model.eval()
    model.to(device)

    criterion = nn.MSELoss().to(device)

    predictions = []
    for batch_idx, x in enumerate(test_data):
        x = x.to(device).float()
        decode = model(x)
        e_loss = criterion(decode, x)

        if tr > e_loss.item():
            predictions.append(batch_idx)

    return predictions


if __name__ == "__main__":
    # load data
    training_inputs, testing_inputs, eval_data = load_data()

    # train data
    autoencoder = AutoEncoder()
    epoch = 300
    lr = 0.5
    loss = train(model=autoencoder, train_data=training_inputs, device="cpu", epochs=epoch, learning_rate=lr)

    # plot_loss(loss, epoch, lr)

    alpha = 0
    tr = get_threshold(eval_data, device="cpu", model=autoencoder, alpha=alpha)
    print(tr)
    #
    predictions = test_model(model=autoencoder, test_data=testing_inputs, device="cpu", tr=tr)

    print(len(predictions))
