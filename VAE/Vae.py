import argparse

import torch
import torch.nn as nn
import matplotlib.pyplot as plt
from torchvision.utils import make_grid
import numpy as np
import matplotlib
import math
from scipy.stats import norm
from train import retrieve_data
import lstm
from torch.utils.data import DataLoader

class Encoder(nn.Module):

    def __init__(self, vocabulary_size, lstm_num_hidden=250, lstm_num_layers=2, hidden_dim=250, z_dim=20, device='cuda:0', dropout_prob=0.):
        super().__init__()
        self.to(device)
        embedding_dim = 200
        self.embed = nn.Embedding(vocabulary_size, embedding_dim=embedding_dim)
        self.lstm = nn.LSTM(embedding_dim, lstm_num_hidden, lstm_num_layers, dropout=dropout_prob)

        self.h = nn.Linear(lstm_num_hidden, hidden_dim)
        self.mean = nn.Linear(hidden_dim, z_dim)
        self.std = nn.Linear(hidden_dim, z_dim)
        self.relu = nn.ReLU()
        self.softplus = nn.Softplus()

    def forward(self, input):
        """
        Perform forward pass of encoder.
        Returns mean and std with shape [batch_size, z_dim]. Make sure
        that any constraints are enforced.
        """
        lstm = self.lstm(input)
        h = self.h(lstm)
        h = self.relu(h)

        mean = self.mean(h)
        std = self.std(h)
        std = self.softplus(std)

        return mean, std


class Decoder(nn.Module):

    def __init__(self, vocabulary_size, lstm_num_hidden=250, lstm_num_layers=3, hidden_dim=250, z_dim=20, device='cuda:0', dropout_prob=0.):
        super().__init__()
        embedding_dim = 200
        self.embed = nn.Embedding(vocabulary_size, embedding_dim=embedding_dim)
        self.lstm = nn.LSTM(z_dim, embedding_dim, lstm_num_layers, dropout=dropout_prob)



    def forward(self, z):
        """
        Perform forward pass of decoder.
        Returns mean with shape [batch_size, 784].
        """
        out = self.lstm(z)
        return out


class VAE(nn.Module):

    def __init__(self, hidden_dim=500, z_dim=20):
        super().__init__()

        self.z_dim = z_dim
        self.encoder = Encoder(hidden_dim, z_dim)
        self.decoder = Decoder(hidden_dim, z_dim)

    def forward(self, input):
        """
        Given input, perform an encoding and decoding step and return the
        negative average elbo for the given batch.
        """
        mean, std = self.encoder(input)

        e = torch.zeros(mean.shape).normal_()
        z = std * e + mean

        y = self.decoder(z)

        eps = 1e-8
        L_reconstruction = input * y.log() + (1 - input) * (1 - y).log()
        KLD = 0.5 * (std.pow(2) + mean.pow(2) - 1 - torch.log(std.pow(2)+eps))
        elbo = KLD.sum(dim=-1) - L_reconstruction.sum(dim=-1)
        average_negative_elbo = elbo.mean()

        return average_negative_elbo


    def sample(self, n_samples):
        """
        Sample n_samples from the model. Return both the sampled images
        (from bernoulli) and the means for these bernoullis (as these are
        used to plot the data manifold).
        """

        samples = torch.randn((n_samples, self.z_dim))
        y = self.decoder(samples)

        im_means = y.reshape(n_samples, 1, 28, 28)
        sampled_ims = torch.bernoulli(im_means)

        return sampled_ims, im_means


    def manifold_sample(self, n_samples):
        n = int(math.sqrt(n_samples))
        xy = torch.zeros(n_samples, 2)
        xy[:, 0] = torch.arange(0.01, n, 1 / n) % 1
        xy[:, 1] = (torch.arange(0.01, n_samples, 1) / n).float() / n
        z = torch.erfinv(2 * xy - 1) * math.sqrt(2)


        with torch.no_grad():
            mean = self.decoder(z)
        return mean

def epoch_iter(model, data, optimizer, device):
    """
    Perform a single epoch for either the training or validation.
    use model.training to determine if in 'training mode' or not.
    Returns the average elbo for the complete epoch.
    """
    average_epoch_elbo = 0
    size = len(data)
    data_loader = DataLoader(data, 64, num_workers=1)

    for sample in data_loader:

        tensor_sample = torch.stack(sample, dim=0).to(device)

        device_inputs = sample.reshape(tensor_sample.shape[0], -1)

        elbo = model.forward(device_inputs)
        average_epoch_elbo -= elbo

        if model.training:
            model.zero_grad()
            elbo.backward()
            optimizer.step()

    average_epoch_elbo /= size

    return average_epoch_elbo


def run_epoch(model, train, val, optimizer, device):
    """
    Run a train and validation epoch and return average elbo for each.
    """
    model.train()
    train_elbo = epoch_iter(model, train, optimizer, device)

    model.eval()
    val_elbo = epoch_iter(model, val, optimizer, device)

    return train_elbo, val_elbo


def save_elbo_plot(train_curve, val_curve, filename):
    plt.figure(figsize=(24, 12))
    plt.plot(train_curve, label='train elbo')
    plt.plot(val_curve, label='validation elbo')
    plt.legend()
    plt.xlabel('epochs')
    plt.ylabel('ELBO')
    plt.tight_layout()
    plt.savefig(filename)


def save_sample(sample, size, epoch, nrow=8):
    sample = sample.view(-1, 1, size, size)
    sample = make_grid(sample, nrow=nrow).detach().numpy().astype(np.float).transpose(1, 2, 0)
    matplotlib.image.imsave(f"images/vae_manimani_{epoch}.png", sample)

def main():
    device = torch.device('cpu')
    train, val, test = retrieve_data()
    model = VAE(z_dim=ARGS.zdim)
    optimizer = torch.optim.Adam(model.parameters())
    size_width = 28

    train_curve, val_curve = [], []
    for epoch in range(ARGS.epochs):
        elbos = run_epoch(model, train, val, optimizer, device)
        train_elbo, val_elbo = elbos
        train_curve.append(train_elbo)
        val_curve.append(val_elbo)
        print(f"[Epoch {epoch}] train elbo: {train_elbo} val_elbo: {val_elbo}")

        # --------------------------------------------------------------------
        #  Add functionality to plot samples from model during training.
        #  You can use the make_grid functioanlity that is already imported.
        # --------------------------------------------------------------------

        # _, mean_sample = model.sample(64)
        # save_sample(mean_sample, size_width, epoch)

    if ARGS.zdim == 2:
        print("manifold")
        manifold = model.manifold_sample(256)
        save_sample(manifold, size_width, epoch, 16)

    np.save('curves.npy', {'train': train_curve, 'val': val_curve})
    # --------------------------------------------------------------------
    #  Add functionality to plot plot the learned data manifold after
    #  if required (i.e., if zdim == 2). You can use the make_grid
    #  functionality that is already imported.
    # --------------------------------------------------------------------

    save_elbo_plot(train_curve, val_curve, 'elbo.pdf')


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--epochs', default=40, type=int,
                        help='max number of epochs')
    parser.add_argument('--zdim', default=2, type=int,
                        help='dimensionality of latent space')

    ARGS = parser.parse_args()

    main()