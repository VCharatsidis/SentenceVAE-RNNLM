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

    def __init__(self, embedding_dim, lstm_num_hidden=250, lstm_num_layers=2, z_dim=20, device='cuda:0', dropout_prob=0.):
        super().__init__()
        self.to(device)
        self.lstm_num_hidden = lstm_num_hidden
        self.lstm = nn.LSTM(embedding_dim, lstm_num_hidden, lstm_num_layers, dropout=dropout_prob, batch_first=True)

        self.projection = nn.Linear(lstm_num_hidden, lstm_num_layers * z_dim)
        self.mean = nn.Linear(lstm_num_layers * z_dim, z_dim)
        self.std = nn.Linear(lstm_num_layers * z_dim, z_dim)
        self.softplus = nn.Softplus()

    def forward(self, embeded_input, batch_size):
        """
        Perform forward pass of encoder.
        Returns mean and std with shape [batch_size, z_dim]. Make sure
        that any constraints are enforced.
        """

        outs, hidden = self.lstm(embeded_input)

        h = hidden.view(batch_size, self.hidden_size)
        h = self.projection(h)
        mean = self.mean(h)
        std = self.std(h)
        std = self.softplus(std)

        return mean, std


class Decoder(nn.Module):

    def __init__(self, vocabulary_size, input_size, lstm_num_hidden=250, lstm_num_layers=1, z_dim=20, dropout_prob=0.):
        super().__init__()

        self.lstm_num_layers = 3
        self.latent2hidden = nn.Linear(z_dim, lstm_num_hidden)
        self.tanh = nn.Tanh()

        self.lstm = nn.LSTM(input_size, lstm_num_hidden, num_layers=lstm_num_layers, dropout=dropout_prob, batch_first=True)
        self.projection = nn.Linear(lstm_num_hidden, vocabulary_size)


    def forward(self, z, embeddings):
        """
        Perform forward pass of decoder.
        Returns mean with shape [batch_size, 784].
        """

        hidden = self.latent2hidden(z)
        hdden = self.tanh(hidden)
        hidden = hidden.view(self.lstm_num_layers, batch_size, self.hidden_size)
        hidden_states, (h, c) = self.lstm(packed_input, hidden)

        return hidden_states, (h, c)



class VAE(nn.Module):

    def __init__(self, vocabulary_size, input_size, lstm_num_hidden, z_dim=20):
        super().__init__()

        lstm_num_hidden = 250
        lstm_num_layers = 2
        device = 'cuda:0'
        dropout_prob = 0.

        embedding_dim = 200
        self.embed = nn.Embedding(vocabulary_size, input_size, padding_idx=0)
        self.z_dim = z_dim
        self.encoder = Encoder(embedding_dim, lstm_num_hidden, lstm_num_layers, z_dim, device, dropout_prob)
        self.decoder = Decoder(vocabulary_size, lstm_num_hidden, lstm_num_layers, z_dim, dropout_prob)

    def forward(self, input, lengths):
        """
        Given input, perform an encoding and decoding step and return the
        negative average elbo for the given batch.
        """
        batch_size = input.size(0)
        embeddings = self.embed(input)

        mean, std = self.encoder(embeddings)

        e = torch.zeros(mean.shape).normal_()
        z = std * e + mean

        prob = torch.rand(input.size())
        prob[(input.data - self.vocab.SOS_INDEX) * (input.data - self.vocab.PAD_INDEX) == 0] = 1
        decoder_input_sequence = input.clone()
        decoder_input_sequence[prob < 0.75] = self.vocab.UNK_INDEX
        embeddings = self.embed(decoder_input_sequence)

        y, _ = self.decoder(z, embeddings)

        criterion = nn.CrossEntropyLoss()
        y = y.transpose(0, 1).transpose(1, 2)

        print(y.shape)
        print(input.shape)

        input = input.transpose(0, 1)

        print(input.shape)

        L_reconstruction = criterion.forward(y, input)
        eps = 1e-8

        KLD = 0.5 * (std.pow(2) + mean.pow(2) - 1 - torch.log(std.pow(2)+eps))
        elbo = KLD.sum(dim=-1) - L_reconstruction

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
    batch_size = 16
    data_loader = DataLoader(data, batch_size, num_workers=1)

    for step, (batch_inputs, batch_targets, lengths) in enumerate(data_loader):

        # if not batch_inputs:
        #     continue

        tensor_sample = torch.stack(batch_inputs, dim=0).to(device)

        elbo = model.forward(tensor_sample, lengths)
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

    model = VAE(train.vocab_size, z_dim=ARGS.zdim)
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