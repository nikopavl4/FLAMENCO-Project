import torch


class AutoEncoder(torch.nn.Module):
    """
    Autoencoder model

    Args:
        input_dim (int): Input data dimension.
        encoding_dim (int): Hidden representation dimension.
    """

    def __init__(self, input_dim: int, encoding_dim: int):
        super(AutoEncoder, self).__init__()

        self.encoder = torch.nn.Sequential(
            torch.nn.Linear(input_dim, 64),
            torch.nn.ReLU(),
            torch.nn.Dropout(0.2),  # drop out
            torch.nn.Linear(64, encoding_dim)
        )
        self.decoder = torch.nn.Sequential(
            torch.nn.Linear(encoding_dim, 64),
            torch.nn.Dropout(0.2),
            torch.nn.ReLU(),
            torch.nn.Linear(64, input_dim),
            torch.nn.Sigmoid()
        )

    def forward(self, x):
        encoded = self.encoder(x)
        decoded = self.decoder(encoded)
        return encoded, decoded


class VariationalAutoEncoder(torch.nn.Module):
    """
    Variational Autoencoder model

    Args:
        input_dim (int): Input data dimension.
        encoding_dim (int): Hidden representation dimension.
    """

    def __init__(self, input_dim: int, encoding_dim: int):
        super(VariationalAutoEncoder, self).__init__()

        self.encoder = torch.nn.Sequential(
            torch.nn.Linear(input_dim, 64),
            torch.nn.ReLU(),
            torch.nn.Dropout(0.2),
            torch.nn.Linear(64, encoding_dim)
        )

        self.mu = torch.nn.Linear(encoding_dim, encoding_dim)
        self.logvar = torch.nn.Linear(encoding_dim, encoding_dim)

        self.decoder = torch.nn.Sequential(
            torch.nn.Linear(encoding_dim, 64),
            torch.nn.ReLU(),
            torch.nn.Linear(64, input_dim),
            torch.nn.Sigmoid()
        )

    def encode(self, x):
        x = self.encoder(x)
        mu = self.mu(x)
        logvar = self.logvar(x)
        return mu, logvar

    def reparametrize(self, mu, log_var):
        std = torch.exp(0.5 * log_var)
        eps = torch.rand_like(std)
        return mu + eps * std

    def decode(self, z):
        out = self.decoder(z)
        return out

    def forward(self, x):
        mu, log_var = self.encode(x)
        z = self.reparametrize(mu, log_var)
        decoded = self.decode(z)

        return [mu, log_var], decoded
