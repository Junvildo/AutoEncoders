from torch import nn


class Encoder(nn.Module):
    def __init__(self, input_size, hidden_size):
        super(Encoder, self).__init__()
        self.encoder = nn.Sequential(
            nn.Linear(input_size, hidden_size),
            nn.LeakyReLU(),
            nn.Linear(hidden_size, hidden_size*2),
            nn.LeakyReLU(),
            nn.Dropout(0.2),
            nn.Linear(hidden_size*2, hidden_size*3)
        )

    def forward(self, x):
        x = self.encoder(x)
        return x

class Decoder(nn.Module):
    def __init__(self, input_size, hidden_size):
        super(Decoder, self).__init__()
        self.decoder = nn.Sequential(
            nn.Linear(hidden_size*3, hidden_size*2),
            nn.LeakyReLU(),
            nn.Linear(hidden_size*2, hidden_size),
            nn.LeakyReLU(),
            nn.Dropout(0.2),
            nn.Linear(hidden_size, input_size)
        )

    def forward(self, x):
        x = self.decoder(x)
        return x

class AutoEncoder(nn.Module):
    def __init__(self, input_size, hidden_size):
        super(AutoEncoder, self).__init__()
        self.encoder = Encoder(input_size, hidden_size)
        self.decoder = Decoder(input_size, hidden_size)

    def forward(self, x):
        x = self.encoder(x)
        x = self.decoder(x)
        return x