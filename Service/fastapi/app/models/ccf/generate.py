import pickle
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn as nn


DEVICE = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")


class Generator(nn.Module):

    def __init__(self, n_inputs, n_outputs):
        super(Generator, self).__init__()

        self.net = nn.Sequential(
            nn.Linear(n_inputs, 1000),
            nn.ReLU(),

            nn.Linear(1000, 500),
            nn.BatchNorm1d(500),
            nn.Dropout(0.2),
            nn.ReLU(),

            nn.Linear(500, n_outputs)
        )

    def forward(self, z, y):
        zy = torch.cat((z, y), dim=1)
        return self.net(zy)


# Генератор CCF
latent_dim=100

generator_ccf = Generator(n_inputs=latent_dim+1, n_outputs=30)
qt_model = None
generator_ccf.to(DEVICE)

def frac_gen(frac):
    f = frac * 1000
    n = 100000 - f
    y_1 = np.random.randint(low=0, high=1, size=n)
    y_2 = np.random.randint(low=1, high=2, size=f)
    y = np.concatenate([y_1, y_2])
    np.random.shuffle(y)
    return y

def generate(generator, y, latent_dim):

    Z_noise = torch.normal(0, 1, (len(y), latent_dim)).to(DEVICE)
    y = torch.tensor(y, dtype=torch.float, device=DEVICE)
    X_fake = generator(Z_noise, y.unsqueeze(1)).cpu().detach().numpy()

    return X_fake



def load_model(path_to_model: Path, path_to_qt: Path) -> None:
    generator_ccf.load_state_dict(torch.load(path_to_model))
    generator_ccf.eval()

    global qt_model
    with open(path_to_qt, "rb") as fp:
        qt_model = pickle.load(fp)


def run_model(frac: float) -> pd.DataFrame:
    y = frac_gen(frac)
    X_gan = generate(generator_ccf, y, latent_dim)
    X = qt_model.inverse_transform(X_gan)
    X_df = pd.DataFrame(X)
    y_df = pd.DataFrame(y)
    return pd.concat((X_df, y_df), axis=1)

