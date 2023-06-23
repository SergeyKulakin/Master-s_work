import pickle
from pathlib import Path

import pandas as pd
import torch
import torch.nn as nn
import numpy as np

DEVICE = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")


class Generator_BAF(nn.Module):

    def __init__(self, n_inputs, n_outputs):
        super(Generator_BAF, self).__init__()

        self.net = nn.Sequential(
            nn.Linear(n_inputs, 1000),
            nn.BatchNorm1d(1000),
            nn.Dropout(0.2),
            nn.ReLU(),

            nn.Linear(1000, 2000),
            nn.BatchNorm1d(2000, momentum=0.4),
            nn.Dropout(0.4),
            nn.ReLU(),

            nn.Linear(2000, 100),
            nn.BatchNorm1d(100, momentum=0.4),
            nn.Dropout(0.2),
            nn.ReLU(),

            nn.Linear(100, n_outputs)
        )

    def forward(self, z, y):
        zy = torch.cat((z, y), dim=1)
        return self.net(zy)




generator_baf = Generator_BAF(n_inputs=33, n_outputs=31)
latent_dim_baf=32
generator_baf.eval()
generator_baf.to(DEVICE)
qt_model = None


def load_model(path_to_model: Path, path_to_qt: Path) -> None:
    generator_baf.load_state_dict(torch.load(path_to_model))

    generator_baf.eval()

    global qt_model
    with open(path_to_qt, "rb") as fp:
        qt_model = pickle.load(fp)


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


def run_model(frac: float) -> pd.DataFrame:
    y = frac_gen(frac)
    X_gan = generate(generator_baf, y, latent_dim_baf)

    X_inverse = qt_model.inverse_transform(X_gan)

    # Соберем DataFrame
    names = ['income', 'name_email_similarity', 'prev_address_months_count',
             'current_address_months_count', 'customer_age', 'days_since_request',
             'intended_balcon_amount', 'payment_type', 'zip_count_4w', 'velocity_6h',
             'velocity_24h', 'velocity_4w', 'bank_branch_count_8w',
             'date_of_birth_distinct_emails_4w', 'employment_status',
             'credit_risk_score', 'email_is_free', 'housing_status',
             'phone_home_valid', 'phone_mobile_valid', 'bank_months_count',
             'has_other_cards', 'proposed_credit_limit', 'foreign_request', 'source',
             'session_length_in_minutes', 'device_os', 'keep_alive_session',
             'device_distinct_emails_8w', 'device_fraud_count', 'month']

    df_inverse = pd.DataFrame(X_inverse, columns=names)

    # Обратное преобразование LabelEncoding
    for i in ['payment_type', 'employment_status', 'housing_status', 'source', 'device_os']:
        with open(f'LE_{i}_BAF.pkl', 'rb') as fp:
            le = pickle.load(fp)
        df_inverse[i] = le.inverse_transform(df_inverse[i].astype(int))

    return pd.concat((df_inverse, y), axis=1)
