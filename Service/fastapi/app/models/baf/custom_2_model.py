from pathlib import Path

import numpy as np
from sklearn.preprocessing import OneHotEncoder, Normalizer
from torch import nn
import torch
import pandas as pd
from torch.utils.data import TensorDataset, DataLoader

OUTPUT_SIZE = 1
SAMPLE_SIZE = 61 # FIXME: изменить
DEVICE = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")


class Our_Model_2(nn.Module):
    def __init__(self, input_size, output_size):
        super(Our_Model_2, self).__init__()

        # Энкодер
        self.ae = nn.Sequential(
            nn.Linear(input_size, 128),
            nn.ReLU(),

            nn.Linear(128, 16),
            nn.ReLU(),

            nn.Linear(16, 128),
            nn.ReLU(),

            nn.Linear(128, input_size),
            nn.BatchNorm1d(input_size, momentum=0.4),
        )

        # Добавим нормализацию к исходным данным
        self.norm = nn.Sequential(
            nn.Linear(input_size, input_size),
            nn.BatchNorm1d(input_size, momentum=0.4)
        )

        # Определим голову для обучения классификатора
        self.classifier_head = nn.Sequential(
            nn.Linear(input_size*2, 16),
            nn.BatchNorm1d(16, momentum=0.4),
            nn.ReLU(),

            nn.Linear(16, output_size),
        )

        self.sigm = nn.Sigmoid()


    def forward(self, x):
        ae = self.ae(x)
        # Получаем разность Выхода и входа Автоэнкодера
        x_d = ae - x
        # Исходные данные
        x_norm = self.norm(x)

        # Конкатенируем полученную Разность и исходные данные
        x_hidden = torch.cat((x_norm, x_d),dim=1)
        # Обучаем классификатор
        clf_head = self.classifier_head(x_hidden)
        return clf_head.view(-1)

    def predict(self, x):
        y_logit = self.forward(x)
        y_probs = self.sigm(y_logit)

        return y_probs



model = Our_Model_2(SAMPLE_SIZE, OUTPUT_SIZE).to(DEVICE)
model.eval()


def load_model(path_to_model: Path) -> None:
    model.load_state_dict(torch.load(path_to_model))
    model.eval()


def run_model(df: pd.DataFrame, threshold: float = 0.48) -> list[int]:
    target = 'fraud_bool'

    new_num = ['income',
               'name_email_similarity',
               'prev_address_months_count',
               'current_address_months_count',
               'customer_age',
               'days_since_request',
               'intended_balcon_amount',
               'zip_count_4w',
               'velocity_6h',
               'velocity_24h',
               'velocity_4w',
               'bank_branch_count_8w',
               'date_of_birth_distinct_emails_4w',
               'credit_risk_score',
               'bank_months_count',
               'proposed_credit_limit',
               'session_length_in_minutes',
               'device_fraud_count',
               'month']

    need_lg_columns = ['prev_address_months_count',
                       'current_address_months_count',
                       'days_since_request',
                       'zip_count_4w',
                       'bank_branch_count_8w',
                       'date_of_birth_distinct_emails_4w',
                       'proposed_credit_limit',
                       'session_length_in_minutes']

    df.loc[:, need_lg_columns] = df.loc[:, need_lg_columns].apply(lambda x: np.log1p(x+10))

    new_cat = ['payment_type',
               'employment_status',
               'housing_status',
               'source',
               'device_os',
               'email_is_free',
               'phone_home_valid',
               'phone_mobile_valid',
               'has_other_cards',
               'foreign_request',
               'keep_alive_session',
               'device_distinct_emails_8w']

    ohe = OneHotEncoder(sparse=False, handle_unknown='ignore')
    # Get one-hot-encoded columns
    ohe_cat = pd.DataFrame(ohe.fit_transform(df[new_cat]), index=df.index)
    result = pd.concat([df[new_num], ohe_cat, df[target]], axis=1)
    result = result.iloc[:, :-1]

    transformer = Normalizer().fit(result.values)

    result = transformer.transform(result.values)


    dataset = TensorDataset(torch.tensor(    result.astype(np.float32),
                                             ))
    dataloader = DataLoader(dataset, batch_size=result.shape[0], shuffle=False)

    x_batch = next(iter(dataloader))[0].to(DEVICE)
    pred = model.predict(x_batch).detach().cpu().numpy()
    binary_pred = np.where(pred > threshold, 1, 0)
    return binary_pred.tolist()


