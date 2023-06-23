from pathlib import Path

import numpy as np
import pandas as pd
import torch
from torch import nn
from torch.utils.data import TensorDataset, DataLoader


SAMPLE_SIZE = 30
OUTPUT_SIZE = 1
DEVICE = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")

class Our_Model_2(nn.Module):
    def __init__(self, input_size, output_size):
        super(Our_Model_2, self).__init__()

        # Энкодер
        self.ae = nn.Sequential(
            nn.Linear(input_size, 512),
            nn.BatchNorm1d(512, momentum=0.1),
            nn.ReLU(),

            nn.Linear(512, 64),
            nn.BatchNorm1d(64, momentum=0.1),
            nn.ReLU(),

            nn.Linear(64, 16),
            nn.BatchNorm1d(16, momentum=0.1),
            nn.ReLU(),

            nn.Linear(16, 64),
            nn.BatchNorm1d(64, momentum=0.1),
            nn.ReLU(),

            nn.Linear(64, 512),
            nn.BatchNorm1d(512, momentum=0.6),
            nn.ReLU(),

            nn.Linear(512, input_size)
        )

        # Добавим нормализацию к исходным данным
        self.norm = nn.Sequential(
            nn.Linear(input_size, input_size),
            nn.BatchNorm1d(input_size, momentum=0.1)
        )

        # Определим голову для обучения классификатора
        self.classifier_head = nn.Sequential(
            nn.Linear(input_size * 2, 16),
            nn.BatchNorm1d(16, momentum=0.8),
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
        x_hidden = torch.cat((x_norm, x_d), dim=1)
        # Обучаем классификатор
        clf_head = self.classifier_head(x_hidden)
        return clf_head.view(-1)

    def predict(self, x):
        y_logit = self.forward(x)
        y_probs = self.sigm(y_logit)

        return y_probs

#
# class Generator(nn.Module):
#
#     def __init__(self, n_inputs, n_outputs):
#         super(Generator, self).__init__()
#
#         self.net = nn.Sequential(
#             nn.Linear(n_inputs, 1000),
#             nn.ReLU(),
#
#             nn.Linear(1000, 500),
#             nn.BatchNorm1d(500),
#             nn.Dropout(0.2),
#             nn.ReLU(),
#
#             nn.Linear(500, n_outputs)
#         )
#
#     def forward(self, z, y):
#         zy = torch.cat((z, y), dim=1)
#         return self.net(zy)


model = Our_Model_2(SAMPLE_SIZE, OUTPUT_SIZE).to(DEVICE)
model.eval()


def load_model(path_to_model: Path) -> None:
    model.load_state_dict(torch.load(path_to_model))
    model.eval()


def run_model(df: pd.DataFrame, threshold: float = 0.48) -> list[int]:
    dataset = TensorDataset(torch.tensor(    df.iloc[:, :-1].values.astype(np.float32),
                                             ))
    dataloader = DataLoader(dataset, batch_size=df.shape[0], shuffle=False)

    x_batch = next(iter(dataloader))[0].to(DEVICE)
    pred = model.predict(x_batch).detach().cpu().numpy()
    binary_pred = np.where(pred > threshold, 1, 0)
    return binary_pred.tolist()

