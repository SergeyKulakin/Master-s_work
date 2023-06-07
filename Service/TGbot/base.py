import torch
from torch import nn
import numpy as np
import re
from sklearn. metrics import matthews_corrcoef

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

class metrics():
    """Класс для расчета метрик"""

    def show_metrics(y_true, y_score):
        """
        INPUT: y_true (0-norm, 1-fraud),
               y_score (predicted results: 0-norm, 1-fraud)
        UTPUT:
                print: tpr
                        fpr
                        precision
                        recall
                        tnr
                        auc
                        f1
                        mcc
                        g_mean
        """
        # True positive
        tp = np.sum(y_true * y_score)
        # False positive
        fp = np.sum((y_true == 0) * y_score)
        # True negative
        tn = np.sum((y_true==0) * (y_score==0))
        # False negative
        fn = np.sum(y_true * (y_score==0))

        # True positive rate (sensitivity or recall)
        tpr = tp / (tp + fn)
        # False positive rate (fall-out)
        fpr = fp / (fp + tn)
        # Precision
        precision = tp / (tp + fp)
        # Recall
        recall = tp / (tp + fn)
        # True negatvie rate (specificity)
        tnr = 1 - fpr
        # F1 score
        f1 = 2*tp / (2*tp + fp + fn)
        # ROC-AUC for binary classification
        auc = (tpr+tnr) / 2

        # MCC
        mcc = matthews_corrcoef(y_true, y_score)
        #mcc = (tp * tn - fp * fn) / np.sqrt((tp + fp) * (tp + fn) * (tn + fp) * (tn + fn))
        # G-mean
        g_mean = np.sqrt((tp/(tp+fn)) * (tn/(fp+tn)))

        #print("True positive: ", tp)
        #print("False positive: ", fp)
        #print("True negative: ", tn)
        #print("False negative: ", fn)

        print("True positive rate (recall): ", tpr)
        print("False positive rate: ", fpr)
        print("Precision: ", precision)
        print("Recall: ", recall)
        print("True negative rate: ", tnr)
        print("ROC-AUC: ", auc)
        print("F1: ", f1)
        print("MCC: ", mcc)
        print("G-mean: ", g_mean)

    def write_metrics(y_true, y_score):
        """
        INPUT: y_true (0-norm, 1-fraud),
               y_score (predicted results: 0-norm, 1-fraud)
        
        UTPUT:          Recall_not_fraud
                        Recall_fraud
                        auc
                        f1
                        mcc
                        g_mean
        """
        # True positive
        tp = np.sum(y_true * y_score)
        # False positive
        fp = np.sum((y_true == 0) * y_score)
        # True negative
        tn = np.sum((y_true==0) * (y_score==0))
        # False negative
        fn = np.sum(y_true * (y_score==0))

        # True positive rate (sensitivity or recall)
        tpr = tp / (tp + fn)
        # False positive rate (fall-out)
        fpr = fp / (fp + tn)
        # Precision
        precision = tp / (tp + fp)
        
        # Recall_fraud
        recall_f = tpr #tp / (tp + fn)
        # Recall_not_fraud
        recall_nf = 1 - fpr
        
        # True negatvie rate (specificity)
        tnr = 1 - fpr
        # F1 score
        f1 = 2*tp / (2*tp + fp + fn)
        # ROC-AUC for binary classification
        auc = (tpr+tnr) / 2

        # MCC
        mcc = matthews_corrcoef(y_true, y_score)
        #mcc = (tp * tn - fp * fn) / np.sqrt((tp + fp) * (tp + fn) * (tn + fp) * (tn + fn))
        # G-mean
        g_mean = np.sqrt((tp/(tp+fn)) * (tn/(fp+tn)))

        return recall_nf, recall_f, auc, f1, mcc, g_mean

    # Metrics
    def mcc(y_true, y_score)-> float:
        """
        INPUT: y_true (0-norm, 1-fraud),
               y_score (predicted results: 0-norm, 1-fraud)
        OUTPUT:  float -> mcc (коэффициент корреляции Мэтьюса)
        """
        # True positive
        tp = np.sum(y_true * y_score)
        # False positive
        fp = np.sum((y_true == 0) * y_score)
        # True negative
        tn = np.sum((y_true==0) * (y_score==0))
        # False negative
        fn = np.sum(y_true * (y_score==0))

        # MCC
        mcc = (tp * tn - fp * fn) / np.sqrt((tp + fp) * (tp + fn) * (tn + fp) * (tn + fn))
        
        return mcc
    
    def g_mean(y_true, y_score)-> float:
        """
        INPUT: y_true (0-norm, 1-fraud),
               y_score (predicted results: 0-norm, 1-fraud)
        OUTPUT:  float -> mcc (коэффициент корреляции Мэтьюса)
        """
        # True positive
        tp = np.sum(y_true * y_score)
        # False positive
        fp = np.sum((y_true == 0) * y_score)
        # True negative
        tn = np.sum((y_true==0) * (y_score==0))
        # False negative
        fn = np.sum(y_true * (y_score==0))

        # g_mean
        g_mean = np.sqrt((tp/(tp+fn)) * (tn/(fp+tn)))
        
        return g_mean


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


def check_num(string):
    pattern = re.compile("[0][\.\,][0-9][0-9]")
    if pattern.fullmatch(string):
        return True
    else:
        return False

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