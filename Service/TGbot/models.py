import pickle
import pandas as pd
import numpy as np
# Поместим эту функцию в спомогательный файл
from base import metrics, baf_prep_gan
import prettytable as pt
import seaborn as sns

import torch
from base import Our_Model_2, Custom_1_BAF, Custom_3_BAF, Generator, Generator_BAF, generate, frac_gen
from torch.utils.data import TensorDataset, DataLoader


# Load data
ccf = pd.read_csv('CCF_VAL.csv')
ccf_X, ccf_y = ccf.iloc[:, :-1], ccf.iloc[:, -1]
# Load data
baf = pd.read_csv('BAF_VAL.csv')
baf_X, baf_y = baf.iloc[:, :-1], baf.iloc[:, -1]

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


# Загрузим модель Custom_1 CCF
sample_size = 30
output_size = 1
custom_model = Our_Model_2(sample_size, output_size).to(DEVICE)
custom_model.load_state_dict(torch.load('Custom.pth'))
custom_model.eval()

ccf_dataset = TensorDataset(
              torch.tensor(ccf_X.values.astype(np.float32)),
              torch.tensor(ccf_y.values.astype(np.float32)))
# Даталодер
ccf_loader = DataLoader(ccf_dataset, batch_size=ccf_X.shape[0], shuffle=False)

baf_dataset = TensorDataset(
              torch.tensor(baf_X.values.astype(np.float32)),
              torch.tensor(baf_y.values.astype(np.float32)))
baf_loader = DataLoader(baf_dataset, batch_size=7593, shuffle=False)

# Загрузим модель Custom_1 BAF
custom_1_baf = Custom_1_BAF(61, 1).to(DEVICE)
custom_1_baf.load_state_dict(torch.load('Custom_1_BAF.pth'))
custom_1_baf.eval()

# Загрузим модель Custom_1 BAF
custom_3_baf = Custom_3_BAF(61, 1).to(DEVICE)
custom_3_baf.load_state_dict(torch.load('Custom_3_BAF.pth'))
custom_3_baf.eval()

def ccf_get_prediction(model, d_loader):
    threshold = 0.48
    for batch in d_loader:
        x_batch = batch[0].to(DEVICE)
        clf = model.predict(x_batch).detach().cpu().numpy()

    clf[clf > threshold] = 1
    clf[clf <= threshold] = 0

    return clf


def baf_get_prediction(model, d_loader):
    threshold = 0.5
    res = np.array([])
    first = True
    for batch in d_loader:
        x_batch = batch[0].to(DEVICE)
        clf = model.predict(x_batch).detach().cpu().numpy()
        # print(clf.shape)
        if first:
            res = clf
            first = False
        else:
            res = np.hstack((res, clf))

    res[res > threshold] = 1
    res[res <= 0.5] = 0

    return res

def pred_array(pred):
    """Функция predict lgbm"""
    res = []
    for i in pred:
        if i >= 0.8:
            res.append(1)
        else:
            res.append(0)
    return np.array(res)


def predict_real(model):

    if model == 'LGBM':
        with open('LGBM.pickle', 'rb') as f:
            m = pickle.load(f)

        lgbm = m.predict(ccf_X)
        test_model_pred = pred_array(lgbm)
        #print(test_model_pred.shape)
        recall_nf, recall_f, auc, f1, mcc, g_mean = metrics.write_metrics(ccf_y, test_model_pred)
        tmp = {'Recall_not_fraud': round(recall_nf, 4),
               'Recall_fraud____': round(recall_f, 4),
               'AUC___________': round(auc, 4),
               'F1_____________': round(f1, 4),
               'MCC___________': round(mcc, 4),
               'G_mean________': round(g_mean, 4)}

        table = pt.PrettyTable(['Metrics names', 'Values'])
        table.align['Metrics names'] = 'l'
        table.align['Values'] = 'l'
        for key, val in tmp.items():
            table.add_row([key, f'{val:.4f}'])

    elif model == 'Custom_1_ccf':
        test_model_pred = ccf_get_prediction(custom_model, ccf_loader)
        recall_nf, recall_f, auc, f1, mcc, g_mean = metrics.write_metrics(ccf_y, test_model_pred)
        tmp = {'Recall_not_fraud': round(recall_nf, 4),
               'Recall_fraud____': round(recall_f, 4),
               'AUC___________': round(auc, 4),
               'F1_____________': round(f1, 4),
               'MCC___________': round(mcc, 4),
               'G_mean________': round(g_mean, 4)}

        table = pt.PrettyTable(['Metrics names', 'Values'])
        table.align['Metrics names'] = 'l'
        table.align['Values'] = 'l'
        for key, val in tmp.items():
            table.add_row([key, f'{val:.4f}'])

    elif model == 'SK':
        with open('StackingClassifier.pickle', 'rb') as f:
            m = pickle.load(f)

        test_model_pred = m.predict(baf_X.values)
        recall_nf, recall_f, auc, f1, mcc, g_mean = metrics.write_metrics(baf_y, test_model_pred)
        tmp = {'Recall_not_fraud': round(recall_nf, 4),
               'Recall_fraud____': round(recall_f, 4),
               'AUC___________': round(auc, 4),
               'F1_____________': round(f1, 4),
               'MCC___________': round(mcc, 4),
               'G_mean________': round(g_mean, 4)}

        table = pt.PrettyTable(['Metrics names', 'Values'])
        table.align['Metrics names'] = 'l'
        table.align['Values'] = 'l'
        for key, val in tmp.items():
            table.add_row([key, f'{val:.4f}'])

    elif model == 'Custom_1_baf':
        test_model_pred = baf_get_prediction(custom_1_baf, baf_loader)
        recall_nf, recall_f, auc, f1, mcc, g_mean = metrics.write_metrics(baf_y, test_model_pred)
        tmp = {'Recall_not_fraud': round(recall_nf, 4),
               'Recall_fraud____': round(recall_f, 4),
               'AUC___________': round(auc, 4),
               'F1_____________': round(f1, 4),
               'MCC___________': round(mcc, 4),
               'G_mean________': round(g_mean, 4)}

        table = pt.PrettyTable(['Metrics names', 'Values'])
        table.align['Metrics names'] = 'l'
        table.align['Values'] = 'l'
        for key, val in tmp.items():
            table.add_row([key, f'{val:.4f}'])

    return table

# Генератор CCF
latent_dim=100

generator_ccf = Generator(n_inputs=latent_dim+1, n_outputs=30)
generator_ccf.load_state_dict(torch.load('CCF_GAN.pth'))
generator_ccf.eval()
generator_ccf.to(DEVICE)

# Генератор BAF
generator_baf = Generator_BAF(n_inputs=33, n_outputs=31)
latent_dim_baf=32
generator_baf.load_state_dict(torch.load('GAN_BAF.pth'))
generator_baf.eval()
generator_baf.to(DEVICE)

def gen_data(frac):
    y = frac_gen(frac)
    X_gan = generate(generator_ccf, y, latent_dim)
    with open('QT_CCF.pkl', 'rb') as fp:
        qt = pickle.load(fp)
    X = qt.inverse_transform(X_gan)
    return X, y

def gen_data_baf(frac):
    y = frac_gen(frac)
    X_gan = generate(generator_baf, y, latent_dim_baf)

    # Обратное квантильное преобразование
    with open('QT_BAF.pkl', 'rb') as f:
        qt = pickle.load(f)

    X_inverse = qt.inverse_transform(X_gan)

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

    return df_inverse, y

def plot_report(X, y, predict, id, label='ccf'):

    if label == 'ccf':
        a = X[:, -1]
    else:
        a = X.iloc[:, 0]
    b = y.astype(int)
    c = predict.astype(int)

    nm = {'Amount': a, 'Class': b, 'Predict': c}
    data = pd.DataFrame(nm)
    TP = data[(data['Class'] == 1) & ((data['Predict'] == 1))]
    TN = data[(data['Class'] == 0) & ((data['Predict'] == 0))]
    FP = data[(data['Class'] == 0) & ((data['Predict'] == 1))]
    FN = data[(data['Class'] == 1) & ((data['Predict'] == 0))]

    data.loc[TP.index, ['Label']] = 'Заблокированные \nмошенники'  # 'TP'
    data.loc[TN.index, ['Label']] = 'Одобренные \nнормальные'  # 'TN'
    data.loc[FP.index, ['Label']] = 'Заблокированные \nнормальные'  # 'FP'
    data.loc[FN.index, ['Label']] = 'Одобренные \nмошенники'  # 'FN'

    g = sns.catplot(data=data, x="Label", kind="count", height=5, aspect=1.4,
                    order=['Одобренные \nнормальные',
                           'Одобренные \nмошенники',
                           'Заблокированные \nнормальные',
                           'Заблокированные \nмошенники'],
                    hue_order=['blue', 'red', 'orange', 'green'],
                    palette=sns.color_palette(['#247EC3', '#FF4B2F', '#FFAC2F', '#1C8920'])
                    )
    g.set_axis_labels("Действия с транзакциями", "Количество транзакций")
    #g.set_xticklabels(["Одобренные \nнормальные", "Заблокированные \nмошенники",
    #                   "Пропущенные \nмошенники", "Заблокированные \nнормальные"]);

    g.savefig(f"plot_{id}.png")

    keys = {'Одобренные нормальные_____': TN['Amount'].sum(),
            'Одобренные мошенники_____': FN['Amount'].sum(),
            'Заблокированные мошенники_': TP['Amount'].sum(),
            'Заблокированные нормальные': FP['Amount'].sum()}

    table = pt.PrettyTable(['Действие с транзакциями', 'Денежная сумма'])
    table.align['Действие с транзакциями'] = 'l'
    table.align['Денежная сумма'] = 'r'
    for key, val in keys.items():
        table.add_row([key, f'{val:.1f}'])

    return table



def predict_gan(model, X, y):
    ccf_gan = TensorDataset(
        torch.tensor(X.astype(np.float32)),
        torch.tensor(y.astype(np.float32)))
    # Даталодер
    gan_loader_ccf = DataLoader(ccf_gan, batch_size=X.shape[0], shuffle=False)


    if model == 'LGBM':
        with open('LGBM.pickle', 'rb') as f:
            m = pickle.load(f)

        lgbm = m.predict(X)
        test_model_pred = pred_array(lgbm)
        recall_nf, recall_f, auc, f1, mcc, g_mean = metrics.write_metrics(y, test_model_pred)
        tmp = {'Recall_not_fraud': recall_nf,
               'Recall_fraud____': recall_f,
               'AUC___________': auc,
               'F1_____________': f1,
               'MCC___________': mcc,
               'G_mean________': g_mean}

        table = pt.PrettyTable(['Metrics names', 'Values'])
        table.align['Metrics names'] = 'l'
        table.align['Values'] = 'l'
        for key, val in tmp.items():
            table.add_row([key, f'{val:.4f}'])

    elif model == 'Stacking Classifier':
        with open('StackingClassifier_ccf.pickle', 'rb') as f:
            m = pickle.load(f)

        test_model_pred = m.predict(X)
        recall_nf, recall_f, auc, f1, mcc, g_mean = metrics.write_metrics(y, test_model_pred)
        tmp = {'Recall_not_fraud': round(recall_nf, 4),
               'Recall_fraud____': round(recall_f, 4),
               'AUC___________': round(auc, 4),
               'F1_____________': round(f1, 4),
               'MCC___________': round(mcc, 4),
               'G_mean________': round(g_mean, 4)}

        table = pt.PrettyTable(['Metrics names', 'Values'])
        table.align['Metrics names'] = 'l'
        table.align['Values'] = 'l'
        for key, val in tmp.items():
            table.add_row([key, f'{val:.4f}'])

    elif model == 'Custom_1':
        test_model_pred = ccf_get_prediction(custom_model, gan_loader_ccf)
        recall_nf, recall_f, auc, f1, mcc, g_mean = metrics.write_metrics(y, test_model_pred)
        tmp = {'Recall_not_fraud': round(recall_nf, 4),
               'Recall_fraud____': round(recall_f, 4),
               'AUC___________': round(auc, 4),
               'F1_____________': round(f1, 4),
               'MCC___________': round(mcc, 4),
               'G_mean________': round(g_mean, 4)}

        table = pt.PrettyTable(['Metrics names', 'Values'])
        table.align['Metrics names'] = 'l'
        table.align['Values'] = 'l'
        for key, val in tmp.items():
            table.add_row([key, f'{val:.4f}'])


    return table, test_model_pred

def predict_gan_baf(model, X, y):

    # X- pd.DataFrame
    # X_pre np.array - prepare data

    X_pre = baf_prep_gan(X)
    baf_gan = TensorDataset(
        torch.tensor(X_pre.astype(np.float32)),
        torch.tensor(y.astype(np.float32)))
    # Даталодер
    gan_loader_baf = DataLoader(baf_gan, batch_size=7593, shuffle=False)

    if model == 'Custom_1':
        test_model_pred = baf_get_prediction(custom_1_baf, gan_loader_baf)
        recall_nf, recall_f, auc, f1, mcc, g_mean = metrics.write_metrics(y, test_model_pred)
        tmp = {'Recall_not_fraud': round(recall_nf, 4),
               'Recall_fraud____': round(recall_f, 4),
               'AUC___________': round(auc, 4),
               'F1_____________': round(f1, 4),
               'MCC___________': round(mcc, 4),
               'G_mean________': round(g_mean, 4)}

        table = pt.PrettyTable(['Metrics names', 'Values'])
        table.align['Metrics names'] = 'l'
        table.align['Values'] = 'l'
        for key, val in tmp.items():
            table.add_row([key, f'{val:.4f}'])

    elif model == 'Custom_3':
        test_model_pred = baf_get_prediction(custom_3_baf, gan_loader_baf)
        recall_nf, recall_f, auc, f1, mcc, g_mean = metrics.write_metrics(y, test_model_pred)
        tmp = {'Recall_not_fraud': round(recall_nf, 4),
               'Recall_fraud____': round(recall_f, 4),
               'AUC___________': round(auc, 4),
               'F1_____________': round(f1, 4),
               'MCC___________': round(mcc, 4),
               'G_mean________': round(g_mean, 4)}

        table = pt.PrettyTable(['Metrics names', 'Values'])
        table.align['Metrics names'] = 'l'
        table.align['Values'] = 'l'
        for key, val in tmp.items():
            table.add_row([key, f'{val:.4f}'])

    elif model == 'Stacking Classifier':
        with open('StackingClassifier.pickle', 'rb') as f:
            m = pickle.load(f)

        test_model_pred = m.predict(X_pre)
        recall_nf, recall_f, auc, f1, mcc, g_mean = metrics.write_metrics(y, test_model_pred)
        tmp = {'Recall_not_fraud': round(recall_nf, 4),
               'Recall_fraud____': round(recall_f, 4),
               'AUC___________': round(auc, 4),
               'F1_____________': round(f1, 4),
               'MCC___________': round(mcc, 4),
               'G_mean________': round(g_mean, 4)}

        table = pt.PrettyTable(['Metrics names', 'Values'])
        table.align['Metrics names'] = 'l'
        table.align['Values'] = 'l'
        for key, val in tmp.items():
            table.add_row([key, f'{val:.4f}'])


    return table, test_model_pred