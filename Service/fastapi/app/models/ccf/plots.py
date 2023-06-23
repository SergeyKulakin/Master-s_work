import pandas as pd
from matplotlib import pyplot as plt


def make_hists(df: pd.DataFrame, limit: int = 100) -> plt.Figure:
    df = df.iloc[:, :-1]  # удаляем последний столбец
    df = df[:limit]
    fig, axs = plt.subplots(10, 3, figsize=(15, 40))
    for i, ax in enumerate(axs.flat):
        ax.hist(df.iloc[:, i])
        ax.set_title(df.columns[i], fontsize=18, fontweight='bold')
        ax.tick_params(axis='y', labelsize=14)
        ax.set_xlabel('Значения', fontsize=15)
        ax.set_ylabel('Частота', fontsize=15)
    fig.tight_layout()
    return fig


def make_count_frauds(df: pd.DataFrame) -> plt.Figure:
    df["day"] = (df["Time"] / 60 / 24).astype(int)
    #n = df.loc[:, ['day', 'Class']].groupby('day')['Class'].sum()
    n = df.loc[:, ['day', 'Fraud']].groupby('day')['Fraud'].sum()
    fig, ax = plt.subplots()
    ax.plot(n)
    ax.set_title('График количества fraud в зависимости от дня')
    ax.set_xlabel('День')
    ax.set_ylabel('Сумма fraud')
    fig.tight_layout()
    return fig
