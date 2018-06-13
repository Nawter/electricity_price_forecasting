"""
Generic matplotlib functions
"""

import matplotlib
import matplotlib.pyplot as plt

matplotlib.rc('font', size=22)
matplotlib.rc('figure', figsize=(20, 40))


def plot_columns(df, cols=None, show=False, save_name=None):
    if cols is None:
        cols = df.columns

    f, axes  = plt.subplots(len(cols))

    for col, a in zip(cols, axes.flatten()):
        df.plot(y=col, ax=a)

    if show:
        f.show()

    if save_name:
        f.savefig(save_name)

    return f


def plot_bar(df, cols=None, show=False, save_name=None):
    if cols is None:
        cols = df.columns

    f, axes  = plt.subplots(len(cols))

    for col, a in zip(cols, axes.flatten()):
        df.plot(y=col, ax=a, kind='bar', color='blue')

    if show:
        f.show()

    if save_name:
        f.savefig(save_name)

    return f


def plot_series(series, **kwargs):
    f, a = plt.subplots(figsize=(25,5))
    series.plot(ax=a, **kwargs)
    return f


def plot_scatter(df, x, y, **kwargs):
    f, a = plt.subplots(figsize=(10, 10))
    df.plot(x=x, y=y, kind='scatter', ax=a, **kwargs)
    return f
