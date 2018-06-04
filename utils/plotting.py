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
