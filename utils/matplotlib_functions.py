"""
Generic matplotlib functions
"""

import matplotlib
import matplotlib.pyplot as plt

plt.style.use('ggplot')


def single_ax_plot(df, x, y, figsize=(10, 10), **kwargs):
    f, a = plt.subplots(figsize=figsize)
    df.plot(x=x, y=y, ax=a, **kwargs)
    return f


