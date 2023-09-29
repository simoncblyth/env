#!/usr/bin/env python

import os
from sklearn.datasets import load_digits
digits = load_digits()
digits.images.shape


import matplotlib.pyplot as plt

if "PLOT_DIGITS" in os.environ:
    fig, axes = plt.subplots(10, 10, figsize=(8, 8),
                             subplot_kw={'xticks':[], 'yticks':[]},
                             gridspec_kw=dict(hspace=0.1, wspace=0.1))

    for i, ax in enumerate(axes.flat):
        ax.imshow(digits.images[i], cmap='binary', interpolation='nearest')
        ax.text(0.05, 0.05, str(digits.target[i]),transform=ax.transAxes, color='green')
    pass
    fig.show()



X = digits.data

y = digits.target

from sklearn.manifold import Isomap
iso = Isomap(n_components=2)
iso.fit(digits.data)
data_projected = iso.transform(digits.data)
data_projected.shape



#if "PLOT_PROJECTED" in os.environ:

if True:
    fig, ax = plt.subplots(1, figsize=(8,8))

    ax.scatter(data_projected[:, 0], data_projected[:, 1], c=digits.target,
                edgecolor='none', alpha=0.5,
                cmap=plt.cm.get_cmap('viridis', 10))
    #ax.colorbar(label='digit label', ticks=range(10))
    #ax.clim(-0.5, 9.5);

    fig.show()




