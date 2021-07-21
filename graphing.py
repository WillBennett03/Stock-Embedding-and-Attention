"""
################
### graphing.py ###
################

~ Will Bennett 12/07/2021

"""
import tensorflow as tf
from sklearn.manifold import TSNE
from collections import Counter, OrderedDict
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
plt.style.use('fivethirtyeight')
plt.rcParams['font.size'] = 15

def reduce_dim(weights, components = 3, method = 'tsne'):
    """Reduce dimensions of embeddings"""
    if method == 'tsne':
        return TSNE(components, metric = 'cosine').fit_transform(weights)

def count_items(l):
    """Return ordered dictionary of counts of objects in `l`"""
    
    counts = Counter(l)
    
    # Sort by highest count first and place in ordered dictionary
    counts = sorted(counts.items(), key = lambda x: x[1], reverse = True)
    counts = OrderedDict(counts)
    
    return counts

def show_embedding(model, filename, sect):
    model.load_weights(filename)

    Stock_layer = model.get_layer('Stock_embedding')
    Stock_weights = Stock_layer.get_weights()[0]
    Time_layer = model.get_layer('Time_embedding')
    Time_weights = Time_layer.get_weights()[0]

    Stock_weights = Stock_weights / np.linalg.norm(Stock_weights, axis = 1).reshape((-1, 1))
    Stock_weights[0][:10]
    Stocks_r = reduce_dim(Stock_weights, components=2, method='tsne')

    sect_count = count_items(sect)
    sect_summary = list(sect_count.keys())
    idx_include = []
    sects = []

    for i, sector in enumerate(sect):
        idx_include.append(i)
        sects.append(sector.capitalize())

    ints, gen = pd.factorize(sects)

    plt.figure(figsize = (10, 8))

    # Plot embedding
    plt.scatter(Stocks_r[idx_include, 0], Stocks_r[idx_include, 1], 
                c = ints, cmap = plt.cm.tab20)

    # Add colorbar and appropriate labels
    cbar = plt.colorbar()
    cbar.set_ticks([])
    for j, lab in enumerate(gen):
        cbar.ax.text(11, (10 * j + 94) / ((10) * 2), lab, ha='left', va='center')
    cbar.ax.set_title('Stock', loc = 'left')


    plt.xlabel('TSNE 1'); plt.ylabel('TSNE 2'); plt.title('TSNE Visualization of Stock Embeddings');
    plt.show()

def show_control_embedding(model, filename, sect):
    model.load_weights(filename)

    Stock_layer = model.get_layer('Stock_embedding')
    Stock_weights = Stock_layer.get_weights()[0]

    Stock_weights = Stock_weights / np.linalg.norm(Stock_weights, axis = 1).reshape((-1, 1))
    Stock_weights[0][:10]
    Stocks_r = reduce_dim(Stock_weights, components=2, method='tsne')

    sect_count = count_items(sect)
    sect_summary = list(sect_count.keys())
    idx_include = []
    sects = []

    for i, sector in enumerate(sect):
        idx_include.append(i)
        sects.append(sector.capitalize())

    ints, gen = pd.factorize(sects)

    plt.figure(figsize = (10, 8))

    # Plot embedding
    plt.scatter(Stocks_r[idx_include, 0], Stocks_r[idx_include, 1], 
                c = ints, cmap = plt.cm.tab20)

    # Add colorbar and appropriate labels
    cbar = plt.colorbar()
    cbar.set_ticks([])
    for j, lab in enumerate(gen):
        cbar.ax.text(11, (10 * j + 94) / ((10) * 2), lab, ha='left', va='center')
    cbar.ax.set_title('Stock', loc = 'left')


    plt.xlabel('TSNE 1'); plt.ylabel('TSNE 2'); plt.title('TSNE Visualization of Stock Embeddings');
    plt.show()