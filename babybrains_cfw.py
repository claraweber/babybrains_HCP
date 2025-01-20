# functions for babybrain project
# dependencies: numpy, pandas, nibabel, joypy, brainspace, brainstat, cmcrameri, nilearn, pingouin, sklearn, scipy, matplotlib

import pandas as pd
import nibabel as nib
import joypy
from brainspace.datasets import load_conte69
from brainspace.plotting import plot_hemispheres
from brainspace.gradient import GradientMaps
from brainspace.null_models import SpinPermutations
from brainstat.stats.terms import MixedEffect, FixedEffect
from brainstat.stats.SLM import SLM 
from brainstat.datasets import fetch_gradients, fetch_parcellation, fetch_yeo_networks_metadata
from cmcrameri import cm
from nilearn import plotting
from pingouin import partial_corr, pairwise_corr
from sklearn.linear_model import LinearRegression
import scipy.io as sio
from scipy.stats import spearmanr, rankdata
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap, ListedColormap
from matplotlib.cm import register_cmap
import numpy as np


def sort_to_labels(data_in, labels_in):
    """
    Sorts data in right order for labels
    
    Input: data_in = data to be sorted; shape = 552 n(cases) x 296 n(parcels)
    labels_in = 300 labels to sort data to

    Output: sorted = data sorted to 300 labels
    """
    data_in_t = data_in.T
    sorted = np.zeros((552,300))
    for i in range(300):
        try:
            sorted[:,i] = data_in_t[labels_in == i]
        except:
            pass
    return sorted

def sort_gradient_to_labels(data_in, labels_in):
    """
    same as sort_to_labels but for 1d data (e.g. gradient)
    """
    data_in_t = data_in.T
    sorted = np.zeros((300,))
    for i in range(300):
        try:
            sorted[i] = data_in_t[labels_in == i]
        except:
            pass
    return sorted

def parc2surf(data_in, babyparc):
    """
    map 300 labels to conte69 surface
    input: data in 300 labels, must be sorted (refer to above sort function)
    """
    data_out = np.zeros((64984,552))
    for i in range(300):
        data_out[babyparc ==i] = data_in[:,i]
    return data_out

def gradient_parc2surf(data_in, babyparc):
    """
    Same as parc2surf but for one-dimensional data (e.g. gradient)
    """
    data_out = np.zeros((64984,))
    for i in range(300):
        data_out[babyparc ==i] = data_in[i]
    return data_out

def joyplot_mesulam(data_in, name, mes, xmin= None, xmax= None, save = False):
    """
    plots distribution of datapoint within 4 Mesulam modalities

    data_in = datapoints to be plotted (on conte69 surface)
    name = string for plot label and file name if save == True
    xmin, xmax = range of x-axis
    save = True if figure should be saved as svg file
    """
    df_mes = pd.DataFrame({'CT': data_in, 'mes': mes})
    fig, ax = joypy.joyplot(df_mes[df_mes['mes']!=0], by = "mes", fade=False, fill=True, x_range=(xmin,xmax), color=['#ffffff','#bcace3','#8f76cf','#6306ba'], alpha=1, figsize = (5,3), title = f'{name} by Mesulam Types')
    if save:
        fig.savefig(f'joyplot_mesulam_{name}.svg', format = 'svg')

def joyplot_ek(data_in, name, ek, xmin= 0, xmax= 1000, save = False):
    """
    plots distribution of datapoint within 5 Economo-Koskinas types

    data_in = datapoints to be plotted (on conte69 surface)
    name = string for plot label and file name if save == True
    xmin, xmax = range of x-axis
    save = True if figure should be saved as svg file
    """
    colors_ek = ['#512d80','#3656ad','#6dc441','#ffb300', '#fff700']
    df_ek = pd.DataFrame({'CT': data_in, 'ek': ek})
    fig, ax = joypy.joyplot(df_ek[df_ek['ek']!=0], by = "ek", fade=False, fill=True, x_range=(xmin,xmax), color=colors_ek,
     alpha=1, figsize = (5,3), title = f'{name} by Cortical Types')
    if save:
        fig.savefig(f'joyplot_economo_{name}.svg', format = 'svg')

def joyplot_yeo7(data_in, name, yeo7, xmin= 0, xmax= 1000, save = False):
    """
    plots distribution of datapoints within 7 Yeo networks

    data_in = datapoints to be plotted (on conte69 surface)
    name = string for plot label and file name if save == True
    xmin, xmax = range of x-axis
    save = True if figure should be saved as svg file
    """
    colors_yeo_hex= ['#781286', '#4581b3', '#00760d', '#c339fa', '#dcf8a3', '#e59421', '#cd3d4d']
    df_yeo = pd.DataFrame({'CT': data_in, 'yeo': yeo7})
    fig, ax = joypy.joyplot(df_yeo[df_yeo['yeo']!=0], by = "yeo", fade=False, fill=True, x_range=(xmin,xmax), color=colors_yeo_hex,
     alpha=1, figsize = (5,3), title = f'{name} by Yeo Networks')
    if save:
        fig.savefig(f'joyplot_yeo_{name}.svg', format = 'svg')

def correlate(feat, embedding, sp, name, n_rand = 5000):
    """
    Correlates feature with embedding and calculates p-value using spin permutations
    """
    feat = np.nan_to_num(feat, 0)
    feat_rotated = np.hstack(sp.randomize(feat[:int(len(feat)/2)], feat[int(len(feat)/2):]))
    
    r_spin = np.empty(n_rand)

    mask = ~np.isnan(embedding)
    embedding = np.nan_to_num(embedding, 0)

    r_obs, pv_obs = spearmanr(feat[mask], embedding[mask])

    # permutate
    for i, perm in enumerate(feat_rotated):
        mask_rot = mask & ~np.isnan(perm)  # Remove midline
        r_spin[i] = spearmanr(perm[mask_rot], embedding[mask_rot])[0]
    pv_spin = np.mean(np.abs(r_spin) >= np.abs(r_obs))

    # Plot null dist
    plt.hist(r_spin, bins=25, density=True, alpha=0.5, color=(.8, .8, .8))
    plt.axvline(r_obs, lw=2, ls='--', color='k')

    print(f'{name.capitalize()}:\n R: {r_obs} Obs : {pv_obs:.5e} Spin: {pv_spin:.5e}\n')
    plt.show()