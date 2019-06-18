import numpy as np
import random

import seaborn as sns
import matplotlib.pyplot as plt
from . import fit_functions_ic as fitf

from typing import Tuple, Optional

from . stat_functions  import mean_and_std
from . kr_types import Number, Array, Str
from . kr_types        import PlotLabels

from   krcal.core.histo_functions                 import h1, h1d, plot_histo
from   invisible_cities.core.system_of_units_c import units

def plot_s1_and_s2(dst):

    #sns.set_style('whitegrid')
    sns.set_style("white")
    sns.set_style("ticks")
    
    fig = plt.figure(figsize=(15,6))
    ax      = fig.add_subplot(1, 2, 1)
    n, b, _, _  = h1(dst.nS1.values, bins = 10, range = (0,10), histtype='stepfilled', color='steelblue',
                     lbl='nS1')
    plot_histo(PlotLabels('nS1','Entries',''), ax)
    
    ax      = fig.add_subplot(1, 2, 2)
    (_) = h1(dst.nS2.values, bins = 10, range = (0,10), histtype='stepfilled', color='crimson',
             lbl='nS2')
    plot_histo(PlotLabels('nS2','Entries',''), ax)

    return n

def plot_e_before_after_sel(dst, dstsel):

    fig = plt.figure(figsize=(14,6))

    ax      = fig.add_subplot(1, 2, 1)
    (_) = h1(dst.S2e, bins = 100, range = (500,5000), histtype='stepfilled', color='crimson',
             lbl='E (pes)')
    plot_histo(PlotLabels('E (pes)','Entries',''), ax)
    plt.legend(loc='upper left')
    
    ax      = fig.add_subplot(1, 2, 2)
    (_) = h1(dstsel.S2e, bins = 100, range = (500,5000), histtype='stepfilled', color='crimson',
             lbl='E (pes)')
    plot_histo(PlotLabels('E (pes)','Entries',''), ax)
    plt.legend(loc='upper left')




def plot_s1_variables(dst):

    fig = plt.figure(figsize=(15,15))

    ax      = fig.add_subplot(3, 2, 1)
    (_) = h1(dst.S1e, bins = 100, range = (0,40), histtype='stepfilled', color='steelblue',
             lbl='')
    plot_histo(PlotLabels('E (pes)','Entries',''), ax)
    plt.legend(loc='upper right')
    
    ax      = fig.add_subplot(3, 2, 2)
    (_) = h1(dst.S1w, bins = 20, range = (0,500), histtype='stepfilled', color='steelblue',
             lbl='')
    plot_histo(PlotLabels('Width (mus)','Entries',''), ax)
    plt.legend(loc='upper right')
    
    ax      = fig.add_subplot(3, 2, 3)
    (_) = h1(dst.S1h, bins = 100, range = (0,10), histtype='stepfilled', color='steelblue',
             lbl='')
    plot_histo(PlotLabels('height/s1energy','Entries',''), ax)
    plt.legend(loc='upper right')
    
    ax      = fig.add_subplot(3, 2, 4)
    (_) = h1(dst.S1h/dst.S1e, bins = 100, range = (0,0.6), histtype='stepfilled', color='steelblue',
             lbl='')
    plot_histo(PlotLabels('height/s1energy','Entries',''), ax)
    plt.legend(loc='upper right')
    
    ax      = fig.add_subplot(3, 2, 5)
    (_) = h1(dst.S1t/units.mus, bins = 20, range = (0,400), histtype='stepfilled', color='steelblue',
             lbl='')
    plot_histo(PlotLabels('S1 time mus','Entries',''), ax)
    plt.legend(loc='upper right')
    
    ax      = fig.add_subplot(3, 2, 6)
    plt.hist2d(dst.S1t/units.mus, dst.S1e, bins=10, range=[[0, 500],[0, 30]])
    plt.colorbar()
    ax.set_xlabel(r'S1 time ($\mu$s) ',fontsize = 11) #xlabel
    ax.set_ylabel('S1 height (pes)', fontsize = 11)
    plt.grid(True)
    

def plot_s2_variables(dst):

    fig = plt.figure(figsize=(14,14))

    ax      = fig.add_subplot(3, 2, 1)
    (_) = h1(dst.S2e, bins = 100, range = (1000,5000), histtype='stepfilled', color='crimson',
             lbl='')
    plot_histo(PlotLabels('E (pes)','Entries',''), ax)
    plt.legend(loc='upper left')
    
    ax      = fig.add_subplot(3, 2, 2)
    (_) = h1(dst.S2w, bins = 20, range = (0,20), histtype='stepfilled', color='crimson',
             lbl='')
    plot_histo(PlotLabels('Width (mus)','Entries',''), ax)
    plt.legend(loc='upper right')
    
    ax      = fig.add_subplot(3, 2, 3)
    (_) = h1(dst.S2q, bins = 100, range = (0,350), histtype='stepfilled', color='crimson',
             lbl='')
    plot_histo(PlotLabels('Q (pes)','Entries',''), ax)
    plt.legend(loc='upper right')
    
    ax      = fig.add_subplot(3, 2, 4)
    (_) = h1(dst.Nsipm, bins = 25, range = (0,25), histtype='stepfilled', color='crimson',
             lbl='')
    plot_histo(PlotLabels('num Sipm','Entries',''), ax)
    plt.legend(loc='upper right')
    
    ax      = fig.add_subplot(3, 2, 5)
    (_) = h1(dst.X, bins = 20, range = (-80,80), histtype='stepfilled', color='crimson',
             lbl='')
    plot_histo(PlotLabels('X (mm)','Entries',''), ax)
    plt.legend(loc='upper right')
    
    ax      = fig.add_subplot(3, 2, 6)
    (_) = h1(dst.Y, bins = 20, range = (-80,80), histtype='stepfilled', color='crimson',
             lbl='')
    plot_histo(PlotLabels('Y (mm)','Entries',''), ax)
    plt.legend(loc='upper right')


def plot_Z_DT_variables(dst):

    fig = plt.figure(figsize=(14,6))
    ax      = fig.add_subplot(1, 2, 1)
    (_) = h1(dst.Z, bins = 100, range = (0,400), histtype='stepfilled', color='crimson',
             lbl='')
    plot_histo(PlotLabels('Z (mm)','Entries',''), ax)
    plt.legend(loc='upper left')
    
    ax      = fig.add_subplot(1, 2, 2)
    (_) = h1(dst.DT, bins = 100, range = (0,400), histtype='stepfilled', color='crimson',
             lbl='')
    plot_histo(PlotLabels('Drift time (mus)','Entries',''), ax)
    plt.legend(loc='upper left')
    
