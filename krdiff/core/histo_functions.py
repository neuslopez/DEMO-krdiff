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

from invisible_cities.icaro. hst_functions import shift_to_bin_centers

sns.set()

def mypdf_gauss(x, mu, sigma, N):
    return N*np.exp(-0.5*np.power((x - mu)/(sigma),2))

def mypdf_gauss_const(x, mu, sigma, N, Ny):
    return N*np.exp(-0.5*np.power((x - mu)/(sigma),2)) + Ny

def mypdf_double_gauss(x, mux, sigmax, Nx, sigmay, Ny):
    return Nx*np.exp(-0.5*np.power((x - mux)/(sigmax),2)) + Ny*np.exp(-0.5*np.power((x)/(sigmay),2))

def poisson_sigma(x, default=3):
    """
    Get the uncertainty of x (assuming it is poisson-distributed).
    Set *default* when x is 0 to avoid null uncertainties.
    """
    u = x**0.5
    u[x==0] = default
    return u


def labels(pl : PlotLabels):
    """
    Set x and y labels.
    """
    plt.xlabel(pl.x)
    plt.ylabel(pl.y)
    plt.title (pl.title)


def h1_alpha(x      : np.array,
       bins    : int,
       range   : Tuple[float],
       weights : Array = None,
       histtype: str   ='step',
       log     : bool  = False,
       normed  : bool  = False,
       color   : str   = 'black',
       alpha   : float = 1.,
       width   : float = 1.5,
       name    : str   = 'Raw energy',
       style   : str   ='solid',
       stats   : bool  = True,
       lbl     : Optional[str]  = None):
    """
    histogram 1d with continuous steps and display of statsself.
    number of bins (bins) and range are compulsory.
    """

    mu, std = mean_and_std(x, range)

    if stats:
        name     =  'Energy '+name
        entries  =  f'Entries = {len(x)}'
        mean     =  r'$\mu$ = {:7.2f}'.format(mu)
        sigma    =  r'$\sigma$ = {:7.2f}'.format(std)
        stat     =  f'{name}\n{entries}\n{mean}\n{sigma}'
    else:
        stat     = ''

    if lbl == None:
        lab = ' '
    else:
        lab = lbl

    lab = stat + lab

    if color == None:
        n, b, p = plt.hist(x,
                       bins      = bins,
                       range     = range,
                       weights   = weights,
                       log       = log,
                       density   = normed,
#                       histtype  = 'step',
                       histtype  = histtype,
                       linewidth = width,
                       linestyle = style,
                       label     = lab)

    else:

        n, b, p = plt.hist(x,
                       bins      = bins,
                       range     = range,
                       weights   = weights,
                       log       = log,
                       density   = normed,
                   #    histtype  = 'step',
                       histtype  = histtype,
                       #edgecolor = color,
                       color = color,
                       alpha = alpha,
                       linewidth = width,
                       linestyle = style,
                       label     = lab)

    return n, b, mu, std



    
def h2(x         : np.array,
       y         : np.array,
       nbins_x   : int,
       nbins_y   : int,
       range_x   : Tuple[float],
       range_y   : Tuple[float],
       profile   : bool   = True):

    xbins  = np.linspace(*range_x, nbins_x + 1)
    ybins  = np.linspace(*range_y, nbins_y + 1)

    nevt, *_  = plt.hist2d(x, y, (xbins, ybins))
    plt.colorbar().set_label("Entries")

    if profile:
        x, y, yu     = profile1d(x, y, nbins_x, range_x)
        plt.errorbar(x, y, yu, np.diff(x)[0]/2, fmt="kp", ms=7, lw=3)

    return nevt

def h2d(x         : np.array,
        y         : np.array,
        nbins_x   : int,
        nbins_y   : int,
        range_x   : Tuple[float],
        range_y   : Tuple[float],
        pltLabels : PlotLabels   = PlotLabels(x='x', y='y', title=None),
        profile  : bool          = False,
        figsize=(10,6)):

  #  fig = plt.figure(figsize=figsize)
  #  fig.add_subplot(1, 1, 1)

    nevt   = h2(x, y, nbins_x, nbins_y, range_x, range_y, profile)
    labels(pltLabels)
    return nevt



def plot_EQ(dst, Ebins, Eranges, Qbins, Qranges, loc_E='upper right', loc_Q='upper left', figsize=(14,10)):
    fig = plt.figure(figsize=figsize)

    ax      = fig.add_subplot(2, 2, 1)
    (_) = h1(dst.E, bins = Ebins, range = Eranges, histtype='stepfilled', color='crimson',
             lbl='')
    plot_histo(PlotLabels('E per sipm (pes)','Entries',''), ax)
    plt.legend(loc=loc_E)

    ax      = fig.add_subplot(2, 2, 2)
    (_) = h1(dst.Q, bins = Qbins, range = Qranges, histtype='stepfilled', color='crimson',
                 lbl='')
    plot_histo(PlotLabels('Q per sipm (pes)','Entries',''), ax)
    plt.legend(loc=loc_Q)

    
def plot_event_energy(dst, group, Ebins, Eranges, Qbins, Qranges, loc_E='upper right',
                      loc_Q='upper left', figsize=(14,10)):

    sns.set_style("white")
    sns.set_style("ticks")
    plot_EQ(group, 50, [3000, 7000], 50, [100, 300], loc_E= 'upper left',
            loc_Q= 'upper right', figsize=(15,11))

def control_plots_XY(dst, DX, DY):

    sns.set_style("white")
    sns.set_style("ticks")
    
    fig = plt.figure(figsize=(14,10))

    ax      = fig.add_subplot(2, 2, 1)
    (_)  = h2d ( dst.Xpeak, dst.Ypeak, 30, 30, [-75,75], [-75,75],
           pltLabels=PlotLabels(x='X (mm)', y='Y (mm)', title='X vs Y'),
           profile=False,
           figsize=(8,6))

    ax      = fig.add_subplot(2, 2, 2)

    (_)  = h2d(dst.X, dst.Y, 30, 30, [-75,75], [-75,75],
           pltLabels=PlotLabels(x='X (mm)', y='Y (mm)', title='X vs Y'),
           profile=False,
           figsize=(8,6))

    ax      = fig.add_subplot(2, 2, 3)
    (_)  = h2d(DX, DY, 50, 50, [-75,75], [-75,75],
           pltLabels=PlotLabels(x='X (mm)', y='Y (mm)', title='X vs Y'),
           profile=False,
           figsize=(8,6))

    ax      = fig.add_subplot(2, 2, 4)

    plt.hist(dst.Z, 50 , [0, 320] )
    plt.xlabel('Z')


def mypdf_double(x, mux, sigmax, Nx, sigmay, Ny):
    
    return Nx*np.exp(-0.5*np.power((x - mux),2)/(np.power(sigmax,2)+np.power(4.627,2))) + Ny*np.exp(-0.5*np.power((x)/(sigmay),2))
    

def plot_residuals(x, mux, mux_u , sigmax, sigmax_u, Nx, sigmay, sigmay_u, Ny, Qmean, Qmean_u, name_fig):

    import seaborn as sns
    sns.set()
    sns.set_style("white")
    sns.set_style("ticks")

    sigmay = np.abs(sigmay)
    global_linewidth    = 2
    global_linecolor    = "r" 
    
    # compute residuals
    y_from_fit = mypdf_double(x, mux, sigmax, Nx, sigmay, Ny )
    residuals     = (y_from_fit - Qmean)/ Qmean_u
    residuals_err = (y_from_fit - (Qmean+Qmean))/Qmean_u   # ---> no l'he usat
    
    # Plot
    frame_data = plt.gcf().add_axes((.1, .3,.8, .6))
    
    plt.errorbar    (x, Qmean, Qmean_u, 0, "p", c="k")
    plt.plot        (x, y_from_fit, lw=global_linewidth, color=global_linecolor   )

    # add the two gaussians
    #plt.fill_between(x, fitf.gauss(x, *f.values[ :3]),    0,     alpha=0.3, color=subfit_linecolor[0])
    #plt.fill_between(x, bkg       (x, *f.values[3: ]),    0,     alpha=0.3, color=subfit_linecolor[1])
    
    leg1 = plt.gca().legend(('fit', 'data'), loc='upper right')
    # to add two legends, we need to draw again the first one
    #plt.gca().legend(('fit', 'prova'), loc=0)
    #leg2 = plt.gca().legend(('fit', 'prova'), loc='upper left')
    #plt.gca().add_artist(leg1)   
    
    textstr = '\n'.join((
        r'$\mu=%.2f \pm %.2f$ ' % (mux,mux_u ),
        r'$\sigma 1=%.2f \pm %.2f$' % (sigmax, sigmax_u),
        r'$\sigma 2=%.2f$ ' % (sigmay, )))

    ## error on sigma 2
    #textstr = '\n'.join((
    #    r'$\mu=%.2f \pm %.2f$ ' % (mux,mux_u ),
    #    r'$\sigma 1=%.2f \pm %.2f$' % (sigmax, sigmax_u),
    #    r'$\sigma 2=%.2f \pm %.2f$' % (sigmay, sigmay_u)))
    
    
    props = dict(boxstyle='square', facecolor='white', alpha=0.5)
    plt.gca().text(0.05, 0.95, textstr, transform=plt.gca().transAxes, fontsize=14,
                   verticalalignment='top', bbox=props)
    
    frame_data.set_xticklabels([])
    plt.ylabel("Weigthed Charge")
    plt.ylim(0)
    lims = plt.xlim()
    print(lims)
    
    type(lims)
    frame_res = plt.gcf().add_axes((.1, .1, .8, .2))
    plt.plot    (lims, [0,0], "-g", lw=0.7)  # linia en 00 verde
    plt.errorbar(x, residuals, 1, 0, linestyle='None', fmt='|', c="k")
    plt.ylim(-3.9,3.9)
    plt.xlabel("x (mm)")

    plt.savefig('/Users/neus/current-work/diffusion/plots_residuals/'+name_fig+'.png')


def mypdf_EL_double(x, mux, sigmax, Nx, sigmay, Ny):
    return Nx*np.exp(-0.5*np.power((x - mux)/(sigmax),2)) + Ny*np.exp(-0.5*np.power((x)/(sigmay),2))

def plot_residuals_EL(x, mux, mux_u , sigmax, sigmax_u, Nx, sigmay, sigmay_u, Ny, Qmean, Qmean_u, name_fig):
    
    import seaborn as sns
    sns.set()
    sns.set_style("white")
    sns.set_style("ticks")
    
    global_linewidth    = 2
    global_linecolor    = "r" 
    
    # get absolute value sigma y   -> check
    sigmay = np.abs(sigmay)

    # compute residuals
    y_from_fit = mypdf_EL_double(x, mux, sigmax, Nx, sigmay, Ny )
    residuals     = (y_from_fit - Qmean)/ Qmean_u
    residuals_err = (y_from_fit - (Qmean+Qmean))/Qmean_u   # ---> no l'he usat
    
    # Plot
    frame_data = plt.gcf().add_axes((.1, .3,.8, .6))
    
    plt.errorbar    (x, Qmean, Qmean_u, 0, "p", c="k")
    plt.plot        (x, y_from_fit, lw=global_linewidth, color=global_linecolor   )
    
    # add the two gaussians
    #plt.fill_between(x, fitf.gauss(x, *f.values[ :3]),    0,     alpha=0.3, color=subfit_linecolor[0])
    #plt.fill_between(x, bkg       (x, *f.values[3: ]),    0,     alpha=0.3, color=subfit_linecolor[1])
    
    leg1 = plt.gca().legend(('fit', 'data'), loc='upper right')
    # to add two legends, we need to draw again the first one
    #plt.gca().legend(('fit', 'prova'), loc=0)
    #leg2 = plt.gca().legend(('fit', 'prova'), loc='upper left')
    #plt.gca().add_artist(leg1)   
    
    textstr = '\n'.join((
        r'$\mu=%.2f \pm %.2f$ ' % (mux,mux_u ),
        r'$\sigma 1=%.2f \pm %.2f$' % (sigmax, sigmax_u),
        r'$\sigma 2=%.2f$ ' % (sigmay, )))

    ## error on sigma 2
    #textstr = '\n'.join((
    #    r'$\mu=%.2f \pm %.2f$ ' % (mux,mux_u ),
    #    r'$\sigma 1=%.2f \pm %.2f$' % (sigmax, sigmax_u),
    #    r'$\sigma 2=%.2f \pm %.2f$' % (sigmay, sigmay_u)))
    
    
    props = dict(boxstyle='square', facecolor='white', alpha=0.5)
    plt.gca().text(0.05, 0.95, textstr, transform=plt.gca().transAxes, fontsize=14,
                   verticalalignment='top', bbox=props)
    
    frame_data.set_xticklabels([])
    plt.ylabel("Weigthed Charge")
    plt.ylim(0)
    lims = plt.xlim()
    print(lims)
    
    type(lims)
    frame_res = plt.gcf().add_axes((.1, .1, .8, .2))
    plt.plot    (lims, [0,0], "-g", lw=0.7)  # linia en 00 verde
    plt.errorbar(x, residuals, 1, 0, linestyle='None', fmt='|', c="k")
    plt.ylim(-3.9,3.9)
    plt.xlabel("x (mm)")

    plt.savefig('/Users/neus/current-work/diffusion/plots_residuals/'+name_fig+'.png')



def plot_residuals_E_reso_gaussian_const(energy, e_nbins, e_range, mu, mu_u , sigma, sigma_u, N, N_u, N2,N2_u, name_fig):

    import seaborn as sns
    sns.set()
    sns.set_style("white")
    sns.set_style("ticks")

    fig = plt.figure(figsize=(9,7))
    
    global_linewidth    = 2
    global_linecolor    = "r" 
    
    # compute y values from histogram
    e_bins       =  np.linspace(*  e_range   ,   e_nbins + 1)
    entries, e   =  np.histogram(energy, e_bins)
    e            =  shift_to_bin_centers(e)
    #e_u          =  np.diff(e)[0] * 0.5
    entries_u    =  poisson_sigma(entries)
    #entries_u    =  entries**0.5
    
    # compute bin width
    w= (e_range[1]- e_range[0])/e_nbins
    
    # compute residuals
    y_from_fit    = mypdf_gauss_const(e, mu, sigma, N*w, N2*w )
    residuals     = (y_from_fit - entries)/ entries_u
    y_from_fit_1  = mypdf_gauss(e, mu, sigma,  N*w)
    y_from_fit_2  = N2*w
    
    # Plot
    frame_data = plt.gcf().add_axes((.1, .3,.8, .6))
    
    plt.errorbar    (e, entries, entries_u, 0, "p", c="k")
    plt.plot        (e, y_from_fit, lw=global_linewidth, color=global_linecolor   )
    plt.fill_between(e, y_from_fit_1,    0,     alpha=0.3, color='')
    plt.fill_between(e, y_from_fit_2,    0,     alpha=0.5, color='pink')
    
    
    
    leg1 = plt.gca().legend(('fit', 'data'), loc='upper right')

    # compute resolution
    resolution = 235*sigma/mu
    
    textstr = '\n'.join((
        '$\mu={:.2f}      \pm {:.2f} $'     .format(mu,mu_u),
        '$\sigma 1={:.2f} \pm {:.2f}$'      .format(sigma, sigma_u),
        '$N 1={:.2f}      \pm {:.2f}$'      .format(N, N_u),
        '$N 2={:.2f}      \pm {:.2f}$'      .format(N2, N2_u),
        '$\sigma_E/E =        {:.2f} \%  $' .format(resolution,)))
       

    props = dict(boxstyle='square', facecolor='white', alpha=0.5)
    plt.gca().text(0.05, 0.95, textstr, transform=plt.gca().transAxes, fontsize=14,
                   verticalalignment='top', bbox=props)
    
    frame_data.set_xticklabels([])
    plt.ylabel("Entries")
    plt.ylim(-500)
    
    # set my own xlimits
    #lims = plt.xlim()
    lims = plt.xlim(e_range[0], e_range[1])
    frame_res = plt.gcf().add_axes((.1, .1, .8, .2))
    plt.plot    (lims, [0,0], "-g", lw=0.7)  # linia en 00 verde
    plt.errorbar(e, residuals, 1, 0, linestyle='None', fmt='|', c="k")
    plt.ylim(-3.9,3.9)
    plt.xlim(e_range[0], e_range[1])
    plt.xlabel("E (pes)")
    
    plt.savefig('/Users/neus/current-work/diffusion/energy_resolution/'+name_fig+'.png')
    

def plot_residuals_E_reso_double_gaussian(energy, e_nbins, e_range, mu, mu_u , sigma, sigma_u, N, N_u, sigma2,sigma2_u, N2,N2_u, name_fig):

    import seaborn as sns
    sns.set()
    sns.set_style("white")
    sns.set_style("ticks")

    fig = plt.figure(figsize=(9,7))
    
    global_linewidth    = 2
    global_linecolor    = "r" 
    
    # compute y values from histogram
    e_bins       =  np.linspace(*  e_range   ,   e_nbins + 1)
    entries, e   =  np.histogram(energy, e_bins)
    e            =  shift_to_bin_centers(e)
    #e_u          =  np.diff(e)[0] * 0.5
    entries_u    =  poisson_sigma(entries)
    #entries_u    =  entries**0.5
    
    # compute bin width
    w= (e_range[1]- e_range[0])/e_nbins
    
    # compute residuals
    y_from_fit    = mypdf_double_gauss(e, mu, sigma, N*w, sigma2, N2*w )
    residuals     = (y_from_fit - entries)/ entries_u
    y_from_fit_1  = mypdf_gauss(e, mu, sigma,  N*w)
    y_from_fit_2  = mypdf_gauss(e, mu, sigma2, N2*w)
    
    # Plot
    frame_data = plt.gcf().add_axes((.1, .3,.8, .6))
    
    plt.errorbar    (e, entries, entries_u, 0, "p", c="k")
    plt.plot        (e, y_from_fit, lw=global_linewidth, color=global_linecolor   )
    #plt.fill_between(e, y_from_fit_1,    0,     alpha=0.3, color='')
    #plt.fill_between(e, y_from_fit_2,    0,     alpha=0.3, color='green')
    
    
    
    leg1 = plt.gca().legend(('fit', 'data'), loc='upper right')

    # compute resolution
    resolution = 235*sigma/mu

    textstr = '\n'.join((
        '$\mu={:.2f}      \pm {:.2f} $'    .format(mu,mu_u),
        '$\sigma 1={:.2f} \pm {:.2f}$'    .format(sigma, sigma_u),
        '$N 1={:.0f}      \pm {:.0f}$'     .format(N, N_u),
        '$\sigma 2={:.2f} \pm {:.2f}$'    .format(sigma2, sigma2_u ),
        '$N 2={:.0f}      \pm {:.0f}$'     .format(N2, N2_u),
        '$\sigma_E/E =         {:.2f} \%  $'  .format(resolution,)))
       
    
    props = dict(boxstyle='square', facecolor='white', alpha=0.5)
    plt.gca().text(0.05, 0.95, textstr, transform=plt.gca().transAxes, fontsize=14,
                   verticalalignment='top', bbox=props)
    
    frame_data.set_xticklabels([])
    plt.ylabel("Entries")
    plt.ylim(0)
    
    # set my own xlimits
    #lims = plt.xlim()
    lims = plt.xlim(e_range[0], e_range[1])
    frame_res = plt.gcf().add_axes((.1, .1, .8, .2))
    plt.plot    (lims, [0,0], "-g", lw=0.7)  # linia en 00 verde
    plt.errorbar(e, residuals, 1, 0, linestyle='None', fmt='|', c="k")
    plt.ylim(-3.9,3.9)
    plt.xlim(e_range[0], e_range[1])
    plt.xlabel("E (pes)")

    plt.savefig('/Users/neus/current-work/diffusion/energy_resolution/'+name_fig+'.png')
    
    
def plot_residuals_E_reso_gauss(energy, e_nbins, e_range, mu, mu_u , sigma, sigma_u, N,N_u, name_fig):
    
    import seaborn as sns
    sns.set()
    sns.set_style("white")
    sns.set_style("ticks")

    fig = plt.figure(figsize=(9,7))
    global_linewidth    = 2
    global_linecolor    = "r" 
    
    # compute y values from histogram
    e_bins       =  np.linspace(*  e_range   ,   e_nbins + 1)
    entries, e   =  np.histogram(energy, e_bins)
    e            =  shift_to_bin_centers(e)
    e_u          =  np.diff(e)[0] * 0.5
    #entries_u    =  poisson_sigma(entries)
    entries_u    =  entries**0.5
    
    # compute residuals
    y_from_fit    = mypdf_gauss(e, mu, sigma, N*13 )
    residuals     = (y_from_fit - entries)/ entries_u
    
    # Plot
    frame_data = plt.gcf().add_axes((.1, .3,.8, .6))
    
    plt.errorbar    (e, entries, entries_u, 0, "p", c="k")
    plt.plot        (e, y_from_fit, lw=global_linewidth, color=global_linecolor   )
    
    leg1 = plt.gca().legend(('fit', 'data'), loc='upper right')


    # compute resolution
    resolution = 235*sigma/mu
    
    textstr = '\n'.join((
        '$\mu={:.2f}      \pm {:.2f} $'     .format(mu,mu_u),
        '$\sigma 1={:.2f} \pm {:.2f}$'      .format(sigma, sigma_u),
        '$N ={:.2f}      \pm {:.2f}$'       .format(N, N_u),
        '$\sigma_E/E =        {:.2f} \%  $' .format(resolution,)))
    
    
    props = dict(boxstyle='square', facecolor='white', alpha=0.5)
    plt.gca().text(0.05, 0.95, textstr, transform=plt.gca().transAxes, fontsize=14,
                   verticalalignment='top', bbox=props)
    
    frame_data.set_xticklabels([])
    plt.ylabel("Entries")
    plt.ylim(0)
    lims = plt.xlim()
    print(lims)
    
    type(lims)
    frame_res = plt.gcf().add_axes((.1, .1, .8, .2))
    plt.plot    (lims, [0,0], "-g", lw=0.7)  # linia en 00 verde
    plt.errorbar(e, residuals, 1, 0, linestyle='None', fmt='|', c="k")
    plt.ylim(-3.9,3.9)
    plt.xlabel("E (pes)")

    plt.savefig('/Users/neus/current-work/diffusion/plots_residuals/'+name_fig+'.png')

 

