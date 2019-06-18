import numpy as np
import math as math
import matplotlib.pyplot as plt
import scipy             as sci
import seaborn as sns

from typing import Tuple, Optional

from   invisible_cities.icaro.hst_functions       import hist2d_profile
#from   krcal.core.histo_functions                 import h1, h1d, h2, h2d, plot_histo
from   krdiff.core.histo_functions                import h2, h2d, plot_EQ, plot_event_energy, control_plots_XY

from   invisible_cities.core.core_functions       import in_range
from   krcal.core.kr_types                        import PlotLabels, FitType
#from   krdiff.core.plt_functions                 import plot_EQ


sns.set()


def select_slice_in_Y(DX, DY, dst, cut):

    cut_value = 2
    cut = np.abs(DY)<cut_value

    DX_cut  = DX_vlow [cut]
    DY_cut  = DY_vlow [cut]
    dst_cut = dst_vlow[cut]

    return DX_cut, DY_cut, dst_cut




def prepare_tofit(DXA, DYA, xbins, ybins, xrange, yrange,  dst_weights):
    """ Get the 3 histograms from which to extract the variables and return variables and error to fit
    """
    
    h2w ,_,_,_ = plt.hist2d(DXA, DYA, [xbins, ybins], [xrange, yrange], weights = dst_weights )
    plt.colorbar()
    plt.show()

    h2n ,_,_,_ = plt.hist2d(DXA,DYA, [xbins,ybins], [xrange, yrange] )
    plt.colorbar()
    plt.show()

    (xa,ya,za,zua), _, cb = hist2d_profile(DXA, DYA, dst_weights, xbins, ybins,
                                           xrange, yrange, new_figure = True)

    # get matrix of profile in Z (weights_mean/num_entries), num_entries= events x sipms fired.
    # This matrix is the same as variable 'za' from hist2d_profile

    #print(xa.shape)
    #print(ya.shape)
    #print(zua.shape)
    
    #print(h2w.shape)
    #print(h2n.shape)
    wmean_xybin = h2w/h2n
    
    #print(wmean_xybin.shape)
    #print(h2n)
    #print(wmean_xybin[0][20])
    #print(wmean_xybin[20][20])
    h2w_sq = h2w*h2w
    zua_sq = zua*zua
    
    # Get fractions to weight the sum of w mean values
    #h2w_xbin = h2w.sum(axis=0)
    #h2n_xbin = h2n.sum(axis=0)
    #wfinal_xbin = h2w_xbin/h2n_xbin
    #w_xbin      = wmean_xybin.sum(axis=1)
    h2w_xbin      = h2w.sum(axis=1)
    h2n_xbin      = h2n.sum(axis=1)
    h2w_sq_xbin   = h2w_sq.sum(axis=1)
    zua_sq_xbin   = zua_sq.sum(axis=1)
    
    #print(w_xbin.shape)
    #fw_xybin    = wmean_xybin / w_xbin
    
    #print(w_xbin.shape)
    #print(h2n[0])
    #print(h2n[20])
    
    #print(w_xbin[0])
    #print(w_xbin[20])
    #print(w_xbin)
    
    # Get w weighted by fractions
    #wfinal_xybin = fw_xybin*wmean_xybin

    # Sum values per columns (x axis)
    #wfinal_xbin = wfinal_xybin.sum(axis=0)
    #wmean2_xybin = wmean_xybin*wmean_xybin
    #w2_xbin = wmean2_xybin.sum(axis=1)
    
    #wmean_xbin = w_xbin/n_xbin
    
    ###metodo viejo no tan bueno
    #wmean_xbin = h2w_xbin/h2n_xbin
    #wmean_sq_xbin = h2w_sq_xbin/h2n_xbin   
    #wmean_err_xbin = np.sqrt(wmean_sq_xbin - np.power(wmean_xbin,2))/np.sqrt(h2n_xbin)
    
    ###metodo bueno
    wmean_xbin     = wmean_xybin.sum(axis=1)/ybins
    wmean_err_xbin = np.sqrt(zua_sq_xbin)/np.sqrt(ybins)
    
    #wfinal_xbin = wfinal_xybin.sum(axis=0)
    #wmean2_xbin = w2_xbin/h2n_xbin
    #wmean_err_xbin = np.sqrt(wmean2_xbin - np.power(wmean_xbin,2))/np.sqrt(h2n_xbin)
    
    # Compute uncertainties
    #f2w_xybin       = fw_xybin * fw_xybin
    #err_xybin       = f2w_xybin * np.power(zua, 2)
    #wfinal_err_xbin = np.sqrt(err_xybin.sum(axis=0))
    
    #print(len(h2w), len(h2n), len(w_xbin), len(fw_xybin))
    #print(wmean_xbin[20])
    #print(wmean_err_xbin[20])
    
    plt.errorbar(xa , wmean_xbin, wmean_err_xbin, fmt="kp")
    
    #print(wfinal_xbin)

    return xa, wmean_xbin, wmean_err_xbin



def prepare_tofit_first_method(DXA, DYA, xybins, xyrange, dst_weights):
    """ Get the 3 histograms from which to extract the variables and return variables and error to fit
    """
    
    h2w ,_,_,_ = plt.hist2d(DXA, DYA, xybins, [xyrange, xyrange], weights = dst_weights )
    plt.colorbar()
    plt.show()

    h2n ,_,_,_ = plt.hist2d(DXA,DYA, xybins, [xyrange, xyrange] )
    plt.colorbar()
    plt.show()

    (xa,ya,za,zua), _, cb = hist2d_profile(DXA, DYA, dst_weights, xybins, xybins,
                                           xyrange, xyrange, new_figure = True)

    # get matrix of profile in Z (weights_mean/num_entries), num_entries= events x sipms fired.
    # This matrix is the same as variable 'za' from hist2d_profile

    wmean_xybin = h2w/h2n

    # Get fractions to weight the sum of w mean values
    w_xbin      = wmean_xybin.sum(axis=0)
    fw_xybin    = wmean_xybin / w_xbin

    # Get w weighted by fractions
    wfinal_xybin = fw_xybin*wmean_xybin

    # Sum values per columns (x axis)
    wfinal_xbin = wfinal_xybin.sum(axis=0)

    # Compute uncertainties
    f2w_xybin       = fw_xybin * fw_xybin
    err_xybin       = f2w_xybin * np.power(zua, 2)
    wfinal_err_xbin = np.sqrt(err_xybin.sum(axis=0))
    

    plt.errorbar(xa , wfinal_xbin, wfinal_err_xbin, fmt="kp")

    return xa, wfinal_xbin, wfinal_err_xbin



def get_vars(dst):

    group = dst.groupby(['event'])
    Qtot  = group.Q.transform('sum')

    # relative positions of sipms with respect to baricenter
    DX = dst.X.values - dst.Xpeak.values
    DY = dst.Y.values - dst.Ypeak.values

    # weight positions by sensor charge and normalized by event total charge
    w = dst.Q.values/Qtot

    # add a new column to dataframe, for convenience (revisar si es necessari)
    dst = dst.assign(w=w)
    return dst, DX, DY

def plot_sipms_1ev(dst, ev):
    # plot sipms charge for one event

    dst_one = dst[dst.event==ev]
    plt.hist2d(dst_one.X,dst_one.Y, 16,[[-75,75],[-75,75]], weights=dst_one.Q)
    plt.show()


def event_energy_cut(dst, emin, emax):
    """ input dst in which each entry correspond to a given sipm"""
    
    group      = dst.groupby(['event']).sum()
    group_Ecut = group[in_range(group.E, emin, emax)]
    dstCoreE   = dst[dst.event.isin(group_Ecut.index)]
    group_CoreE = dstCoreE.groupby(['event']).sum()

    # plot selected energy
    sns.set_style("white")
    sns.set_style("ticks")
    plot_event_energy(dstCoreE, group_CoreE, 50, [3000, 7000], 50, [100, 300], loc_E= 'upper left',
            loc_Q= 'upper right', figsize=(15,11))

    # plot e/q for sipm
    sns.set_style("white")
    sns.set_style("ticks")
    plot_EQ(dst, 50, [0, 1000], 50, [0, 100], figsize=(15,11))

    return dstCoreE

def sipm_q_cut(dst, minq):
    """ input dst in which each entry correspond to a given sipm"""
    
    group = dst.groupby(['event'])
    Qtot  = group.Q.transform('sum')

    # weight positions by sensor charge and normalized by event total charge
    w   = dst.Q.values/Qtot
    #dst = dst.assign(w=w)

    dst = dst[w > minq]
    
    return dst



def T_coeff (T, P, m, v):

    """ T: temperature of sensors
        P: pressure of demo
        m: slope of the linear fit to the sigmas obtained at different
        drift regions.
        DT: transverse coefficient obtained from the fit.
        slope = 2 * DL / vd**2"""

    T  = T + 273.15
    T0 = 20 + 273.15

    DT = m/2*1000000/100

    DTcoeff = np.sqrt(1000*(T0/T)*2*P*DT/v)

    return DTcoeff
