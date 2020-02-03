import numpy as np
import math as math
import matplotlib.pyplot as plt
import scipy             as sci
import seaborn as sns

from typing import Tuple, Optional

import pandas as pd

from   invisible_cities.icaro.hst_functions       import hist2d_profile
#from   krcal.core.histo_functions                 import h1, h1d, h2, h2d, plot_histo
from   krdiff.core.histo_functions                import plot_histo, h1, h2, h2d, plot_EQ, plot_event_energy, control_plots_XY, plot_wcharged

from   invisible_cities.core.core_functions       import in_range
from   krcal.core.kr_types                        import PlotLabels, FitType
#from   krdiff.core.plt_functions                 import plot_EQ


from numpy import pi, r_

from scipy import optimize


sns.set()



def T_coeff (T, P, m, v):

    """ T: temperature of sensors
        P: pressure of demo
        m: slope of the linear fit to the sigmas obtained at different
        drift regions.
        v: velocity
        DT: transverse coefficient obtained from the fit.
        slope = 2 * DT / vd**2"""

    T  = T + 273.15
    T0 = 20 + 273.15

    DT = m/2*1000000/100

    DTcoeff = np.sqrt(1000*(T0/T)*2*P*DT/v)

    return DTcoeff


def longitudinal_coeff (m, vd):

    """ m: [mus] slope of the linear fit to the sigmas2 (time units) vs drift time.
        vd: [mm/mus] velocity
        slope = 2 * DT / vd**2 
        factor_change_units: from [mm2 / mus]  to [cm2/s], 10^6 s-mus, 100 mm2-cm2
    """

    factor_change_units =  1000000 / 100
    DL = m/2 * vd**2 * factor_change_units

    return DL

def longitudinal_coeff_usingZ (m, vd):

    """ m: [mus] slope of the linear fit to the sigmas2 (time units) vs drift time.
        vd: [mm/mus] velocity
        slope = 2 * DT / vd**2 
        factor_change_units: from [mm2 / mus]  to [cm2/s], 10^6 s-mus, 100 mm2-cm2
    """

    factor_change_units =  1000000 / 100
    DL = m/2 * vd * factor_change_units

    return DL


def longitudinal_star_coeff (DL, T, P, vd):

    """ DL: longitudinal coefficient obtained from the fit
        T: temperature of sensors
        P: pressure of demo
        vd: [mm/mus] velocity  
        change_units: from sqrt([cm2/2 * mus / mm]) to sqrt([mum2/cm]) = mum/sqrt(cm),
        cm2->mum2: 10**8; mm->cm: 1/10**-1;  mus->s 1/10**6
        nota: estem assummint compressibility factors Z0/Z0p =1
        https://encyclopedia.airliquide.com/xenon
   """

    change_units = 10**8 / (10**6 * 10**-1)

    T  = T  + 273.15
    T0 = 20 + 273.15
    
    DLstar = np.sqrt(change_units * (T0/T) * 2 * P * DL / vd)

    return DLstar


def trans_coeff_using_Z (m, vd):

    """ m: [mus] slope of the linear fit to the sigmas2  (space units) vs drift time
        slope = 2 * DT / vd**2 
        factor_change_units: from [mm2 / mus]  to [cm2/s], 10^6 s-mus, 100 mm2-cm2
    This functions takes into account that we have using the Z coordinate instead of 
drift time, for that we multiply up and down in the equation by velocity:
DT = sigma2 * vd /(2t*vd)
    """

    factor_change_units =  1000000 / 100
    DT = vd * m/2 * factor_change_units

    return DT



def trans_coeff (m):

    """ m: [mus] slope of the linear fit to the sigmas2  (space units) vs drift time
        slope = 2 * DT / vd**2 
        factor_change_units: from [mm2 / mus]  to [cm2/s], 10^6 s-mus, 100 mm2-cm2
    """

    factor_change_units =  1000000 / 100
    DT = m/2 * factor_change_units

    return DT


def trans_star_coeff (DT, T, P, vd):

    """ DL: longitudinal coefficient obtained from the fit
        T: temperature of sensors
        P: pressure of demo
        vd: [mm/mus] velocity  
        change_units: from sqrt([cm2/2 * mus / mm]) to sqrt([mum2/cm]) = mum/sqrt(cm),
        cm2->mum2: 10**8; mm->cm: 1/10**-1;  mus->s 1/10**6
        nota: estem assummint compressibility factors Z0/Z0p =1
        https://encyclopedia.airliquide.com/xenon
   """

    change_units = 10**8 / (10**6 * 10**-1)

    T  = T  + 273.15
    T0 = 20 + 273.15
    
    DTstar = np.sqrt(change_units * (T0/T) * 2 * P * DT / vd)

    return DTstar


def select_slice_in_Y(DX, DY, dst, cut):

    cut_value = 2
    cut = np.abs(DY)<cut_value

    DX_cut  = DX_vlow [cut]
    DY_cut  = DY_vlow [cut]
    dst_cut = dst_vlow[cut]

    return DX_cut, DY_cut, dst_cut

#https://www.sharpsightlabs.com/blog/numpy-sum/


def obtain_mean_and_error_projx(DXA, DYA, xbins, ybins, x_range, y_range,  dst_weights, proj):
    """proj = 0   --> proyect X axis, proj = 1   --> proyect Y axis"""

    h2w ,_,_,_  = plt.hist2d(DXA, DYA, [xbins, ybins], [x_range, y_range] , weights = dst_weights,  cmap='coolwarm')
    h2n ,_,_,_  = plt.hist2d(DXA, DYA, [xbins, ybins], [x_range, y_range] , cmap='coolwarm')
    h2w2 ,_,_,_ = plt.hist2d(DXA, DYA, [xbins, ybins], [x_range, y_range] , weights = np.square(dst_weights),  cmap='coolwarm')

    ## check, perque es nomes per a tenir xa
    (xa,ya,za,zua), _, cb = hist2d_profile(DXA, DYA, dst_weights, xbins, ybins,
                                           x_range, y_range, new_figure = False,  cmap='coolwarm')

    ## 1. Sum up al columns to project in axis
    
    array_1d_nsum_projx   = np.sum(h2n,  axis=proj)
    array_1d_wsum_projx   = np.sum(h2w,  axis=proj)
    array_1d_w2sum_projx  = np.sum(h2w2, axis=proj)

    #### 2. Divide by number of entries (= silicons fired) both average and average squared

    array_1d_waverage_projx  = array_1d_wsum_projx /array_1d_nsum_projx
    array_1d_w2average_projx = array_1d_w2sum_projx/array_1d_nsum_projx

    #### 3. Squared average

    array_1d_waverage2_projx = np.square(array_1d_waverage_projx)

    #### 4. average of the squares minus average squared

    sigma2 = array_1d_w2average_projx - array_1d_waverage2_projx
    sigma2n = sigma2/array_1d_nsum_projx

    sigma = np.sqrt(sigma2n)

    return xa, array_1d_waverage_projx, sigma



    




def prepare_tofit(DXA, DYA, xbins, ybins, x_range, y_range,  dst_weights):
    """ Get the 3 histograms from which to extract the variables and return variables and error to fit
    """
    
    h2w ,_,_,_ = plt.hist2d(DXA, DYA, [xbins, ybins], [x_range, y_range], weights = dst_weights )
    plt.colorbar()
    plt.show()

    h2n ,_,_,_ = plt.hist2d(DXA,DYA, [xbins,ybins], [x_range, y_range] )
    plt.colorbar()
    plt.show()

    (xa,ya,za,zua), _, cb = hist2d_profile(DXA, DYA, dst_weights, xbins, ybins,
                                           x_range, y_range, new_figure = True)

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
    h2w_sq = h2w*h2w    #   ----> creo que aqui esta mal, porque estamos elevando al cuadrado sumas de w
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

    group      = dst.groupby(['event'])
    Qtot       = group.Q.transform('sum')
    nsipm_tot  = group.nsipm.transform('sum')

    # relative positions of sipms with respect to baricenter
    DX = dst.X.values - dst.Xpeak.values
    DY = dst.Y.values - dst.Ypeak.values

    # weight positions by sensor charge and normalized by event total charge
    w = dst.Q.values/Qtot

    # add a new column to dataframe, for convenience (revisar si es necessari)
    dst = dst.assign(w=w)
    dst = dst.assign(Qtot=Qtot)
    dst = dst.assign(nsipm_tot=nsipm_tot)

    return dst, DX, DY

def plot_sipms_1ev(dst, ev):
    # plot sipms charge for one event

    dst_one = dst[dst.event==ev]
    plt.hist2d(dst_one.X,dst_one.Y, 16,[[-75,75],[-75,75]], weights=dst_one.Q)
    plt.show()


def event_energy_cut(dst, emin, emax, qmin =100, qmax=300, e_sipm_max = 1000, q_sipm_max = 100):
    """ input dst in which each entry correspond to a given sipm"""
    
    group      = dst.groupby(['event']).sum()
    group_Ecut = group[in_range(group.E, emin, emax)]
    dstCoreE   = dst[dst.event.isin(group_Ecut.index)]
    group_CoreE = dstCoreE.groupby(['event']).sum()

    # plot selected energy
    sns.set_style("white")
    sns.set_style("ticks")
    plot_event_energy(dstCoreE, group_CoreE, 50, [emin, emax],
                      50, [qmin, qmax], loc_E= 'upper left',
            loc_Q= 'upper right', figsize=(15,11))

    # plot e/q for sipm
    sns.set_style("white")
    sns.set_style("ticks")
    # 24 juliol he canviat dst --> dstcore, no se si hi havia rao.
    
    #plot_EQ(dst, 50, [0, 1000], 50, [0, 100], figsize=(15,11))
    plot_EQ(dstCoreE, 50, [0, e_sipm_max], 50, [0, q_sipm_max], figsize=(15,11))

    return dstCoreE




def sipm_q_cut(dst, minq):
    """ input dst in which each entry correspond to a given sipm"""
    
    group = dst.groupby(['event'])
    Qtot  = group.Q.transform('sum')

    # weight positions by sensor charge and normalized by event total charge
    w   = dst.Q.values/Qtot
    #dst = dst.assign(w=w)

    # plot selected w values
    sns.set_style("white")
    sns.set_style("ticks")
    
    dst_simp_q_cut = dst[w > minq]

    # to check cut on w, repeat data managing above
    group_cut = dst_simp_q_cut.groupby(['event'])
    Qtot_cut  = group_cut.Q.transform('sum')

    # weight positions by sensor charge and normalized by event total charge
    w_cut  = dst_simp_q_cut.Q.values/Qtot_cut

    plot_wcharged(w,w_cut)
    
    return dst_simp_q_cut






def numpystd(df):
  
    df['numpy_std']   = np.std (df.Z.values)
    df['numpy_meanZ'] = np.mean(df.Z.values)
    
    return(df)


## te un bug perque estic agafant arbitrariament el primer element
def compute_tau_np (dst_in):

    dst_in_index = dst_in.reset_index()
    dst_in_std   = dst_in_index.groupby('event').apply(numpystd)

    dst = dst_in_std.drop_duplicates('numpy_std', keep='first')

    return dst

##### LONGITUDINAL


def select_event_in_between_lines (d):

    # obtained the two lines to select events
    var_line_low  = d.wVar[(d.wT>10)  &(d.wT<15)]  .std()
    var_line_high = d.wVar[(d.wT>290) &(d.wT<300)] .std()

    y1 = d.wVar[(d.wT>10) &(d.wT<15)].mean() + 1.5*var_line_low
    y2 = d.wVar[(d.wT>290)&(d.wT<300)].mean()+ 2.5*var_line_low 

    x1 = 10
    x2 = 300

    x = d.wT.values
    y = d.wVar.values
    array_xy = np.vstack((x,y)).T

    isbelow = lambda p, a,b: np.cross(p-a, b-a) > 0

    a = np.array([x1, y1])
    b = np.array([x2, y2])
    
    is_below = isbelow(array_xy,a,b)
    x_below = array_xy[:,0][is_below]
    y_below = array_xy[:,1][is_below]

    return x_below, y_below, array_xy, y1, y2


def select_event_given_a_line (d):

    # obtained the two lines to select events
    var_line_low  = d.wVar[(d.wT>10)  &(d.wT<15)]  .std()
    var_line_high = d.wVar[(d.wT>290) &(d.wT<300)] .std()

    y1 = d.wVar[(d.wT>10) &(d.wT<15)].mean() + 1.5*var_line_low
    y2 = d.wVar[(d.wT>290)&(d.wT<300)].mean()+ 2.5*var_line_low 

    x1 = 10
    x2 = 300

    x = d.wT.values
    y = d.wVar.values
    array_xy = np.vstack((x,y)).T

    isbelow = lambda p, a,b: np.cross(p-a, b-a) > 0

    a = np.array([x1, y1])
    b = np.array([x2, y2])
    
    is_below = isbelow(array_xy,a,b)
    x_below = array_xy[:,0][is_below]
    y_below = array_xy[:,1][is_below]

    return x_below, y_below, array_xy, y1, y2



def compute_weighted_variance (dst_in):
    
    d = dst_in.copy()

    group = d.groupby(['event'])
    Etot  = group.E.transform(np.sum)
    Efrac = d.E.values/Etot
    num_ev  = 1

    Zmean  = group.Z.transform(np.mean)
    Zvar   = group.Z.transform(np.var)
    Zstd   = group.Z.transform(np.std)

    d['num_ev'] = num_ev
    d['Etot']   = Etot
    d['Efrac']  = Efrac
    d['num_ev'] = num_ev
    d['Zmean']  = Zmean
    d['Zvar']   = Zvar
    d['Zstd']   = Zstd
    d['wTi']    = d['Efrac'] * d['Z']

    group2      = d.groupby(['event'])
    wT          = group2.wTi.transform(np.sum)
    ntot        = group2.num_ev.transform(np.sum)
    Efrac_check = group2.Efrac.transform(np.sum)

    d['ntot']   = ntot
    d['wT']     = wT
    d['sig_i']  = (d['Z'] - d['wT'])
    d['sig2_i'] = d['Efrac']*(d['Z'] - d['wT'])**2

    var      = group2.sig2_i.transform(np.sum)
    d['wVar'] = var 
    d['wStd'] = np.sqrt(var)

    d = d.drop_duplicates('wVar', keep='first')

    return d

def compute_weighted_variance_check_event (dst_in, ev_range_id_low, ev_range_id_high):
    
    dstE_cut = dst_in.iloc[ev_range_id_low:ev_range_id_high]
    d = dstE_cut.copy()

    group = d.groupby(['event'])
    Etot  = group.E.transform(np.sum)
    Efrac = d.E.values/Etot
    num_ev  = 1

    Zmean  = group.Z.transform(np.mean)
    Zvar   = group.Z.transform(np.var)
    Zstd   = group.Z.transform(np.std)

    d['num_ev'] = num_ev
    d['Etot']   = Etot
    d['Efrac']  = Efrac
    d['num_ev'] = num_ev
    d['Zmean']  = Zmean
    d['Zvar']   = Zvar
    d['Zstd']   = Zstd
    d['wTi']    = d['Efrac'] * d['Z']

    group2      = d.groupby(['event'])
    wT          = group2.wTi.transform(np.sum)
    ntot        = group2.num_ev.transform(np.sum)
    Efrac_check = group2.Efrac.transform(np.sum)

    d['ntot']   = ntot
    d['wT']     = wT
    d['sig_i']  = (d['Z'] - d['wT'])
    d['sig2_i'] = d['Efrac']*(d['Z'] - d['wT'])**2

    var      = group2.sig2_i.transform(np.sum)
    d['wVar'] = var 
    d['wStd'] = np.sqrt(var)

    # comment to check all steps
    #d = d.drop_duplicates('wVar', keep='first')

    return d




def compute_weighted_variance_check (dst_in, ev_range):
    
    dstE_cut = dst_in.iloc[0:ev_range]
    d = dstE_cut.copy()

    group = d.groupby(['event'])
    Etot  = group.E.transform(np.sum)
    Efrac = d.E.values/Etot
    num_ev  = 1

    Zmean  = group.Z.transform(np.mean)
    Zvar   = group.Z.transform(np.var)
    Zstd   = group.Z.transform(np.std)

    d['num_ev'] = num_ev
    d['Etot']   = Etot
    d['Efrac']  = Efrac
    d['num_ev'] = num_ev
    d['Zmean']  = Zmean
    d['Zvar']   = Zvar
    d['Zstd']   = Zstd
    d['wTi']    = d['Efrac'] * d['Z']

    group2      = d.groupby(['event'])
    wT          = group2.wTi.transform(np.sum)
    ntot        = group2.num_ev.transform(np.sum)
    Efrac_check = group2.Efrac.transform(np.sum)

    d['ntot']   = ntot
    d['wT']     = wT
    d['sig_i']  = (d['Z'] - d['wT'])
    d['sig2_i'] = d['Efrac']*(d['Z'] - d['wT'])**2

    var      = group2.sig2_i.transform(np.sum)
    d['wVar'] = var 
    d['wStd'] = np.sqrt(var)

    # comment to check all steps
    #d = d.drop_duplicates('wVar', keep='first')

    return d
    

def compute_weighted_variance_old(dst_in):

    group = dst_in.groupby(['event'])
    Etot  = group.E.transform('sum')
    Efrac = dst_in.E.values/Etot
    d     = dst_in.assign(Efrac=Efrac)

    num_ev  = 1
    d['num_ev'] = num_ev
    d['Etot']   = Etot
    d['Twi']    = d['Efrac'] * d['Z']

    group2      = d.groupby(['event'])
    Tw          = group2.Twi.transform('sum')
    ntot        = group2.num_ev.transform('sum')
    Efrac_check = group2.Efrac.transform('sum')

    d['Efrac_check'] = Efrac_check 
    d['ntot']       = ntot - 1
    d['Tw']         = Tw
    d['sig_i']      = (d['Z'] - d['Tw'])
    d['sig2_i']     = d['Efrac']*(d['Z'] - d['Tw'])**2

    sig2sum      = group2.sig2_i.transform('sum')
    d['sig2sum'] = sig2sum 

#d['sig2'] = d['sig2sum']/d['ntot']
    d['sig2'] = d['sig2sum']
    d['sig']  = np.sqrt(d['sig2sum'])

    d = d.drop_duplicates('sig2', keep='first')

    return d


    
def compute_tau (dst_in):
    
    '''
    tau = time_bin_event - weighted_time_by_chargefraction
    '''
    
    group = dst_in.groupby(['event'])
    Etot  = group.E.transform('sum')
    # weight positions by sensor charge and normalized by event total charge
    Efrac   = dst_in.E.values/Etot
    num_ev  = 1
    d_intermediate = dst_in.assign(Efrac=Efrac)
    d = d_intermediate.assign(Etot=Etot)
    
    d['ZtimesEfrac'] = d['Z'] * d['Efrac']
    d['num_ev'] = num_ev
    
    group2 = d.groupby(['event'])
    wmeanT = group2.ZtimesEfrac.transform('sum')
    ntot   = group2.num_ev.transform('sum')
    dst = d.assign(wmeanT=wmeanT)
    dst['tau'] = dst['Z'] - dst['wmeanT']
    dst['ntot'] = ntot

    
    # plot selected energy
    sns.set_style("white")
    sns.set_style("ticks")

    fig = plt.figure(figsize=(14,10))
    ax      = fig.add_subplot(2, 2, 1)
    #(_) = h1(dst.tau[dst.ntot>8], 15, [-10,10], histtype='stepfilled',
    (_) = h1(dst.tau, 10, [-10,10], histtype='stepfilled',
             color='crimson',
             lbl='')
    plot_histo(PlotLabels('tau','Entries',''), ax)
    plt.legend(loc='upper right')


#    std_tau = np.std(dst.tau[dst.ntot>8])
    std_tau = np.std(dst.tau)
    
    return std_tau

def compute_tau_old (dst_in):
    
    '''
    tau = time_bin_event - weighted_time_by_chargefraction
    '''
    
    group = dst_in.groupby(['event'])
    Etot  = group.E.transform('sum')
    # weight positions by sensor charge and normalized by event total charge
    Efrac   = dst_in.E.values/Etot
    num_ev  = 1
    d_intermediate = dst_in.assign(Efrac=Efrac)
    d = d_intermediate.assign(Etot=Etot)
    
    d['ZtimesEfrac'] = d['Z'] * d['Efrac']
    d['num_ev'] = num_ev
    
    group2 = d.groupby(['event'])
    wmeanT = group2.ZtimesEfrac.transform('sum')
    ntot   = group2.num_ev.transform('sum')
    dst = d.assign(wmeanT=wmeanT)
    dst['tau'] = dst['Z'] - dst['wmeanT']
    dst['ntot'] = ntot

    
    # plot selected energy
    sns.set_style("white")
    sns.set_style("ticks")

    fig = plt.figure(figsize=(14,10))
    ax      = fig.add_subplot(2, 2, 1)
    #(_) = h1(dst.tau[dst.ntot>8], 15, [-10,10], histtype='stepfilled',
    (_) = h1(dst.tau, 10, [-10,10], histtype='stepfilled',
             color='crimson',
             lbl='')
    plot_histo(PlotLabels('tau','Entries',''), ax)
    plt.legend(loc='upper right')


#    std_tau = np.std(dst.tau[dst.ntot>8])
    std_tau = np.std(dst.tau)
    
    return std_tau

def prepare_to_fit_tau(dst_in):

    group = dst_in.groupby(['event'])
    Etot  = group.E.transform('sum')
    # weight positions by sensor charge and normalized by event total charge
    Efrac   = 1
    d_intermediate = dst_in.assign(Efrac=Efrac)
    d = d_intermediate.assign(Etot=Etot)
    
    group2 = d.groupby(['event'])
    meanT = group2.Z.transform('sum')
    nT = group2.Efrac.transform('sum')
    dst2 = d.assign(meanT=meanT)
    dst = dst2.assign(nT=nT)
    
    dst['tau'] = dst['Z'] - dst['meanT']/dst['nT']

    dst['Z2'] = dst['Z'] * dst['Z']
    group3 = dst.groupby(['event'])
    meanT2 = group3.Z2.transform('sum')

    dst['rms'] = dst['meanT2']/dst['nT'] - (dst['meanT']/dst['nT'])**2
    
    # plot selected energy
    sns.set_style("white")
    sns.set_style("ticks")
    
    fig = plt.figure(figsize=(14,10))
    ax      = fig.add_subplot(2, 2, 1)
    (_) = h1(dst.tau, 20, [-10,10], histtype='stepfilled',
             color='crimson',
             lbl='')
    plot_histo(PlotLabels('tau','Entries',''), ax)
    plt.legend(loc='upper right')
    
    return dst.tau








### Transversal
## From https://scipy-cookbook.readthedocs.io/items/FittingData.html


def gaussian(height, center_x, center_y, width_x, width_y):
    """Returns a gaussian function with the given parameters"""
    width_x = float(width_x)
    width_y = float(width_y)
    return lambda x,y: height*np.exp(
                -(((center_x-x)/width_x)**2+((center_y-y)/width_y)**2)/2)

def moments(data):
    """Returns (height, x, y, width_x, width_y)
    the gaussian parameters of a 2D distribution by calculating its
    moments """
    total = data.sum()
    X, Y = np.indices(data.shape)
    x = (X*data).sum()/total
    y = (Y*data).sum()/total
    col = data[:, int(y)]
    width_x = np.sqrt(np.abs((np.arange(col.size)-y)**2*col).sum()/col.sum())
    row = data[int(x), :]
    width_y = np.sqrt(np.abs((np.arange(row.size)-x)**2*row).sum()/row.sum())
    height = data.max()
    return height, x, y, width_x, width_y

def fitgaussian(data):
    """Returns (height, x, y, width_x, width_y)
    the gaussian parameters of a 2D distribution found by a fit"""
    params = moments(data)
    errorfunction = lambda p: np.ravel(gaussian(*p)(*np.indices(data.shape)) -
                                 data)
  
    out = optimize.leastsq(errorfunction, params, full_output=1)
   
    return out



#  Residual variance = reduced chi square = s_sq = sum[(f(x)-y)^2]/(N-n),
#where N is number of data points and n is the number of
#fitting parameters. Reduced chi square.
#More info:
#https://stackoverflow.com/questions/14854339/in-scipy-how-and-why-does-curve-fit-calculate-the-covariance-of-the-parameter-es

#infodict['fvec'] = f(x) -y.

#https://stackoverflow.com/questions/14581358/getting-standard-errors-on-fitted-parameters-using-the-optimize-leastsq-method-i

def fit_2D_Gaussian(z):
    
    plt.matshow(z, cmap='coolwarm')
    out = fitgaussian(z)

    fit_values = out[0]
    cov        = out[1]
    infodict   = out[2]

# compute covariance: need to multiply reduced chi square to cov from leastsq 
    s_sq = (infodict['fvec']**2).sum()/ (len(z)-len(out[0]))
    pcov = cov * s_sq

    fit_errors = np.sqrt(np.diag(pcov))

    # Plotting
    fit = gaussian(*out[0])
    plt.contour(fit(*np.indices(z.shape)), cmap='coolwarm')
    ax = plt.gca()
    
    (my_height, my_x, my_y, my_width_x, my_width_y) = out[0]
    covar = out[1]
    
    plt.colorbar().set_label("Number of events")
    
    plt.text(0.95, 0.05, """
    x : %.2f
    y : %.2f
    width_x : %.2f
    width_y : %.2f""" %(my_x, my_y, my_width_x, my_width_y),
             fontsize=16, horizontalalignment='right',
             verticalalignment='bottom', transform=ax.transAxes)

    #fit_errors = np.sqrt(np.diag(covar))

    print(out[3])
    print('fit converged:' +str(out[4]))
    return fit_values, fit_errors



