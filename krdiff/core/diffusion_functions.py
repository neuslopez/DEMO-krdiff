import numpy as np
import math as math
import matplotlib.pyplot as plt
import scipy             as sci

from  invisible_cities. icaro. hst_functions import shift_to_bin_centers

import invisible_cities.core.core_functions as coref
import invisible_cities.core.fit_functions  as fitf
import invisible_cities.reco.corrections    as corrf
import invisible_cities.icaro.hst_functions as histFun
import invisible_cities.core.stat_functions as statf
import scipy.signal

def h1d(x, bins=None, range=None, xlabel='Variable', ylabel='Number of events',
        title=None, legend = 'upper right', weights = None, ax = None):

    if ax is None:
        fig = plt.figure(figsize=figsize)
        ax = fig.add_subplot(1, 1, 1)
    if weights is None:
        weights = [1.]*len(x)

    ax.set_xlabel(xlabel)#,fontsize = 11)
    ax.set_ylabel(ylabel)#, fontsize = 11)
    yhist, xhist, _ = ax.hist(x,
			    bins= bins,
			    range=range,
		   	    histtype='step',
			    edgecolor='black',
			    linewidth=1.5,
                            weights=weights,
			    rasterized=True)
    plt.grid(True)

    if title:
        plt.title(title)

    return yhist, shift_to_bin_centers(xhist)

def fround(number):
    """
    Given a number, return the closest integer.
    """
    
    if number%int(number) > 0.5:
        return math.ceil(number)
    return math.floor(number)

def sqrt(x, k, c):
    """
    Function y = sqrt(c + k·x)

    Parameters:
    c = constant term
    k = diffusion term
    """
    return np.sqrt(c+k*x)

def lin(x, k, c):
    """
    Function y = c + k·x
    
    c = y-intercept
    k = slope
    """
    
    return c+k*x

def rfun(x, rc, s):
    """
    Function y = (1/(1 + exp((x-rc)/s)))

    rc = Parameter such as x-rc is negative (therefore, bigger than the chamber size)
    s = slope
    """

    return (1/(1 + np.exp((x-rc)/s)))

def AddXY(df):
    """
    Add the X and Y mean position of the events to a grouped hits DST 'df' (grouped using groupby, by event id).
    """
    df['Xmean'] = (df['X']*df['E']).sum()/df['E'].sum()
    df['Ymean'] = (df['Y']*df['E']).sum()/df['E'].sum()

    return(df)

def AddSlicePos(df):
    """
    Add the relative slice position of a slice to a grouped hits DST 'df' (group using groupby, by event id).
    
    Slice position = 0 will correspond to the highest energy slice.
    """
    
    max_pos = np.argmax(df.E.values)
    df['PosSlice'] = np.arange(-max_pos, len(df)-max_pos,1)
    
    return(df)

def lifetime_calculation(z, zrange, energy, erange, elim, slope, axes):
    """
    Measure the lifetime for the given events and plots the raw energy distribution and the energy vs drift time in a separate axes instance.

    Parameters:
    z = Array of events' drift time
    zrange = Range in drift time for the fit.
    energy = Array of events' energy.
    erange = Range in energy for plotting.
    elim = Limits for the energy at z=0.
    slope = slope of the exponential, used in the selection of the events, should be the inverse of the expected lifetime (1/[expected lifetime])
    axes = Axes object from a subplot, should have length >= 2.
    """

    axes[0].hist(energy, 50, erange, rasterized=True)
    histFun.labels("S2 energy (pes)", "Entries", "Fiducialized energy spectrum")

    low_cut   = elim[0] * np.exp(-slope*z)
    high_cut  = elim[1] * np.exp(-slope*z)
    sel       = coref.in_range(energy, low_cut, high_cut) # remove low and high E background

    axes[1].hist2d(z, energy, (100, 50), range=(zrange, erange), rasterized=True)
    x, y, u_y = fitf.profileX(z[sel], energy[sel], 100, xrange=zrange, yrange=erange)

    
    axes[1].plot(x, y, "--k", rasterized=True)
    axes[1].plot(z[z<500],  low_cut[z<500], "k.", rasterized=True)
    axes[1].plot(z[z<500], high_cut[z<500], "k.", rasterized=True)

    Zrange_LT = zrange
    
    seed = np.max(y), (x[15] - x[5])/np.log(y[15]/y[5])
    f    = fitf.fit(fitf.expo, x, y, seed, fit_range=Zrange_LT, sigma=u_y)

    axes[1].plot(x, f.fn(x), "r", lw=3, rasterized=True)
    print("Energy at z=0 = {:.1f} +- {:.1f}".format( f.values[0], f.errors[0]))
    print("Lifetime      = {:.1f} +- {:.1f}".format(-f.values[1], f.errors[1]))
    print("Chi2          = {:.2f}          \n".format(f.chi2))

    #    axes[1].text(zrange[0] + 0.05*(zrange[1]-zrange[0]), erange[0] + 1000, \
    #        "Lifetime = {:.1f} $\pm$ {:.1f}".format(-f.values[1], f.errors[1]), color = "white")#, fontsize = 14)
    
    histFun.labels("Drift time ($\mu$s)", "S2 energy (pes)", "")

    return np.array([-f.values[1], f.errors[1]]), corrf.LifetimeCorrection(-f.values[1], f.errors[1])

def measureDriftVelocity(run, drift_field, z, zrange_dv, bins_dv, buffersize, z_cathode, el_time):
    """
    Given a Z distribution, returns the drift velocity with its error.
    The drift velocity is obtained through an smooth Heavyside fit or logistic function (https://en.wikipedia.org/wiki/Logistic_function).

    Parameters:
    run = Run number, to add it to the figure title
    drift_field = Drift field of the run, to add it to the figure title.
    groupDST = Grouped DST (by event id).
    zrange_dv = Range in drift time for the fit.
    bins_dv = Bins for plotting the region of interest.
    buffersize = Buffersize of the run, to plot the drift time distribution.
    z_cathode = Cathode z-position in mm.
    """
    
    fig, axes = plt.subplots(1, 3, figsize=(24,6))
    st = fig.suptitle("Run {0} (drift field = {1:.0f} V/cm/bar) Z distribution".format(run, drift_field[0]), fontsize = 16)
    
    axes[0].hist(z, 100, [0, buffersize], rasterized=True)
    axes[0].set_xlabel("Z (mm)")
    axes[0].set_ylabel("")

    axes[1].hist(z, 100, [0, buffersize], rasterized=True)
    axes[1].set_xlabel("Z (mm)")
    axes[1].set_ylabel("")    
    axes[1].set_yscale('log')

    plt.axes(axes[2])
    y, x = h1d(z, bins_dv, zrange_dv, "Z (mm)", "Number of events", ax = axes[2])
    axes[2].set_yscale('log')

    sigmoid  = lambda x, A, B, C, D: A/(1 + np.exp(-C*(x-B))) + D
    seed     = np.max(y), np.mean(zrange_dv), np.diff(zrange_dv)[0]/100, np.min(y)
    f        = fitf.fit(sigmoid, x, y, seed, fit_range=zrange_dv,
                        sigma = statf.poisson_sigma(y))
    
    plt.plot(x, f.fn(x), "r", lw=1.25, rasterized=True)
    histFun.labels("Drift time ($\mu$s)", "Number of events", "")

    dv = z_cathode[0]/(f.values[1] - el_time[0])
    dv_e_stat = z_cathode[0]/((f.values[1] - el_time[0])**2) * f.errors[1] # Statistic error derived from fit.
    dv_e_syst = (z_cathode[0]/((f.values[1] - el_time[0])**2) * 1.)**2 + \
                (z_cathode[1]/(f.values[1] - el_time[0]))**2 + \
                (z_cathode[0]/((f.values[1] - el_time[0])**2) * el_time[1])**2 # Systematic error derived from drift time measurement (asumed to be 1 mus), drift length and drifted time in EL.
    dv_e_syst = np.sqrt(dv_e_syst)
    
    dv = np.array([dv, dv_e_stat, dv_e_syst])
    
    print("Max drift time = {:.3f} +- {:.3f}".format(f.values[1], f.errors[1]))
    print("Drift velocity   = {:.5f} +- {:.5f} (stat.) +- {:.5f} (syst.)".format(dv[0], dv[1], dv[2]))
    print("======================================")
    axes[2].annotate("Drift time = \n{:.3f} $\pm$ {:.3f}".format(f.values[1], f.errors[1]),
                     xy = (0.665, 0.9),
                     xycoords = 'axes fraction',
                     color = "black")#, fontsize = 14)
    
    extent = axes[0].get_tightbbox(fig.canvas.get_renderer()).transformed(fig.dpi_scale_trans.inverted())
    fig.savefig("DriftVel_R{}.pdf".format(run), bbox_inches=extent)
    extent = axes[1].get_tightbbox(fig.canvas.get_renderer()).transformed(fig.dpi_scale_trans.inverted())
    fig.savefig("DriftVel_Log_R{}.pdf".format(run), bbox_inches=extent)
    extent = axes[2].get_tightbbox(fig.canvas.get_renderer()).transformed(fig.dpi_scale_trans.inverted())
    fig.savefig("DriftVel_LogZoom_R{}.pdf".format(run), bbox_inches=extent)     
    plt.close(fig)
    
    return dv

def radialCuts(R, Z, E, ZCorr, elim, run_number = 0, fPlot = False):
    """
    Returns a R-dependent lower and a higher cut to the Z-corrected energy of the event.

    Parameters:
    R = R position of the event.
    Z = Drift time of the event.
    E = Energy of th event.
    ZCorr = Lifetime Correction object.
    elim = Limits for the energy at R=0.
    fPlot = If true, draw the E vs R distribution and the selection (between the two black lines).
    """

    low_cut   = elim[0] * rfun(R, 205, 20) #(1/(1 + np.exp((R-205)/20)))
    high_cut  = elim[1] * rfun(R, 220, 20) #(1/(1 + np.exp((R-220)/20)))

    if fPlot:
        fig = plt.figure(figsize=(8,6))
        ax1 = fig.add_subplot(111)
        plt.axes(ax1)
        histFun.hist2d(R, ZCorr(Z).value*E, (100, 50), range=([0, 150], [elim[0] - 3000, elim[1] + 3000]), new_figure=False, rasterized=True)
        histFun.labels("Radial position (mm)", "S2 energy (pes)", "")
        plt.plot(R,  low_cut, "k.", rasterized=True)
        plt.plot(R, high_cut, "k.", rasterized=True)

        fig.tight_layout()
        extent = ax1.get_tightbbox(fig.canvas.get_renderer()).transformed(fig.dpi_scale_trans.inverted()) 
        fig.savefig("ECorrvsR_R{}.pdf".format(run_number), bbox_inches=extent)     
        plt.close(fig)
        
    return low_cut, high_cut

def transversalPSF(groupDST, mapDST, sipm_lim, run_number = 0, sigma_lim = [0.1,12.1], sigma_step = 1., z_step = 25, \
                   low_drift_int = [10., 75.], zrange = [0, 550], el_time = [0., 0.], cut = 0.02, min_count = 5, norm=True):
    """
    Given a dataset, it divides it by drift time intervals and returns an array
    with the transverse rms (gaussian sigma) of each interval and the chi2 of each fit. 
    This fit is done to the PSF (point-spread function).

    Parameters:
    groupDST = Grouped DST (by event id).
    mapDST = Full raw DST
    sipm_lim = Limit in the distance point-sensor for the fit.
    sigma_lim = Range of RMS that will be evaluated.
    sigma_step = Step between sigmas to be evaluated.
    z_step = Width of the drift time intervals.
    low_drift_int = Drift time interval to obtain the PSF. Near anode so diffusion is negligible.
    zrange = Drift time range to be evaluated.
    cut = Minimum charge in a bin to be used for the fit.
    min_count = Minimum number of events in a bin to be used for the fit.
    norm = If true, SiPM charge is normalized to the charge of the SiPM with most charge.
    """
    z_scan = np.arange(zrange[0], zrange[1], z_step)

    sigmaAll = []
    chiAll = []
    
#    DX = mapDST.Xevt.values - mapDST.X.values
#    DY = mapDST.Yevt.values - mapDST.Y.values

    DX = mapDST.X.values - mapDST.Xpeak.values
    DY = mapDST.Y.values - mapDST.Ypeak.values

    # 
    groupQ = np.array([1.] * len(mapDST))
    if norm:
        groupQ = groupDST.Q.transform('sum').values
    groupZ = groupDST.Z.transform("mean") - el_time[0]

    sipm_xyrange = (-sipm_lim, sipm_lim)
    binfactor = 2.

    # Obtain the PSF
    base_selection = (np.sqrt(DX**2 + DY**2) < 50.) & (mapDST.Q > 5.) & (mapDST.Q.values/groupQ < 1.)
    low_drift_selection = base_selection & coref.in_range(groupZ, low_drift_int[0], low_drift_int[1])

    fig1 = plt.figure()
    ax1 = fig1.add_subplot(111)

    (xc, yc, Ec_o, Ece_o), _, cb = \
    histFun.hist2d_profile(DX[low_drift_selection], DY[low_drift_selection], mapDST[low_drift_selection].Q.values / groupQ[low_drift_selection], \
                           int(binfactor*sipm_lim), int(binfactor*sipm_lim), sipm_xyrange, sipm_xyrange,  new_figure = False, rasterized=True)
    cb.set_label("Charge fraction")
    histFun.labels("X (mm)", "Y (mm)", "")   

    #    extent = ax1.get_tightbbox(fig1.canvas.get_renderer()).transformed(fig1.dpi_scale_trans.inverted()) 
    ax1.set_xticks(np.linspace(-20., 20., 5))
    ax1.set_yticks(np.linspace(-20., 20., 5))
    fig1.tight_layout()
    fig1.savefig("TransPSF_EL_R{}.pdf".format(run_number))

    histFun.labels("X (mm)", "Y (mm)", "Energy vs XY for Z = [{0},{1}]".format(low_drift_int[0], low_drift_int[1]))   

    #Normalize the measured charge distribution
    Ec_o = Ec_o/Ec_o.sum()
    Ece_o = Ece_o/Ec_o.sum()
    
    # Obtain the set of PSF+gaussian convolutions to be tested (one per evaluated sigma)
    sigma_range = np.arange(sigma_lim[0], sigma_lim[1], sigma_step)
    Ec_conv = []
    
    mu = 0.    
    for sigma in sigma_range:
        gaussian = np.outer(np.exp(-np.power(xc - mu, 2.) / (2 * np.power(sigma, 2.))), \
                             np.exp(-np.power(yc - mu, 2.) / (2 * np.power(sigma, 2.))))
        gaussian = gaussian/gaussian.sum()
        Ec_c = sci.signal.convolve2d(Ec_o, gaussian, mode='same')
        Ec_conv.append(Ec_c)
        
    # Fit to obtain the sigma that best matches the distribution.
    for i,z in enumerate(z_scan):
        sigmamin, chi = psf_fit(mapDST, DX, DY, groupZ, groupQ, [z, z+z_step], \
                           np.array(Ec_conv), sigma_range, sipm_lim, cut, min_count, run_number)
        sigmaAll.append(sigmamin)
        chiAll.append(chi)

    return np.array(sigmaAll), np.array(chiAll)

def diffFit(sigma, z, zfit, dv, pressure, temp, diff_type, ax, fHisto2D = False):
    """
    Calculates (through a linear fit) and returns the transverse diffusion coefficient for a given dataset.

    Parameters:
    sigma = Longitudinal or transverse spread.
    Z = Drift time of the event.
    zfit = Drift time interval to do the fit.
    dv = Drift velocity under current conditions.
    pressure = Pressure in the chamber. 
    diff_type = Type of spread being considered; if not 'longitudinal', it will be transverse.
    ax = Subfigure to plot distributions.
    fHisto2D = Flag to plot a 2D distribution.
    """    

    seed = 0.03,0.01
    sel = coref.in_range(z, zfit[0], zfit[1])
    f = fitf.fit(lin, z[sel], sigma[sel]**2, seed=seed)

    if fHisto2D:
        x, _, _, _ = histFun.hist2d(z[sel], sigma[sel]**2, 100, [[zfit[0], zfit[1]],[0., (sigma[sel].max())**2+2]], new_figure=False, rasterized=True)        
        histFun.labels("Drift time ($\mu$s)", "Longitudinal RMS$^2$ ($\mu$s$^2$)", "")
        fit_label = "[({:.4f} +- {:.4f})·x + \n ({:.4f} +- {:.4f})]".format(f.values[0], f.errors[0], f.values[1], f.errors[1])
        ax.plot(z[sel], f.fn(z[sel]), 'r', label=fit_label if fit_label not in plt.gca().get_legend_handles_labels()[1] else '', rasterized=True)

    else:
        ax.plot(z, f.fn(z), 'r', label="[({:.4f} +- {:.4f})·x + \n ({:.4f} +- {:.4f})]".format(f.values[0], f.errors[0], f.values[1], f.errors[1]), rasterized=True)

    ax.legend(fontsize=12)
    const = f.values[0]
    const_e = f.errors[0]    
    D_e_stat = 0.
    D_e_syst = 0.    

    if diff_type == "transverse":
        print("Type Trans")
        D = const/2*1000000/100
        D_e_stat = (const_e/2)*1000000/100

    else:
        D = dv[0]*dv[0]*const/2*1000000/100
        D_e_stat = 1000000/100*np.sqrt((dv[0]*dv[0]*const_e/2)**2 + (const*dv[0]*dv[1])**2)
        D_e_syst = (const*dv[0]*dv[2])*1000000/100


    # Reduced diffusion coefficient (in um/cm^2)
    temp0 = 20. + 273.15
    D = np.sqrt(1000*2*D*pressure[0]/dv[0]*temp0/temp[0]) 

    # Error derived from fits (diffusion and drift velocity)
    new_D_e_stat = (D_e_stat * (1000*2*pressure[0]/dv[0]*temp0/temp[0]) / (2*np.sqrt(1000*2*D*pressure[0]/dv[0]*temp0/temp[0]))) ** 2 + \
                   (np.sqrt(1000*2*D*pressure[0]/dv[0]*temp0/temp[0])*dv[1]/(2*dv[0]))**2 
    D_e_stat = np.sqrt(new_D_e_stat)
    
    # Error derived from systematics (pressure, drift velocity and temperature)
    new_D_e_syst = (D_e_syst    * (1000*2*pressure[0]/dv[0]*temp0/temp[0]) / (2*np.sqrt(1000*2*D*pressure[0]/dv[0]*temp0/temp[0]))) ** 2 + \
                   (pressure[1] * (1000*2*D/dv[0]*temp0/temp[0])           / (2*np.sqrt(1000*2*D*pressure[0]/dv[0]*temp0/temp[0]))) ** 2 + \
                   (temp[1]     * np.sqrt(1000*2*D*pressure[0]/dv[0]*temp0/temp[0]) / (2*temp[0]))                                  ** 2 + \
                   (dv[2]       * np.sqrt(1000*2*D*pressure[0]/dv[0]*temp0/temp[0]) / (2*dv[0]))                                    ** 2  
    D_e_syst = np.sqrt(new_D_e_syst)                  
    
    c = np.sqrt(np.abs(f.values[1]))
    c_e = 0.5*np.abs(f.errors[1])/np.sqrt(np.abs(f.values[1]))
    
    print("Diffusion coefficient (um^2/cm^-1/2) = {:.4f} +- {:.4f} +- {:.4f}".format(D, D_e_stat, D_e_syst))
    print("Constant parameter (mm) = {:.4f} +- {:.4f}\n".format(c, c_e))
    
    D = [D, D_e_stat, D_e_syst]
    C = [c, c_e]

    return D, C

def diffusionFit(run_number, sigmaAll, zrange, zfit, step, dv = [1.,0], pressure = [7., 0.], temp = [20., 0.], diff_type = "transverse"):
    """
    Preparation for the diffusion fit (creates subfigure, creates drift time divisions and calls the fitting function).
    Returns the diffusion coefficient and the residual constant spread.
    
    Parameters:
    run = Run number being analyzed.
    sigmaAll = Longitudinal or transverse spread.
    zrange = Limits to the drift times considered to do the divisions.
    zfit = Drift time interval to do the fit.
    step = Length of the drift time divisions.
    dv = Drift velocity under current conditions.
    pressure = Pressure in the chamber. 
    diff_type = Type of spread being considered; if not 'longitudinal', it will be transverse.
    """
    
    sigma = np.array(sigmaAll)
    z_scan_diff = np.arange(zrange[0]+step/2., zrange[1]+step/2.,step)
    fig1 = plt.figure()
    ax1 = fig1.add_subplot(111)
    ax1.scatter(z_scan_diff, sigma**2, label="")
    ax1.set_xlabel("Drift time ($\mu$s)")
    if diff_type == "longitudinal":
        ax1.set_ylabel("$\sigma^2$ ($\mu$s$^2$)")
    else:
        ax1.set_ylabel("$\sigma^2$ (mm$^2$)")
        
    fig1.suptitle("Run {0}, {1} diffusion".format(run_number, diff_type, fontsize = 30))
    
    diff, const = diffFit(sigma, z_scan_diff, zfit, dv, pressure, temp, diff_type, ax1)

    #    ax1.legend(fontsize=12)

    extent = ax1.get_tightbbox(fig1.canvas.get_renderer()).transformed(fig1.dpi_scale_trans.inverted()) 
    fig1.savefig("Diffusion_{0}_R{1}.pdf".format(diff_type, run_number), bbox_inches=extent)    

    return np.array(diff), np.array(const)

def longitudinalPSF(groupDST, mapDST, run_number = 0, sigma_lim = [0.1,12.1], \
                   sigma_step = 1., z_step = 25, low_drift_int = [10., 75.], zrange = [0, 550], el_time = [0., 0.], norm=True):
    """
    Given a dataset, it divides it by drift time intervals and returns an array
    with the longitudinal rms (gaussian sigma) of each interval. This is done using
    the PSF along Z (longitudinal point-spread function).

    Parameters:
    groupDST = Grouped DST (by event id).
    mapDST = Full raw DST
    sigma_lim = Range of RMS that will be evaluated.
    sigma_step = Step between sigmas to be evaluated.
    z_step = Width of the drift time intervals.
    low_drift_int = Drift time interval to obtain the PSF. Near anode so diffusion is negligible.
    zrange = Drift time range to be evaluated.
    norm = If true, SiPM charge is normalized to the charge of the SiPM with most charge.
    """
    z_scan = np.arange(zrange[0], zrange[1], z_step)

    sigmaAll = []

    groupE = np.array([1.] * len(mapDST))
    if norm:
        groupE = groupDST.E.transform('sum').values

    groupZ = groupDST.Z.transform("mean") - el_time[0]

    nbins = 20
    
    # Obtain the PSF
    fig1 = plt.figure(figsize=(4.5, 4.))
    ax1 = fig1.add_subplot(111)
    plt.axes(ax1)
    low_drift_selection = coref.in_range(groupZ, low_drift_int[0], low_drift_int[1])
    Ec_o, xc  = h1d(mapDST[low_drift_selection].PosSlice.values, nbins+1, [-nbins/2.-0.5, nbins/2+0.5],\
                      "Relative time position ($\mu$s)", "Energy fraction",
                      weights = mapDST[low_drift_selection].E/(len(set(mapDST[low_drift_selection].event))*groupE[low_drift_selection]), ax = ax1)
    Ec_o = Ec_o/Ec_o.sum()

    extent = ax1.get_tightbbox(fig1.canvas.get_renderer()).transformed(fig1.dpi_scale_trans.inverted()) 
    fig1.savefig("DiffusionLongPSF_EL_R{0}.pdf".format(run_number), bbox_inches=extent) 
    histFun.labels("Relative time position ($\mu$s)",\
                   "Energy fraction",\
                   "Energy per slice for Z = [{0},{1}]".format(low_drift_int[0], low_drift_int[1]))
    
    # Obtain the set of PSF+gaussian convolutions to be tested (one per evaluated sigma)
    sigma_range = np.arange(sigma_lim[0], sigma_lim[1], sigma_step)
    Ec_conv = []
    
    mu = 0.    
    for sigma in sigma_range:
        gaussian = np.exp(-np.power(xc - mu, 2.) / (2 * np.power(sigma, 2.)))
        gaussian = gaussian/gaussian.sum()
        Ec_c = sci.signal.convolve(Ec_o, gaussian, mode='same')
        Ec_c = Ec_c/Ec_c.sum() 
        Ec_conv.append(Ec_c)
        
    n_row = 4
    nrows = len(z_scan)//n_row
    if len(z_scan)%n_row>0:
        nrows +=1        
    fig, axes = plt.subplots(nrows, n_row, figsize=(18, 4*nrows))

    # Loop through the different drift time intervals
    for i, z in enumerate(z_scan):
        chimin = 1e10
        sigmamin = 0.
        Ec_fit = Ec_o
        
        z_interval_selection = coref.in_range(groupZ, z, z + z_step)

        plt.axes(axes[i//n_row][i%n_row])

    
        Ec, xc, = h1d(mapDST[z_interval_selection].PosSlice, nbins+1, [-nbins/2.-0.5, nbins/2+0.5],\
                      "Relative time position ($\mu$s)", "Energy fraction",\
                       weights = mapDST[z_interval_selection].E/(len(set(mapDST[z_interval_selection].event))*groupE[z_interval_selection]), ax = axes[i//n_row][i%n_row])
    
        # Fit to obtain the sigma that best matches the distribution.
        for j, sigma in enumerate(sigma_range):
            Ec_c = Ec.sum()*Ec_conv[j]
            chi = ((Ec-Ec_c)**2/Ec_c).sum()/(len(Ec)-1)
            if chi < chimin:
                chimin = chi
                sigmamin = sigma
                Ec_fit = Ec_c
                
        plt.plot(xc, Ec_fit, "r", rasterized=True)
        sigmaAll.append(sigmamin)
        
    fig.tight_layout()
    for zcount, axi in enumerate(np.ravel(axes)):
        plt.axes(axi)
        extent = axi.get_tightbbox(fig.canvas.get_renderer()).transformed(fig.dpi_scale_trans.inverted()) 
        fig.savefig("DiffusionLongPSF_Z{0}-{1}_R{2}.pdf".format(zrange[0] + zcount*z_step, zrange[0] + (zcount+1)*z_step, run_number), bbox_inches=extent)        
        histFun.labels("Relative time position ($\mu$s)",\
                       "Energy fraction",\
                       "Energy per slice for Z = [{0},{1}]".format(zrange[0] + zcount*z_step, zrange[0] + (zcount+1)*z_step))        
    plt.close(fig)
    
    return np.array(sigmaAll)

def psf_fit(mapDST, DX, DY, groupZ, groupQ, z_int, Ec_conv, sigma_range, sipm_lim, cut, min_count, run_number):
    """
    Calculates (through a data-driven fit) and returns the transverse spread for a given dataset.

    Parameters:
    mapDST = Full raw DST.
    DX = Difference in position between event and sensor in x-dimension.
    DY = Difference in position between event and sensor in y-dimension.
    groupQ = Charge of the SiPMs.
    groupZ = Mean drift time of the event.
    z_int = Drift time interval to do the fit.
    Ec_conv = Collection (for several sigmas) of PSF+gaussian convolutions (for the fit).
    sigma_range = Collection of sigmas considered in the convolution.
    sipm_lim = Limit in the distance point-sensor for the fit.
    cut = Minimum charge in a bin to be used for the fit.
    min_count = Minimum number of events in a bin to be used for the fit.
    """
    fig, axes = plt.subplots(1, 3, figsize=(24,6))

    chimin = 1e20
    sipm_xyrange = (-sipm_lim, sipm_lim)
    binfactor = 2.
    sigmamin = 0.
    bin_lim_min = -50.
    sipm_lim_min = -50.
    nbins = int(binfactor*sipm_lim)

    # Plot the distribution to be fitted.
    base_selection = ((np.sqrt(DX**2 + DY**2) < 50.) & (mapDST.Q > 0.) & (mapDST.Q.values/groupQ < 1.))
    z_interval_selection = base_selection & coref.in_range(groupZ, z_int[0], z_int[1])           

    xc, Ec, Ece = fitf.profileX(DX[z_interval_selection], mapDST[z_interval_selection].Q/groupQ[z_interval_selection],\
                                  nbins, sipm_xyrange)    
    
    try:
        x_min = xc[(Ec<cut) & (xc<0)].max()
        x_max = xc[(Ec<cut) & (xc>0)].min()    
    except: 
        x_min = xc.min()
        x_max = xc.max()

    fig1 = plt.figure()
    ax1 = fig.add_subplot(1, 1, 1)
    (xc, yc, Ec_m, Ece_m), _, cb = \
    histFun.hist2d_profile(DX[z_interval_selection], DY[z_interval_selection], mapDST[z_interval_selection].Q.values / groupQ[z_interval_selection],\
                           nbins, nbins, sipm_xyrange, sipm_xyrange,  new_figure = False, rasterized=True)
    cb.set_label("Charge fraction")
    histFun.labels("X (mm)", "Y (mm)", "")
    ax1.set_xticks(np.linspace(-20., 20., 5))
    ax1.set_yticks(np.linspace(-20., 20., 5))
    fig1.tight_layout()
    fig1.savefig("TransPSF_Z{0}-{1}_R{2}.pdf".format(z_int[0], z_int[1], run_number))

    fig, axes = plt.subplots(1, 3, figsize=(24,6))        
    plt.axes(axes[0])
    (xc, yc, Ec_m, Ece_m), _, cb = \
    histFun.hist2d_profile(DX[z_interval_selection], DY[z_interval_selection], mapDST[z_interval_selection].Q.values / groupQ[z_interval_selection],\
                           nbins, nbins, sipm_xyrange, sipm_xyrange,  new_figure = False, rasterized=True)
    cb.set_label("E (pes)")
    histFun.labels("X (mm)", "Y (mm)", "")
        
    # Fit.
    Ec_fit = 0
    
    Ec = Ec_m
    Ece = Ece_m

    counts, *_ = np.histogram2d(DX[z_interval_selection], DY[z_interval_selection],\
                                [np.linspace(-sipm_lim, sipm_lim, nbins+1), np.linspace(-sipm_lim, sipm_lim, nbins+1)])
                            
    for lim in np.arange(sipm_lim, sipm_lim+1, 1.):
        bin_lim = [len(xc[xc<x_min]), len(xc[xc<x_max])+1]
        Ec = Ec_m[bin_lim[0]:bin_lim[1],bin_lim[0]:bin_lim[1]]
        Ece = Ece_m[bin_lim[0]:bin_lim[1],bin_lim[0]:bin_lim[1]]
        counts_sel = counts[bin_lim[0]:bin_lim[1],bin_lim[0]:bin_lim[1]]
        for i, sigma in enumerate(sigma_range):
            Ec_c = Ec_conv[i]
            Ec_c = Ec_c[bin_lim[0]:bin_lim[1],bin_lim[0]:bin_lim[1]]
            Ec_c = Ec_c*Ec.sum()/Ec_c.sum()
            sel = counts_sel > min_count
            chi = ((Ec[sel]-Ec_c[sel])**2/(Ece[sel]**2)).sum()/(len(Ec[sel])-1)
            if chi < chimin:
                chimin = chi
                sigmamin = sigma
                Ec_fit = Ec_c
                
    # Plot the projection at x = 0 / y = 0
    xproj = Ec_m[np.argmax(Ec_m.sum(axis=0))]
    yproj = Ec_m[:,np.argmax(Ec_m.sum(axis=0))]
    
    xprojf = Ec_fit[np.argmax(Ec_fit.sum(axis=0))]
    yprojf = Ec_fit[:,np.argmax(Ec_fit.sum(axis=1))]

    h1d(xc, len(xc), [np.min(xc), np.max(xc)], weights = xproj, ax = axes[1])
    axes[1].plot(xc[bin_lim[0]:bin_lim[1]], xprojf, "r", lw=1.25, rasterized=True)
    axes[1].set_xlabel("X (mm)")
    axes[1].set_ylabel("Charge fraction")
    axes[1].grid(True)

    h1d(yc, len(yc), [np.min(yc), np.max(yc)], weights = yproj, ax=axes[2])
    axes[2].plot(yc[bin_lim[0]:bin_lim[1]], yprojf, "r", lw=1.25, rasterized=True)
    axes[2].set_xlabel("Y (mm)")
    axes[2].set_ylabel("Charge fraction")
    axes[2].grid(True)

    fig.tight_layout()
    
    extent = axes[0].get_tightbbox(fig.canvas.get_renderer()).transformed(fig.dpi_scale_trans.inverted()) 
    axes[0].set_xticks(np.linspace(-20., 20., 5))
    axes[0].set_yticks(np.linspace(-20., 20., 5))
    #    fig.savefig("TransPSF_Z{0}-{1}_R{2}.pdf".format(z_int[0], z_int[1], run_number), bbox_inches=extent)
    extent = axes[1].get_tightbbox(fig.canvas.get_renderer()).transformed(fig.dpi_scale_trans.inverted()) 
    fig.savefig("TransPSF_X_Z{0}-{1}_R{2}.pdf".format(z_int[0], z_int[1], run_number), bbox_inches=extent)
    extent = axes[2].get_tightbbox(fig.canvas.get_renderer()).transformed(fig.dpi_scale_trans.inverted()) 
    fig.savefig("TransPSF_Y_Z{0}-{1}_R{2}.pdf".format(z_int[0], z_int[1], run_number), bbox_inches=extent)

    axes[0].set_title("Energy vs XY for Z = [{0},{1}]".format(z_int[0], z_int[1])) 
    axes[1].set_title("Charge profile X projection at Y = 0") 
    axes[2].set_title("Charge profile Y projection at X = 0")
    fig.tight_layout()
    plt.close(fig)

    return sigmamin, chimin
