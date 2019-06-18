import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

#from . analysis_functions import kre_concat
from . histo_functions    import labels
from . histo_functions    import h1, h2, plot_histo
from . kr_types           import PlotLabels
from . kr_types           import KrEvent
#from  typing              import List, Tuple, Sequence, I2terable, Dict
from  typing              import List, Tuple, Sequence,  Dict

import sys
import logging
log = logging.getLogger()


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
   



def plot_2d_dst_vars(dst, zbins, s2bins, qbins, figsize=(14,10)):
    fig = plt.figure(figsize=figsize)
    
    fig.add_subplot(2, 2, 1)
    nevt, *_  = plt.hist2d(dst.Z, dst.E, (zbins, s2bins))
    plt.colorbar().set_label("Number of events")
    labels(PlotLabels("Z (mm)", "E (pes)", f"E vs Z"))

    fig.add_subplot(2, 2, 2)
    nevt, *_  = plt.hist2d(dst.Z, dst.Q, (zbins, qbins))
    plt.colorbar().set_label("Number of events")
    labels(PlotLabels("Z (mm)", "Q (pes)", f"Q vs Z"))


def plot_energy_distributions(dst, zbins, s2bins, qbins, figsize=(14,10)):
    fig = plt.figure(figsize=figsize)
    
    fig.add_subplot(2, 2, 1)
    nevt, *_  = plt.hist2d(dst.Z, dst.E, (zbins, s2bins))
    plt.colorbar().set_label("Number of events")
    labels(PlotLabels("Z (mm)", "E (pes)", f"E vs Z"))
    
    fig.add_subplot(2, 2, 2)
    nevt, *_  = plt.hist2d(dst.Z, dst.Q, (zbins, qbins))
    plt.colorbar().set_label("Number of events")
    labels(PlotLabels("Z (mm)", "Q (pes)", f"Q vs Z"))
    


def plot_xy_density(dst, xybins, figsize=(10,8)):
    fig = plt.figure(figsize=figsize)
    fig.add_subplot(1, 1, 1)
    XYpitch = np.diff(xybins)[0]
    nevt_full, *_ = plt.hist2d(dst.X, dst.Y, (xybins, xybins))
    plt.colorbar().set_label("Number of events")
    labels(PlotLabels("X (mm)", "Y (mm)", f"full distribution for {XYpitch:.1f} mm pitch"))
    return nevt_full


def plot_s1_vs_z(dst, zbins, s1bins, figsize=(10,8)):
    fig = plt.figure(figsize=figsize)
    fig.add_subplot(1, 1, 1)
    nevt, *_  = plt.hist2d(dst.Z, dst.S1e, (zbins, s1bins))
    plt.colorbar().set_label("Number of events")
    labels(PlotLabels("Z (mm)", "S1 (pes)", f"S1 vs Z"))


def plot_s2_vs_z(dst, zbins, s2bins, figsize=(10,8)):
    fig = plt.figure(figsize=figsize)
    fig.add_subplot(1, 1, 1)
    nevt, *_  = plt.hist2d(dst.Z, dst.S2e, (zbins, s2bins))
    plt.colorbar().set_label("Number of events")
    labels(PlotLabels("Z (mm)", "S2 (pes)", f"S2 vs Z"))


def plot_s2_vs_s1(dst, s1bins, s2bins, figsize=(10,8)):
    fig = plt.figure(figsize=figsize)
    fig.add_subplot(1, 1, 1)
    nevt, *_  = plt.hist2d(dst.S1e, dst.S2e, (s1bins, s2bins))
    plt.colorbar().set_label("Number of events")
    labels(PlotLabels("S1 (pes)", "S2 (pes)", f"S2 vs S1"))


def plot_q_vs_s2(dst, s2bins, qbins, figsize=(10,8)):
    fig = plt.figure(figsize=figsize)
    fig.add_subplot(1, 1, 1)
    nevt, *_  = plt.hist2d(dst.S2e, dst.S2q, (s2bins, qbins))
    plt.colorbar().set_label("Number of events")
    labels(PlotLabels("S2 (pes)", "Q (pes)", f"Q vs S2"))


def plot_s2e_vs_z_r_regions(kdsts, krBins, figsize=(14,10)):

    full, fid, core, hcore = kdsts

    fig = plt.figure(figsize=figsize)
    fig.add_subplot(2, 2, 1)
    nevt, *_  = plt.hist2d(full.Z, full.S2e, (krBins.Z, krBins.S2e))
    plt.colorbar().set_label("Number of events")
    labels(PlotLabels("Z (mm)", "S2e (pes)", f" full "))

    fig.add_subplot(2, 2, 2)
    nevt, *_  = plt.hist2d(fid.Z, fid.S2e, (krBins.Z, krBins.S2e))
    plt.colorbar().set_label("Number of events")
    labels(PlotLabels("Z (mm)", "S2e (pes)", f" fid "))

    fig.add_subplot(2, 2, 3)
    nevt, *_  = plt.hist2d(core.Z, core.S2e, (krBins.Z, krBins.S2e))
    plt.colorbar().set_label("Number of events")
    labels(PlotLabels("Z (mm)", "S2e (pes)", f" core "))

    fig.add_subplot(2, 2, 4)
    nevt, *_  = plt.hist2d(hcore.Z, hcore.S2e, (krBins.Z, krBins.S2e))
    plt.colorbar().set_label("Number of events")
    labels(PlotLabels("Z (mm)", "S2e (pes)", f" hard core Z"))
    plt.tight_layout()



def plot_Eg_vs_z_r_regions(kdsts, krBins, figsize=(14,10)):

    full, fid, core, hcore = kdsts

    fig = plt.figure(figsize=figsize)
    fig.add_subplot(2, 2, 1)
    nevt, *_  = plt.hist2d(full.Z, full.E, (krBins.Z, krBins.S2e))
    plt.colorbar().set_label("Number of events")
    labels(PlotLabels("Z (mm)", "Eg (pes)", f" full "))

    fig.add_subplot(2, 2, 2)
    nevt, *_  = plt.hist2d(fid.Z, fid.E, (krBins.Z, krBins.S2e))
    plt.colorbar().set_label("Number of events")
    labels(PlotLabels("Z (mm)", "Eg (pes)", f" fid "))

    fig.add_subplot(2, 2, 3)
    nevt, *_  = plt.hist2d(core.Z, core.E, (krBins.Z, krBins.S2e))
    plt.colorbar().set_label("Number of events")
    labels(PlotLabels("Z (mm)", "Eg (pes)", f" core "))

    fig.add_subplot(2, 2, 4)
    nevt, *_  = plt.hist2d(hcore.Z, hcore.E, (krBins.Z, krBins.S2e))
    plt.colorbar().set_label("Number of events")
    labels(PlotLabels("Z (mm)", "Eg (pes)", f" hard core Z"))
    plt.tight_layout()


def plot_Qg_vs_z_r_regions(kdsts, krBins, figsize=(14,10)):

    full, fid, core, hcore = kdsts

    fig = plt.figure(figsize=figsize)
    fig.add_subplot(2, 2, 1)
    nevt, *_  = plt.hist2d(full.Z, full.Q, (krBins.Z, krBins.S2q))
    plt.colorbar().set_label("Number of events")
    labels(PlotLabels("Z (mm)", "Qg (pes)", f" full "))

    fig.add_subplot(2, 2, 2)
    nevt, *_  = plt.hist2d(fid.Z, fid.Q, (krBins.Z, krBins.S2q))
    plt.colorbar().set_label("Number of events")
    labels(PlotLabels("Z (mm)", "Qg (pes)", f" fid "))

    fig.add_subplot(2, 2, 3)
    nevt, *_  = plt.hist2d(core.Z, core.Q, (krBins.Z, krBins.S2q))
    plt.colorbar().set_label("Number of events")
    labels(PlotLabels("Z (mm)", "Qg (pes)", f" core "))

    fig.add_subplot(2, 2, 4)
    nevt, *_  = plt.hist2d(hcore.Z, hcore.Q, (krBins.Z, krBins.S2q))
    plt.colorbar().set_label("Number of events")
    labels(PlotLabels("Z (mm)", "Qg (pes)", f" hard core Z"))
    plt.tight_layout()

def plot_s2q_vs_z_r_regions(kdsts, krBins, figsize=(14,10)):

    full, fid, core, hcore = kdsts

    fig = plt.figure(figsize=figsize)
    fig.add_subplot(2, 2, 1)
    nevt, *_  = plt.hist2d(full.Z, full.S2q, (krBins.Z, krBins.S2q))
    plt.colorbar().set_label("Number of events")
    labels(PlotLabels("Z (mm)", "S2q (pes)", f" full "))

    fig.add_subplot(2, 2, 2)
    nevt, *_  = plt.hist2d(fid.Z, fid.S2q, (krBins.Z, krBins.S2q))
    plt.colorbar().set_label("Number of events")
    labels(PlotLabels("Z (mm)", "S2q (pes)", f" fid "))

    fig.add_subplot(2, 2, 3)
    nevt, *_  = plt.hist2d(core.Z, core.S2q, (krBins.Z, krBins.S2q))
    plt.colorbar().set_label("Number of events")
    labels(PlotLabels("Z (mm)", "S2q (pes)", f" core "))

    fig.add_subplot(2, 2, 4)
    nevt, *_  = plt.hist2d(hcore.Z, hcore.S2q, (krBins.Z, krBins.S2q))
    plt.colorbar().set_label("Number of events")
    labels(PlotLabels("Z (mm)", "S2q (pes)", f" hard core Z"))
    plt.tight_layout()

def plot_s1e_vs_z_r_regions(kdsts, krBins, figsize=(14,10)):

    full, fid, core, hcore = kdsts

    fig = plt.figure(figsize=figsize)
    fig.add_subplot(2, 2, 1)
    nevt, *_  = plt.hist2d(full.Z, full.S1e, (krBins.Z, krBins.S1e))
    plt.colorbar().set_label("Number of events")
    labels(PlotLabels("Z (mm)", "S1e (pes)", f" full "))

    fig.add_subplot(2, 2, 2)
    nevt, *_  = plt.hist2d(fid.Z, fid.S1e, (krBins.Z, krBins.S1e))
    plt.colorbar().set_label("Number of events")
    labels(PlotLabels("Z (mm)", "S1e (pes)", f" fid "))

    fig.add_subplot(2, 2, 3)
    nevt, *_  = plt.hist2d(core.Z, core.S1e, (krBins.Z, krBins.S1e))
    plt.colorbar().set_label("Number of events")
    labels(PlotLabels("Z (mm)", "S1e (pes)", f" core "))

    fig.add_subplot(2, 2, 4)
    nevt, *_  = plt.hist2d(hcore.Z, hcore.S1e, (krBins.Z, krBins.S1e))
    plt.colorbar().set_label("Number of events")
    labels(PlotLabels("Z (mm)", "S1e (pes)", f" hard core Z"))
    plt.tight_layout()


def plot_energy_vs_t(kce, krBins, krRanges, figsize=(14,10)):


    fig = plt.figure(figsize=figsize)
    fig.add_subplot(2, 2, 1)
    (_) = h1(kce.T, bins = krBins.T, range =krRanges.T)

    fig.add_subplot(2, 2, 2)
    nevt, *_  = plt.hist2d(kce.T, kce.E, (krBins.T, krBins.S2e))
    plt.colorbar().set_label("Number of events")
    labels(PlotLabels("T", "E (pes)", f" E (corrected) vs T"))

    fig.add_subplot(2, 2, 3)
    nevt, *_  = plt.hist2d(kce.T, kce.Q, (krBins.T, krBins.S2q))
    plt.colorbar().set_label("Number of events")
    labels(PlotLabels("T", "Q (pes)", f" Q (corrected)  vs T"))

    fig.add_subplot(2, 2, 4)
    nevt, *_  = plt.hist2d(kce.T, kce.S1e, (krBins.T, krBins.S1e))
    plt.colorbar().set_label("Number of events")
    labels(PlotLabels("T", "S2q (pes)", f" S1 vs T"))
    plt.tight_layout()


def plot_s2e_vs_s1e_r_regions(kdsts, krBins, figsize=(14,10)):

    full, fid, core, hcore = kdsts

    fig = plt.figure(figsize=figsize)
    fig.add_subplot(2, 2, 1)
    nevt, *_  = plt.hist2d(full.S1e, full.S2e, (krBins.S1e, krBins.S2e))
    plt.colorbar().set_label("Number of events")
    labels(PlotLabels("S1e (pes)", "S2e (pes)", f" full "))

    fig.add_subplot(2, 2, 2)
    nevt, *_  = plt.hist2d(fid.S1e, fid.S2e, (krBins.S1e, krBins.S2e))
    plt.colorbar().set_label("Number of events")
    labels(PlotLabels("S1e (pes)", "S2e (pes)", f" fid "))

    fig.add_subplot(2, 2, 3)
    nevt, *_  = plt.hist2d(core.S1e, core.S2e, (krBins.S1e, krBins.S2e))
    plt.colorbar().set_label("Number of events")
    labels(PlotLabels("S1e (pes)", "S2e (pes)", f" core "))

    fig.add_subplot(2, 2, 4)
    nevt, *_  = plt.hist2d(hcore.S1e, hcore.S2e, (krBins.S1e, krBins.S2e))
    plt.colorbar().set_label("Number of events")
    labels(PlotLabels("S1e (pes)", "S2e (pes)", f" hard core Z"))
    plt.tight_layout()


def plot_s2q_vs_s2e_r_regions(kdsts, krBins, figsize=(14,10)):

    full, fid, core, hcore = kdsts

    fig = plt.figure(figsize=figsize)
    fig.add_subplot(2, 2, 1)
    nevt, *_  = plt.hist2d(full.S2e, full.S2q, (krBins.S2e, krBins.S2q))
    plt.colorbar().set_label("Number of events")
    labels(PlotLabels("S2e (pes)", "S2q (pes)", f" full "))

    fig.add_subplot(2, 2, 2)
    nevt, *_  = plt.hist2d(fid.S2e, fid.S2q, (krBins.S2e, krBins.S2q))
    plt.colorbar().set_label("Number of events")
    labels(PlotLabels("S2e (pes)", "S2q (pes)", f" fid "))

    fig.add_subplot(2, 2, 3)
    nevt, *_  = plt.hist2d(core.S2e, core.S2q, (krBins.S2e, krBins.S2q))
    plt.colorbar().set_label("Number of events")
    labels(PlotLabels("S2e (pes)", "S2q (pes)", f" core "))

    fig.add_subplot(2, 2, 4)
    nevt, *_  = plt.hist2d(hcore.S2e, hcore.S2q, (krBins.S2e, krBins.S2q))
    plt.colorbar().set_label("Number of events")
    labels(PlotLabels("S2e (pes)", "S2q (pes)", f" hard core Z"))
    plt.tight_layout()




def plot_s2e_vs_z_r_regions(kdsts, krBins, figsize=(14,10)):

    full, fid, core, hcore = kdsts

    fig = plt.figure(figsize=figsize)
    fig.add_subplot(2, 2, 1)
    nevt, *_  = plt.hist2d(full.Z, full.S2e, (krBins.Z, krBins.S2e))
    plt.colorbar().set_label("Number of events")
    labels(PlotLabels("Z (mm)", "S2e (pes)", f" full "))

    fig.add_subplot(2, 2, 2)
    nevt, *_  = plt.hist2d(fid.Z, fid.S2e, (krBins.Z, krBins.S2e))
    plt.colorbar().set_label("Number of events")
    labels(PlotLabels("Z (mm)", "S2e (pes)", f" fid "))

    fig.add_subplot(2, 2, 3)
    nevt, *_  = plt.hist2d(core.Z, core.S2e, (krBins.Z, krBins.S2e))
    plt.colorbar().set_label("Number of events")
    labels(PlotLabels("Z (mm)", "S2e (pes)", f" core "))

    fig.add_subplot(2, 2, 4)
    nevt, *_  = plt.hist2d(hcore.Z, hcore.S2e, (krBins.Z, krBins.S2e))
    plt.colorbar().set_label("Number of events")
    labels(PlotLabels("Z (mm)", "S2e (pes)", f" hard core Z"))
    plt.tight_layout()


def plot_energy_vs_z_histo_and_profile(e, zc, emean_z, esigma_z,
                                       bins_e = 50, range_e = (2e+3,14e+3),
                                       figsize=(10,6)):
    fig = plt.figure(figsize=figsize)
    pltLabels =PlotLabels(x='Energy-like', y='Events', title='true')
    ax      = fig.add_subplot(1, 2, 1)
    (_)     = h1(e, bins=50, range=(7e+3,11e+3))
    plot_histo(pltLabels, ax)

    ax      = fig.add_subplot(1, 2, 2)
    plt.errorbar(zc, emean_z, esigma_z, np.diff(zc)[0]/2, fmt="kp", ms=7, lw=3)

    plt.tight_layout()
