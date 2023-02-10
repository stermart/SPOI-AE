# imports - numpy & scipy
from cProfile import label
import matplotlib as mpl
import numpy as np

# imports - visualizations
import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1.anchored_artists import AnchoredSizeBar
from mpl_toolkits.axes_grid1.axes_divider import make_axes_locatable
import matplotlib.axes as axes
import matplotlib.figure as figure
from matplotlib.font_manager import FontProperties
from matplotlib.colorbar import Colorbar

# Utilities
from typing import List, Tuple, Union

def crop(us, spa, mask, disp, xaxis, yaxis, 
         xmin, xmax, ymin, ymax):
    
    xidxs = [np.argmin(np.abs(xaxis - x)) for x in [xmin, xmax]]
    yidxs = [np.argmin(np.abs(yaxis - y)) for y in [ymin, ymax]] 
    
    us_new = us[yidxs[0]:yidxs[1]+1, xidxs[0]:xidxs[1]+1]
    spa_new = spa[yidxs[0]:yidxs[1]+1, xidxs[0]:xidxs[1]+1, ...]
    mask_new = mask[yidxs[0]:yidxs[1]+1, xidxs[0]:xidxs[1]+1]
    disp_new = disp[yidxs[0]:yidxs[1]+1, xidxs[0]:xidxs[1]+1]
    xaxis_new = xaxis[xidxs[0]:xidxs[1]+1]
    yaxis_new = yaxis[yidxs[0]:yidxs[1]+1]
    
    return us_new, spa_new, mask_new, disp_new, xaxis_new, yaxis_new
    
def wl_crop(fgs, wls, wl):
    wlidx = np.argmin(np.abs(wls - wl))
    
    spa_new = fgs[..., wlidx]
    
    return spa_new   

# Plots of Summary Statistics

def err_hists(fig: figure.Figure, ax: axes.Axes, 
              spas, recons, wls, 
              lbl="", wl=None, c=None):
        
    if wl:
        wlidx = np.argmin(np.abs(wls - wl))
        spas_flat = [np.ravel(spa[:, wlidx]) for spa in spas]
        recons_flat = [np.ravel(recon[:, wlidx]) for recon in recons]
    else:
        spas_flat = [np.ravel(spa[:, :]) for spa in spas]
        recons_flat = [np.ravel(recon[:, :]) for recon in recons]
    
    errs = [(recons_flat[i] - spas_flat[i]) / (spas_flat[i] + 1e-10) * 100 
            for i in range(len(spas_flat))]
    
    if c is None:
        ax.hist(errs, bins=151, range=(-150, 150), stacked=True)
    else:
        ax.hist(errs, bins=151, range=(-150, 150), 
                stacked=True, color=c[:len(spas)])
        
    ax.tick_params(axis='both', which='major', labelsize=8)
    ax.tick_params(axis='both', which='minor', labelsize=8)
    ax.yaxis.get_offset_text().set_size(6.5)
    
    if lbl:
        ax.text(0.95, 0.95, lbl, 
                ha='right', va='top', 
                transform=ax.transAxes, 
                fontsize=10, fontweight='bold', c='k')

    return fig, ax 

def avg_spectrum(fig: figure.Figure, ax: axes.Axes, 
                 ref_data, recons, wls, 
                 lbl="", names=None, c=None, 
                 show_xlabel=True, show_ylabel=True): 
    
    avg_ref = np.mean(ref_data, axis=1)
    avg_recons = [np.mean(recon, axis=1) for recon in recons]
    
    if c is not None: 
        for i, avg_recon in enumerate(avg_recons): 
            ax.plot(wls, avg_recon, c=c[i], zorder=10*(len(recons)+1-i))
        ax.plot(wls, avg_ref, c="k", zorder=10*(len(recons)+1)-5)
    else:
        for i, avg_recon in enumerate(avg_recons): 
            ax.plot(wls, avg_recon, zorder=10*(len(recons)+1-i))
        ax.plot(wls, avg_ref, c="k", zorder=10*(len(recons)+1)-5)
            
    ax.tick_params(axis='both', which='major', labelsize=8)
    ax.tick_params(axis='both', which='minor', labelsize=8)
    
    if lbl:
        ax.text(0.25, 0.95, lbl, 
                ha='center', va='top', 
                transform=ax.transAxes, 
                fontsize=10, fontweight='bold', c='k')
    
    if names is not None:
        ax.legend(names + ["Input"], 
                  loc='upper right', 
                  fontsize=6.5)
        
    if show_xlabel:
        ax.set_xlabel("Wavelength (nm)", fontsize=10)
    
    if show_ylabel:
        ax.set_ylabel("Intensity (a.u.)", fontsize=10)
        ax.get_yaxis().set_visible(True)
    else:
        ax.get_yaxis().set_visible(False)
    
    return fig, ax

def learning_curve(fig: figure.Figure, ax: axes.Axes, 
                   loss_hists, refs, 
                   lbl="", names=None, c=None, 
                   show_xlabel=True, ylabel: str=""):
        
    if c is None:
        for i, loss_hist in enumerate(loss_hists):
            ax.semilogy(np.arange(len(loss_hist)), loss_hist, '-')
            
        for i, ref in enumerate(refs):
            ax.hlines(ref, 
                      xmin=0, xmax=len(loss_hists[0])-1,
                      linestyles="dashed")
    else:
        for i, loss_hist in enumerate(loss_hists):
            ax.semilogy(np.arange(len(loss_hist)), loss_hist, '-', c=c[i])
            
        for i, ref in enumerate(refs):
            ax.hlines(ref, 
                      xmin=0, xmax=len(loss_hists[0])-1,
                      linestyles="dashed", 
                      colors=c[i+len(loss_hists)])
    
    ax.tick_params(axis='both', which='major', labelsize=8)
    ax.tick_params(axis='both', which='minor', labelsize=8)
    
    if lbl:
        ax.text(0.95, 0.95, lbl, 
                ha='right', va='top', 
                transform=ax.transAxes, 
                fontsize=10, fontweight='bold', c='k')
    
    if names is not None:
        ax.legend(names, 
                  loc='upper left', 
                  fontsize=6.5)
        
    if show_xlabel:
        ax.set_xlabel("Epoch", fontsize=10)
    
    if ylabel:
        ax.set_ylabel(ylabel, fontsize=10)
        ax.get_yaxis().set_visible(True)
    else:
        ax.get_yaxis().set_visible(False)
    
    return fig, ax

def log_learning_curve(fig: figure.Figure, ax: axes.Axes, 
                       tr_loss, ts_loss, 
                       lbl="", c=None, xv_mode=False, 
                       show_xlabel=True, show_ylabel=True):
        
    assert len(tr_loss) == len(ts_loss)
    
    if c is None:
        ax.semilogy(np.arange(len(tr_loss)), tr_loss, 'k-')
        ax.semilogy(np.arange(len(ts_loss)), ts_loss, 'k:')
    else:
        ax.semilogy(np.arange(len(tr_loss)), tr_loss, 
                    marker='None', linestyle='-', c=c)
        ax.semilogy(np.arange(len(tr_loss)), tr_loss, 
                    marker='None', linestyle=':', c=c)
    if xv_mode:
        ax.legend(["Training", "Cross-Validation"], 
                  loc='upper right', 
                  fontsize=6)
    else:
        ax.legend(["Training", "Testing"], 
                  loc='upper right',
                  fontsize=6)
    
    ax.tick_params(axis='both', which='major', labelsize=8)
    ax.tick_params(axis='both', which='minor', labelsize=8)
    
    if lbl:
        ax.text(0.25, 0.95, lbl, 
                ha='center', va='top', 
                transform=ax.transAxes, 
                fontsize=10, fontweight='bold', c='k')
        
    if show_xlabel:
        ax.set_xlabel("Epoch", fontsize=10)
    
    if show_ylabel:
        ax.set_ylabel("Cost", fontsize=10)
    
    return fig, ax

def Hb_spectra(E_HbO2s, E_HHbs, wls,
      lbls=None, names=None, c=None, 
      show_xlabel=False, show_ylabel=False):
    
    fig, axs = plt.subplots(1, 2, sharex=True, sharey=True)
    
    for i, (E_HbO2, E_Hb) in enumerate(zip(E_HbO2s, E_HHbs)):
        if c is not None:
            axs[0].plot(wls, E_HbO2 / max(np.max(E_HbO2), np.max(E_Hb)), 
                        c=c[i], 
                        zorder=10*(len(E_HbO2s)-i))
            axs[1].plot(wls, E_Hb / max(np.max(E_HbO2), np.max(E_Hb)), 
                        c=c[i], 
                        zorder=10*(len(E_HbO2s)-i))       
        else:
            axs[0].plot(wls, E_HbO2 / max(np.max(E_HbO2), np.max(E_Hb)), 
                        zorder=10*(len(E_HbO2s)-i))
            axs[1].plot(wls, E_Hb / max(np.max(E_HbO2), np.max(E_Hb)), 
                        zorder=10*(len(E_HbO2s)-i))
                
    
    
    axs[0].tick_params(axis='both', which='major', labelsize=8)
    axs[0].tick_params(axis='both', which='minor', labelsize=8)
    
    axs[1].tick_params(axis='both', which='major', labelsize=8)
    axs[1].tick_params(axis='both', which='minor', labelsize=8)
    
    if show_xlabel:
        axs[0].set_xlabel("Wavelength (nm)", fontsize=10)
        axs[1].set_xlabel("Wavelength (nm)", fontsize=10)
    
    if show_ylabel:
        axs[0].set_ylabel("Absorption (a.u.)", fontsize=10)
    else:
        axs[0].get_yaxis().set_visible(False)
    axs[1].get_yaxis().set_visible(False)
    
    if lbls is not None:
        t0 = axs[0].text(0.95, 0.95, lbls[0],
                         ha='right', va='top',
                         transform=axs[0].transAxes,
                         fontsize=10, fontweight='bold', c='k',
                         zorder=100)
        t0.set_bbox(dict(facecolor="white", alpha=0.5, linewidth=0))
        t1 = axs[1].text(0.95, 0.95, lbls[1],
                         ha='right', va='top',
                         transform=axs[1].transAxes,
                         fontsize=10, fontweight='bold', c='k',
                         zorder=100)
        t1.set_bbox(dict(facecolor="white", alpha=0.7, linewidth=0))

    if names is not None:
        axs[0].legend(names,
                      loc='upper left',
                      fontsize=6.5)
    
    return fig, axs

def data_distr(fig: figure.Figure, ax: axes.Axes, 
               data: np.ndarray, 
               lbl: str="", c=None, 
               show_xlabel=False, show_ylabel=False):
    
    if c is not None:
        ax.hist(np.ravel(data), bins=75, range=(0, 1.5), color=c)
    else:
        ax.hist(np.ravel(data), bins=75, range=(0, 1.5))
    
    if show_xlabel:
        ax.set_xlabel("Intensity (a.u.)", fontsize=10)
    
    if show_ylabel:
        ax.set_ylabel("Count", fontsize=10)
    else:
        ax.get_yaxis().set_visible(False)

    ax.tick_params(axis='both', which='major', labelsize=8)
    ax.tick_params(axis='both', which='minor', labelsize=8)
    ax.yaxis.get_offset_text().set_size(8)
    
    if lbl:
        ax.text(0.95, 0.95, lbl, 
                ha='right', va='top', 
                transform=ax.transAxes, 
                fontsize=10, fontweight='bold', c='k') 
        
    return fig, ax

def R2(fig: figure.Figure, ax: axes.Axes, 
       r2_list, wls, 
       lbl="", names=None, c=None, 
       show_xlabel=False, show_ylabel=False): 
    
    for i in range(len(r2_list)-1,-1,-1):
        if c is not None:
            if names is not None:
                ax.plot(wls, r2_list[i], marker='.', 
                        linestyle='None', c=c[i], label=names[i])
            else:
                ax.plot(wls, r2_list[i], marker='.', 
                        linestyle='None', c=c[i])
        else:
            if names is not None:
                ax.plot(wls, r2_list[i], marker='.', 
                        linestyle='None', label=names[i])
            else:
                ax.plot(wls, r2_list[i], marker='.', linestyle='None')
            
    ax.set_ylim(bottom=min(0, np.min(r2_list)), 
                top=1.0)
    
    ax.tick_params(axis='both', which='major', labelsize=8)
    ax.tick_params(axis='both', which='minor', labelsize=8)
    
    if lbl and names is not None:
        ax.text(0.05, 0.35, lbl, 
                ha='left', va='bottom', 
                transform=ax.transAxes, 
                fontsize=10, fontweight='bold', c='k')
    elif lbl:
        ax.text(0.05, 0.05, lbl, 
                ha='left', va='bottom', 
                transform=ax.transAxes, 
                fontsize=10, fontweight='bold', c='k')
    
    if names is not None:
        handles, lables = ax.get_legend_handles_labels()
        ax.legend(handles[::-1], lables[::-1], loc='lower left', fontsize=6.5)
    
    if show_xlabel:    
        ax.set_xlabel("Wavelength (nm)", fontsize=10)
        
    if show_ylabel:
        ax.set_ylabel(r"$R^2$", fontsize=10)
    else:
        ax.get_yaxis().set_visible(False)
    
    return fig, ax   
    
# Image Processing Figures

def seg(psi, psi0, Ihat, us, xaxis, yaxis, lbls=None): 
    
    if lbls is None:
        lbls = ["A", "B", "C"]
    
    fig, axs = plt.subplots(1, 3)
    
    X, Y = np.meshgrid(xaxis, yaxis)
    
    fig, axs[0] = US(fig, axs[0], us, xaxis, yaxis, lbl=lbls[0])
    
    fig, axs[1] = US(fig, axs[1], Ihat, xaxis, yaxis, lbl=lbls[1])
    axs[1].contour(X, Y, psi0, [0], colors='red')
    
    fig, axs[2] = US(fig, axs[2], Ihat, xaxis, yaxis, lbl=lbls[2])
    axs[2].contour(X, Y, psi, [0], colors='red')
    
    return fig, axs 

def unmixing(us, C, mask, xaxis, yaxis, lbls=None): 
    
    if lbls is None:
        lbls = ["A", "B", "C"]
    
    fig, axs = plt.subplot(1, 3)
        
    fig, axs[0] = HbO2_con(fig, axs[0], 
                           us, C, mask, xaxis, yaxis, 
                           lbl=lbls[0])
    fig, axs[1] = Hb_con(fig, axs[1], 
                         us, C, mask, xaxis, yaxis, 
                         lbl=lbls[1])
    fig, axs[2] = SO2(fig, axs[2], 
                      us, C, mask, xaxis, yaxis, 
                      lbl=lbls[2])
    
    return fig, axs

def horizontal_cbar(fig: figure.Figure, cax: axes.Axes=None, ax=None,
                    fgmin=0, fgmax=1, 
                    lbl="",
                    cmap="viridis"):
    
    cmap0 = mpl.colormaps[cmap]
    cnorm0 = mpl.colors.Normalize(vmin=fgmin, vmax=fgmax)

    assert cax is not None or ax is not None, "Axes not specified"
    assert not (
        cax is not None and ax is not None), "Only one axes can be specified"
    
    if cax is not None:
        cbar = fig.colorbar(mpl.cm.ScalarMappable(norm=cnorm0, cmap=cmap0),
                            cax=cax,
                            orientation='horizontal',
                            shrink=0.8)
    elif ax is not None:
        cbar = fig.colorbar(mpl.cm.ScalarMappable(norm=cnorm0, cmap=cmap0),
                            ax=ax,
                            location='top',
                            orientation='horizontal',
                            shrink=0.8,
                            pad=0.01)
    if lbl and cax is not None:
        cax.set_xlabel(lbl, fontsize=8, fontweight='bold')
        cax.xaxis.set_label_position('top')
        cax.tick_params(axis='both', which='major', labelsize=6)
        cax.tick_params(axis='both', which='minor', labelsize=6)
        return fig, cax
    elif lbl and ax is not None:
        cbar.ax.set_xlabel(lbl, fontsize=8, fontweight='bold')
        cbar.ax.xaxis.set_label_position('top')
        cbar.ax.tick_params(axis='both', which='major', labelsize=6)
        cbar.ax.tick_params(axis='both', which='minor', labelsize=6)
        return fig, cbar.ax
    
    return None
    
def spa_recon_err(us, spa, recon, mask, xaxis, yaxis, wls, wl, lbls=None):
    
    if lbls is None:
        lbls = ["A", "B", "C"]
    
    fig, axs = plt.subplot(1, 3)
    
    fig, axs[0] = img(fig, axs[0], 
                      us, spa, mask, xaxis, yaxis, wls, wl, 
                      lbl=lbls[0])
    fig, axs[1] = img(fig, axs[1], 
                      us, recon, mask, xaxis, yaxis, wls, wl, 
                      lbl=lbls[1])
    fig, axs[2] = err(fig, axs[2], 
                      us, spa, recon, mask, xaxis, yaxis, wls, wl, 
                      lbl=lbls[2])
    
    fig.tight_layout(pad=0.4, w_pad=0.3, h_pad=0.3)
    
    return fig, axs

def HbO2_con(fig: figure.Figure, ax: axes.Axes, 
             us, C, mask, xaxis, yaxis, 
             fgmin=None, fgmax=None, 
             make_cbar=True, cbar_loc="right", 
             lbl=""):
    
    return nth_con(fig, ax, 
                 us, C, mask, xaxis, yaxis, 
                 fgmin=fgmin, fgmax=fgmax,
                 n=0, 
                 make_cbar=make_cbar, cbar_loc=cbar_loc,
                 lbl=lbl, cmap="Reds")
    
def Hb_con(fig: figure.Figure, ax: axes.Axes, 
           us, C, mask, xaxis, yaxis, 
           fgmin=None, fgmax=None, 
           make_cbar=True, cbar_loc="right",
           lbl=""):
    
    return nth_con(fig, ax, 
                 us, C, mask, xaxis, yaxis, 
                 fgmin=fgmin, fgmax=fgmax,
                 n=1, 
                 make_cbar=make_cbar, cbar_loc=cbar_loc, 
                 lbl=lbl, cmap="Blues")
    
def nth_con(fig: figure.Figure, ax: axes.Axes, 
          us, C, mask, xaxis, yaxis, 
          fgmin=None, fgmax=None,
          n=0, make_cbar=True, cbar_loc="right", 
          lbl="", cmap="Greens"):
    
    if fgmin is None:
        fgmin = 0
    if fgmax is None:
        fgmax = np.percentile(C[..., n], 99)
    return __magic2D(fig, ax, 
                     us, 
                     C[..., n], 
                     mask, xaxis, yaxis, 
                     fgmin=fgmin, fgmax=fgmax, 
                     make_cbar=make_cbar, cbar_loc=cbar_loc, 
                     lbl=lbl, cmap=cmap)
    
def SO2(fig: figure.Figure, ax: axes.Axes, 
        us, C, mask, xaxis, yaxis, 
        make_cbar=True, cbar_loc="right", 
        lbl=""):
    
    so2 = C[..., 0] / np.sum(C, axis=-1) * 100
    
    return __magic2D(fig, ax, 
                     us, 
                     so2, 
                     mask, xaxis, yaxis, 
                     fgmin=0, fgmax=100, 
                     make_cbar=make_cbar, cbar_loc=cbar_loc, 
                     lbl=lbl, cmap="RdBu_r")
    
def US(fig: figure.Figure, ax: axes.Axes, 
       us, xaxis, yaxis, 
       lbl=""): 
    
    return __magic2D(fig, ax, 
                     us, 
                     np.zeros(us.shape), 
                     np.ones(us.shape), 
                     xaxis, yaxis, 
                     make_cbar=False, lbl=lbl)
    
def disp(fig: figure.Figure, ax: axes.Axes, 
         us, disp, mask, xaxis, yaxis, 
         make_cbar=True, cbar_loc="right", lbl=""):
    
    return __magic2D(fig, ax,
                     us,  disp, mask, xaxis, yaxis,
                     fgmin=np.min(disp[mask==0]), 
                     fgmax=np.max(disp[mask==0]),
                     make_cbar=make_cbar, cbar_loc=cbar_loc, 
                     lbl=lbl)


def mua(fig: figure.Figure, ax: axes.Axes,
        us, mua, mask, xaxis, yaxis, wls, wl,
        fgmax=None,
        make_cbar=True, cbar_loc="right", 
        lbl=""):

    if fgmax is None:
        return __magicWL(fig, ax,
                         us, mua, mask, xaxis, yaxis, wls, wl,
                         fgmin=0,
                         make_cbar=make_cbar, cbar_loc=cbar_loc, 
                         lbl=lbl, cmap="cividis")
    else:
        return __magicWL(fig, ax,
                         us, mua, mask, xaxis, yaxis, wls, wl,
                         fgmin=0, fgmax=fgmax,
                         make_cbar=make_cbar, cbar_loc=cbar_loc, 
                         lbl=lbl, cmap="cividis")


def mus(fig: figure.Figure, ax: axes.Axes,
        us, mus, mask, xaxis, yaxis, wls, wl,
        fgmax=None,
        make_cbar=True, cbar_loc="right", 
        lbl=""):

    if fgmax is None:
        return __magicWL(fig, ax,
                         us, mus, mask, xaxis, yaxis, wls, wl,
                         fgmin=0,
                         make_cbar=make_cbar, cbar_loc=cbar_loc, 
                         lbl=lbl, cmap="cividis")
    else:
        return __magicWL(fig, ax,
                         us, mus, mask, xaxis, yaxis, wls, wl,
                         fgmin=0, fgmax=fgmax,
                         make_cbar=make_cbar, cbar_loc=cbar_loc, 
                         lbl=lbl, cmap="cividis")
    
def fluence(fig: figure.Figure, ax: axes.Axes, 
            us, flu, mask, xaxis, yaxis, wls, wl, 
            make_cbar=True, cbar_loc="right",
            lbl=""):
    
    return __magicWL(fig, ax, 
                     us, flu, mask, xaxis, yaxis, wls, wl, 
                     fgmin=0, 
                     make_cbar=make_cbar, cbar_loc=cbar_loc, 
                     lbl=lbl, cmap="viridis")

def psi(fig: figure.Figure, ax: axes.Axes, 
        us, psi, mask, xaxis, yaxis, wls, wl, 
        make_cbar=True, cbar_loc="right", 
        lbl=""):
    
    return __magicWL(fig, ax, 
                     us, psi, mask, xaxis, yaxis, wls, wl, 
                     fgmin=-1, fgmax=0, 
                     make_cbar=make_cbar, cbar_loc=cbar_loc, 
                     lbl=lbl, cmap="viridis")

def img(fig: figure.Figure, ax: axes.Axes, 
        us, spa, mask, xaxis, yaxis, wls, wl, 
        fgmax=None, 
        make_cbar=True, cbar_loc="right", 
        lbl=""): 
    
    if fgmax is None:
        return __magicWL(fig, ax, 
                        us, spa, mask, xaxis, yaxis, wls, wl, 
                        fgmin=0, 
                        make_cbar=make_cbar, cbar_loc=cbar_loc,
                        lbl=lbl, cmap="plasma")
    else:
        return __magicWL(fig, ax, 
                        us, spa, mask, xaxis, yaxis, wls, wl, 
                        fgmin=0, fgmax=fgmax,
                        make_cbar=make_cbar, cbar_loc=cbar_loc,
                        lbl=lbl, cmap="plasma")
    
def err(fig: figure.Figure, ax: axes.Axes, 
        us, spa, recon, mask, xaxis, yaxis, wls, wl, 
        make_cbar=True, cbar_loc="right", lbl=""): 
    
    spa_wl = wl_crop(spa, wls, wl)
    recon_wl = wl_crop(recon, wls, wl)
    err_img = (recon_wl - spa_wl) / (spa_wl + 1e-10) * 100
    
    return __magic2D(fig, ax, 
                     us, err_img, mask, xaxis, yaxis, 
                     fgmin=-150, fgmax=150, 
                     make_cbar=make_cbar, cbar_loc=cbar_loc, 
                     lbl=lbl, cmap="PiYG") 

# Magic Functions
# TODO: Relative coordinates for ax.text calls

def __magic2D(fig: figure.Figure, ax: axes.Axes, 
             us, fg, mask, xaxis, yaxis, 
             fgmin=0, fgmax=np.inf, 
             make_cbar=True, cbar_loc="right", 
             lbl="", cmap="viridis"):
    
    ax.imshow(us, 
              cmap='gray',
              extent=[np.min(xaxis), np.max(xaxis), 
                      np.max(yaxis), np.min(yaxis)]
              )
    alpha = np.zeros(mask.shape)
    alpha[mask == 0] = 1
    im = ax.imshow(fg, 
                   cmap=cmap, 
                   vmin=fgmin, vmax=fgmax,
                   alpha=alpha,
                   extent=[np.min(xaxis), np.max(xaxis), 
                           np.max(yaxis), np.min(yaxis)]
                   )

    bar = AnchoredSizeBar(ax.transData, 
                          1, '1 mm', 'upper left', 
                          pad=0.5,
                          color='white', 
                          frameon=False,
                          size_vertical=0.1, 
                          label_top=False, 
                          sep=4, 
                          fontproperties=FontProperties(size=8, weight='bold')
                          )
    ax.add_artist(bar)

    if make_cbar and cbar_loc == 'right':
        div = make_axes_locatable(ax)
        cax = div.append_axes('right', size='5%', pad=0.1)
        fig.colorbar(im, cax=cax, orientation='vertical')
        cax.yaxis.set_ticks_position('right')
        cax.tick_params(axis='both', which='major', labelsize=5)
        cax.tick_params(axis='both', which='minor', labelsize=5)
        
    if make_cbar and cbar_loc == 'top':        
        div = make_axes_locatable(ax)
        cax = div.append_axes('top', size='5%', pad=0.1)
        fig.colorbar(im, cax=cax, orientation='horizontal')
        cax.xaxis.set_ticks_position('top')
        cax.tick_params(axis='both', which='major', labelsize=5)
        cax.tick_params(axis='both', which='minor', labelsize=5)
        
    if lbl:
        ax.text(0.95, 0.95, lbl, 
                fontsize=10, fontweight="bold", c="w", 
                ha='right', va='top', 
                transform=ax.transAxes)
    
    ax.axis('off')
    ax.set_aspect('equal')
    
    return fig, ax

def __magicWL(fig: figure.Figure, ax: axes.Axes, 
              us, fgs, mask, xaxis, yaxis, wls, wl, 
              fgmin=None, fgmax=None, 
              make_cbar=True, cbar_loc="right", 
              lbl="", cmap="viridis"):
    
    ax.imshow(us, 
              cmap='gray',
              extent=[np.min(xaxis), np.max(xaxis), 
                      np.max(yaxis), np.min(yaxis)]
              )
    alpha = np.zeros(mask.shape)
    alpha[mask == 0] = 1
    
    fg = wl_crop(fgs, wls, wl)
    
    if fgmin is None:
        fgmin = np.percentile(fg, 1)
    if fgmax is None:
        fgmax = np.percentile(fg, 99)
    
    im = ax.imshow(fg, 
                   cmap=cmap, 
                   vmin=fgmin, vmax=fgmax, 
                   alpha=alpha, 
                   extent=[np.min(xaxis), np.max(xaxis), 
                           np.max(yaxis), np.min(yaxis)]
                   )

    bar = AnchoredSizeBar(ax.transData, 
                          1, '1 mm', 'upper left', 
                          pad=0.5,
                          color='white', 
                          frameon=False,
                          size_vertical=0.1, 
                          label_top=False, 
                          sep=4, 
                          fontproperties=FontProperties(size=8, weight='bold')
                          )
    ax.add_artist(bar)

    if make_cbar and cbar_loc == 'right':
        div = make_axes_locatable(ax)
        cax = div.append_axes('right', size='5%', pad=0.1)
        fig.colorbar(im, cax=cax, orientation='vertical')
        cax.yaxis.set_ticks_position('right')
        cax.tick_params(axis='both', which='major', labelsize=5)
        cax.tick_params(axis='both', which='minor', labelsize=5)
        
    if make_cbar and cbar_loc == 'top':        
        div = make_axes_locatable(ax)
        cax = div.append_axes('top', size='5%', pad=0.1)
        fig.colorbar(im, cax=cax, orientation='horizontal')
        cax.xaxis.set_ticks_position('top')
        cax.tick_params(axis='both', which='major', labelsize=5)
        cax.tick_params(axis='both', which='minor', labelsize=5)
        
    if lbl:
        ax.text(0.95, 0.95, lbl, 
                fontsize=10, fontweight="bold", c="w", 
                ha='right', va='top', 
                transform=ax.transAxes)
    
    ax.text(0.025, 0.6, f"λ={wl:d}nm", 
            fontsize=8, fontweight="bold", c="w", 
            ha='left', va='center_baseline', 
            transform=ax.transAxes)
    
    ax.axis('off')
    ax.set_aspect('equal')
    
    return fig, ax

    



