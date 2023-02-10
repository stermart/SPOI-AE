# imports - numpy & scipy
from matplotlib.font_manager import FontProperties
import numpy as np
from numpy import linalg as la

# imports - sklearn
from skimage.measure import find_contours

# imports - visualizations
import matplotlib.pyplot as plt
import matplotlib.path as path
from matplotlib.widgets import LassoSelector
from mpl_toolkits.axes_grid1.anchored_artists import AnchoredSizeBar

import pa_util

def segment(fname, nu=1e2, iters=750):
    
    # load image
    spa, us, mask, disp, xaxis, yaxis = pa_util.load_sPA_img(fname)
    dx = xaxis[1]-xaxis[0]
    dy = yaxis[1]-yaxis[0]
    
    # normalize w.r.t. wavelength axis
    spa_norm = la.norm(spa, axis=2)
    spa_norm99 = np.percentile(spa_norm, 99)
    spa_norm[spa_norm > spa_norm99] = spa_norm99
    
    # generate initial level-set
    X, Y = np.meshgrid(xaxis, yaxis)
    psi0 = np.full(spa_norm.shape, np.inf)
    mask[mask == 0] = -1
    mask = np.pad(mask, ((1,1), (1,1)), mode="constant", constant_values=1)
    curves = find_contours(mask, level=0)
    curve = curves[0]
    for i, point in enumerate(curve):
        yidx, xidx = point
        x = np.interp(xidx, np.arange(len(xaxis)), xaxis)
        y = np.interp(yidx, np.arange(len(yaxis)), yaxis)
        psi0 = np.minimum(psi0, np.sqrt((X-x)**2 + (Y-y)**2))
    psi0[mask[1:-1, 1:-1] == -1] = -psi0[mask[1:-1, 1:-1] == -1]
    
    # main loop
    psi = np.copy(psi0)
    for i in range(iters):
        alpha, u, v, contrast = __calc_alpha(spa_norm, psi)
        dt = __cfl(alpha, nu, dx, dy)
        psi = __evolve(psi, alpha, dt, dx, dy, nu)
    
    # done!
    return psi, spa_norm

def segment_interactive(fname, nu=1e2, iters=750):
    
    # load image
    spa, us, mask, disp, xaxis, yaxis = pa_util.load_sPA_img(fname)
    dx = xaxis[1]-xaxis[0]
    dy = yaxis[1]-yaxis[0]
    
    # normalize w.r.t. wavelength axis
    spa_norm = la.norm(spa, axis=2)
    spa_norm99 = np.percentile(spa_norm, 99)
    spa_norm[spa_norm > spa_norm99] = spa_norm99
    
    fig, axs = plt.subplots(1,2)
    fig.tight_layout(h_pad=3)

    axs[0].imshow(us,
            cmap="gray",
            extent=[np.min(xaxis), np.max(xaxis), 
                    np.max(yaxis), np.min(yaxis)])
    axs[0].set_ylabel("Depth Axis (nm)")
    axs[0].set_xlabel("Lateral Axis (nm)")
    axs[0].set_title("Ultrasound Image")

    axs[1].imshow(spa_norm, 
            cmap="viridis",
            extent=[np.min(xaxis), np.max(xaxis), 
                    np.max(yaxis), np.min(yaxis)])
    axs[1].set_ylabel("Depth Axis (nm)")
    axs[1].set_xlabel("Lateral Axis (nm)")
    axs[1].set_title("Wavlength-Normalize sPA Image")

    def onSelect(x):
        global curve, curve_path
        curve_path = path.Path(x)
        curve = np.array(x)
        axs[1].plot(curve[:,0], curve[:,1], "r-")

    lineprops = {"color": "red",
                "linewidth": "2", 
                "alpha": 0.7}
    lasso = LassoSelector(ax=axs[1], 
                        onselect=onSelect, 
                        lineprops=lineprops,
                        button=1)
    
    fig.canvas.draw()
    plt.show(block=False)
    input("Press enter to accept selection >> ")
    
    # generate initial level-set
    X, Y = np.meshgrid(xaxis, yaxis)
    psi0 = np.full(spa_norm.shape, np.inf)
    for point in curve:
        x,y = point
        dist_mat = np.sqrt((X-x)**2 + (Y-y)**2)
        update_mask = dist_mat < psi0
        psi0[update_mask] = dist_mat[update_mask]
    inside = curve_path.contains_points(np.vstack([np.ravel(X), np.ravel(Y)]).T)
    inside = np.reshape(inside, spa_norm.shape)
    psi0[inside] = -psi0[inside]
    
    # main loop
    psi = np.copy(psi0)
    for i in range(iters):
        alpha, u, v, contrast = __calc_alpha(spa_norm, psi)
        dt = __cfl(alpha, nu, dx, dy)
        psi = __evolve(psi, alpha, dt, dx, dy, nu)
    
    # done!
    return psi, psi0, spa_norm

def plot_seg(psi, Ihat, xaxis, yaxis):
    plt.figure()
    plt.imshow(Ihat, 
               cmap='viridis', 
               extent=[np.min(xaxis), np.max(xaxis), 
                       np.max(yaxis), np.min(yaxis)]
               )
    plt.contour(psi, [0], colors='red')
    
def plot_evolution(psi, psi0, Ihat, us, xaxis, yaxis,
                   savename=None):
    
    fig, axs = plt.subplots(1, 3)
    X, Y = np.meshgrid(xaxis, yaxis)
    
    axs[0].imshow(us, 
                  cmap="gray", 
                  extent=[np.min(xaxis), np.max(xaxis), 
                          np.max(yaxis), np.min(yaxis)]
                  )
    axs[0].axis('off')
    bar0 = AnchoredSizeBar(axs[1].transData, 
                           1, '1 mm', 'upper right', 
                           pad=0.5,
                           color='white', 
                           frameon=False,
                           size_vertical=0.1,
                           label_top=False,
                           sep=4, 
                           fontproperties=FontProperties(size=12, 
                                                         weight='bold')
                           )
    axs[0].text(0.95, 0.95, "US", 
                ha='right', va='top', 
                transform=axs[0].transAxes, 
                fontsize=24, fontweight="bold", c="w")
    axs[0].add_artist(bar0)
    axs[0].set_aspect('equal')
    
    axs[1].imshow(Ihat, 
                  cmap="gray", 
                  extent=[np.min(xaxis), np.max(xaxis), 
                          np.max(yaxis), np.min(yaxis)]
                  )
    axs[1].contour(X, Y, psi0, [0], colors='red')
    axs[1].axis('off')
    bar1 = AnchoredSizeBar(axs[1].transData, 
                           1, '1 mm', 'upper right', 
                           pad=0.5,
                           color='white', 
                           frameon=False,
                           size_vertical=0.1,
                           label_top=False,
                           sep=4, 
                           fontproperties=FontProperties(size=12, 
                                                         weight='bold')
                           )
    axs[1].text(0.95, 0.95, "Hand", 
                ha='right', va='top', 
                transform=axs[0].transAxes, 
                fontsize=24, fontweight="bold", c="w")
    axs[1].add_artist(bar1)
    axs[1].set_aspect('equal')
    
    axs[2].imshow(Ihat, 
                  cmap="gray", 
                  extent=[np.min(xaxis), np.max(xaxis), 
                          np.max(yaxis), np.min(yaxis)]
                  )
    axs[2].contour(X, Y, psi, [0], colors='red')
    axs[2].axis('off')
    bar2 = AnchoredSizeBar(axs[2].transData, 
                           1, '1 mm', 'upper right', 
                           pad=0.5,
                           color='white', 
                           frameon=False,
                           size_vertical=0.1,
                           label_top=False,
                           sep=4, 
                           fontproperties=FontProperties(size=12, 
                                                         weight='bold')
                           )
    axs[2].text(0.95, 0.95, "C-V", 
                ha='right', va='top', 
                transform=axs[0].transAxes, 
                fontsize=24, fontweight="bold", c="w")
    axs[2].add_artist(bar2)
    axs[2].set_aspect('equal')
    
    return fig, axs    
    
def plot_ls_evolution(psi, psi0, xaxis, yaxis):
    
    fig, axs = plt.subplots(1, 2)
    X, Y = np.meshgrid(xaxis, yaxis)
    
    axs[0].imshow(psi0, 
                  cmap="viridis", 
                  extent=[np.min(xaxis), np.max(xaxis), 
                          np.max(yaxis), np.min(yaxis)]
                  )
    axs[0].contour(X, Y, psi0, [0], colors='red')
    axs[0].axis('off')
    bar0 = AnchoredSizeBar(axs[0].transData, 
                           1, '1 mm', 'upper right', 
                           pad=0.5,
                           color='white', 
                           frameon=False,
                           size_vertical=0.1,
                           label_top=False,
                           sep=4, 
                           fontproperties=FontProperties(size=12, 
                                                         weight='bold')
                           )
    axs[0].text(4, 10, "A", fontsize=24, fontweight="bold", c="w")
    axs[0].add_artist(bar0)
    axs[0].set_aspect('equal')
    
    axs[1].imshow(psi, 
                  cmap="viridis", 
                  extent=[np.min(xaxis), np.max(xaxis), 
                          np.max(yaxis), np.min(yaxis)]
                  )
    axs[1].contour(X, Y, psi, [0], colors='red')
    axs[1].axis('off')
    bar0 = AnchoredSizeBar(axs[1].transData, 
                           1, '1 mm', 'upper right', 
                           pad=0.5,
                           color='white', 
                           frameon=False,
                           size_vertical=0.1,
                           label_top=False,
                           sep=4, 
                           fontproperties=FontProperties(size=12, 
                                                         weight='bold')
                           )
    axs[1].text(4, 10, "B", fontsize=24, fontweight="bold", c="w")
    axs[1].add_artist(bar0)
    axs[1].set_aspect('equal')
    
    return fig, axs
    
    
def __cfl(alpha, nu, dx, dy):
    max_alpha = np.max(np.abs(alpha))
    dxx = min(dx, dy)
    cfl_transport = dxx / (np.sqrt(2) * max_alpha)
    cfl_heat = dxx**2 / (4 * nu)
    cfl = min(cfl_transport, cfl_heat)
    return cfl

def __heat_pde(psi, dx, dy):
    psi_pad = np.pad(psi, ((1,1), (1,1)))
    
    psi_x = np.zeros(psi_pad.shape)
    psi_x[:, 1:-1] = (psi_pad[:, 2:] - psi_pad[:,:-2]) / (2*dx)
    psi_x = psi_x[1:-1, 1:-1]
    
    psi_y = np.zeros(psi_pad.shape)
    psi_y[1:-1, :] = (psi_pad[2:, :] - psi_pad[:-2, :]) / (2*dy)
    psi_y = psi_y[1:-1, 1:-1]
    
    psi_xx = np.zeros(psi_pad.shape)
    psi_xx[:, 1:-1] = (psi_pad[:, 2:] - 2*psi_pad[:, 1:-1] + psi_pad[:, :-2]) / (dx**2)
    psi_xx = psi_xx[1:-1, 1:-1]
    
    psi_yy = np.zeros(psi_pad.shape)
    psi_yy[1:-1, :] = (psi_pad[2:, :] - 2*psi_pad[1:-1, :] + psi_pad[:-2, :]) / (dy**2)
    psi_yy = psi_yy[1:-1, 1:-1]
    
    psi_xy = np.zeros(psi_pad.shape)
    psi_xy[1:-1, :] =  (psi_pad[2:, :] - psi_pad[:-2, :]) / (2*dy)
    psi_xy = psi_xy[1:-1, 1:-1]
    
    psi_t = (psi_x**2*psi_yy - 2*psi_x*psi_y*psi_xy + psi_y**2*psi_xx) / (psi_x**2 + psi_y**2 + 1e-10)
    
    return psi_t

def __transport_pde(psi, alpha, dx, dy):
    psi_pad = np.pad(psi, ((1,1), (1,1)))
    
    Dxp = np.zeros(psi_pad.shape)
    Dxp[:, :-1] = (psi_pad[:, 1:] - psi_pad[:, :-1]) / dx
    Dxp = Dxp[1:-1, 1:-1]
    
    Dxn = np.zeros(psi_pad.shape)
    Dxn[:, 1:] = (psi_pad[:, 1:] - psi_pad[:, :-1]) / dx
    Dxn = Dxn[1:-1, 1:-1]
    
    Dyp = np.zeros(psi_pad.shape)
    Dyp[:-1, :] = (psi_pad[1:, :] - psi_pad[:-1, :]) / dy
    Dyp = Dyp[1:-1, 1:-1]
    
    Dyn = np.zeros(psi_pad.shape)
    Dyn[1:, :] = (psi_pad[1:, :] - psi_pad[:-1, :]) / dy
    Dyn = Dyn[1:-1, 1:-1]
    
    alpha_pos = alpha >= 0
    alpha_neg = alpha < 0
    
    psi_t_pos = -alpha * np.sqrt(np.maximum(0, Dxn)**2 
                                 + np.minimum(0, Dxp)**2
                                 + np.maximum(0, Dyn)**2
                                 + np.minimum(0, Dyp)**2)
    psi_t_neg = -alpha * np.sqrt(np.minimum(0, Dxn)**2 
                                 + np.maximum(0, Dxp)**2
                                 + np.minimum(0, Dyn)**2
                                 + np.maximum(0, Dyp)**2)
    psi_t = np.zeros(psi.shape)
    psi_t[alpha_pos] = psi_t_pos[alpha_pos]
    psi_t[alpha_neg] = psi_t_neg[alpha_neg]
    
    return psi_t

def __calc_alpha(I_hat, psi):
    interior = psi < 0
    exterior = psi >= 0
    
    I_in = I_hat[interior]
    I_out = I_hat[exterior]
    
    u = np.sum(I_in) / len(I_in)
    v = np.sum(I_out) / len(I_out)
    
    alpha = 2 * (v-u) * ((u+v)/2 - I_hat)
    
    # Weber contrast
    contrast = (u-v) / v
    
    return alpha, u, v, contrast
    
def __evolve(psi, alpha, dt, dx, dy, nu):
    psi_transport = __transport_pde(psi, alpha, dx, dy)
    psi_heat = __heat_pde(psi, dx, dy)
    
    new_psi = psi + dt * (psi_transport + nu * psi_heat)
    
    return new_psi

if __name__ == '__main__':
    img = "M1 Right LN Spectro_2020-09-08-16-03-25_data"
    fname = f"data/All/{img}.mat"
    psi, Ihat = segment_interactive(fname, nu=5e2, iters=500)
    
    plt.figure()
    plt.imshow(Ihat, cmap='viridis')
    plt.contour(psi, [0], colors='red')
    plt.show()