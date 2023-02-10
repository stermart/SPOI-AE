# imports - numpy & scipy
import numpy as np

# imports - utils
import os
import json
import dill

# imports - pytorch
import torch
from torch import cuda

# imports - visualizations
import matplotlib.pyplot as plt
from matplotlib.cm import get_cmap

# spoiae libraries
import pa_nmf
import pa_util
import spoiae
import pa_viz

def main():
    
    # parameters
    fpath = "Data"
    fname = 'example_sPA_image'
    model_dir = "Paper Models"
    model_name = "Model_5e0_Adapt"
    state_name = "final_state"
    plot_wls = [680, 760, 970]
    
    # CUDA
    print(f"CUDA Available? {cuda.is_available()}")
    if cuda.is_available():
        available_gpus = [(i, torch.cuda.get_device_name(torch.cuda.device(i))) 
                          for i in range(torch.cuda.device_count())]
        tmp = '\n - '
        print(f"Available GPUs: {tmp}{tmp.join(map(str, available_gpus))}")
    
    devices = []
    for i in range(max(cuda.device_count(), 1)):
        devices.append(torch.device(f"cuda:{i}") 
                       if cuda.is_available() else "cpu") 
    print(devices)
    
    # Define nice colors for plots
    cmap = get_cmap("Set1")
    colors = cmap.colors
    print(colors)
    legend_names = ["SPOI-AE",
                    "NMF",
                    "Lit. NLS"]
    
    # Load Literature Endmembers
    print("Loading literature spectra......", end="")
    E_lit = pa_util.load_spectra("Hbspectra2.xlsx")
    wls = pa_util.generate_wls(680, 970, 2)
    lit_unmix = lambda x: pa_util.nnls_unmix(E_lit, x)[0]
    lit_remix = lambda x: E_lit @ lit_unmix(x)
    print("Done")
    
    # Load In Vivo Data
    print("Loading data......", end="")
    tr, tr_r, tr_fnames = pa_util.load_data(f"{fpath}/Segmented_600_tr9.mat")
    xv, xv_r, xv_fnames = pa_util.load_data(f"{fpath}/Segmented_600_xv2.mat")
    tr = np.concatenate((tr, xv), axis=1)
    tr_r = np.concatenate((tr_r, xv_r), axis=0)
    tr_fnames = tr_fnames + xv_fnames
    tr, tr_scale = pa_util.norm_data(tr)
    
    ts, ts_r, ts_fnames = pa_util.load_data(f"{fpath}/Segmented_600_ts3.mat")
    ts *= tr_scale
    print("Done")
    
    # Load Model
    model_path = f"{os.getcwd()}/Trained Models/{model_dir}/{model_name}"
    print(f"Current model path: {model_path}")
    print("Loading model.......", end="", flush=True)
    spoiae_model = torch.load(f"{model_path}/model_arch.pt", 
                            pickle_module=dill, 
                            map_location=devices[0])
    spoiae_final_state = torch.load(f"{model_path}/{state_name}.pt", 
                                  pickle_module=dill,
                                  map_location=devices[0])
    spoiae_model.load_state_dict(spoiae_final_state)
    
    E_spoiae = torch.relu(spoiae_model.E).detach().cpu().numpy()
    spoiae_model = spoiae.DataParallelPassthrough(spoiae_model)
    spoiae_model.eval()
    spoiae_unmix = lambda x, r: spoiae.fast_unmix(spoiae_model, x, r, devices[0])
    spoiae_remix = lambda x, r: spoiae.fast_remix(spoiae_model, x, r, devices[0])
    
    spoiae_loss = torch.load(f"{model_path}/loss_obj.pt", pickle_module=dill)
    with open(f"{model_path}/learning_curves.json", "r") as fin:
        spoiae_curves = json.load(fin)
    for key in spoiae_curves:
        spoiae_curves[key] = np.array(spoiae_curves[key])
    print("Done", flush=True)
    
    # Load Image
    spa, us, mask, disp, xaxis, yaxis = \
        pa_util.load_sPA_img(f"{fpath}/{fname}.mat")
    spa, _ = pa_util.norm_data(spa)

    # Plot SPOI-AE Spectra
    fig_sp, axs_sp = pa_viz.Hb_spectra([E_spoiae[:, 0]],  [E_spoiae[:, 1]],
                              wls, 
                              lbls=["HbO2", "HHb"], 
                              c=colors)
    fig_sp.set_size_inches(3.5, 1.3)
    fig_sp.set_dpi(700)
    fig_sp.tight_layout(pad=0.1, w_pad=0.2, h_pad=0.1)
    fig_sp.canvas.draw()
    fig_sp.savefig(f"figs/{model_name}/{fname}/E.pdf",
                   bbox_inches='tight', pad_inches=0.05, dpi=700)
    fig_sp.savefig(f"figs/{model_name}/{fname}/E.png",
                   bbox_inches='tight', pad_inches=0.05, dpi=700, 
                   transparent=True)
    plt.show(block=False)
    
    # Plot Wavelength Dependent Fluence
    fig_grunflu, ax_grunflu = plt.subplots()
    ax_grunflu.plot(wls, spoiae_model.grun_flu.detach().cpu().numpy(), 'k-')
    # ax_grunflu.set_ylabel(r"$\Gamma\phi_0$", fontsize=14)
    ax_grunflu.text(0.95, 0.95, r"$\Gamma\phi_0$", 
                    ha='right', va='top', 
                    transform=ax_grunflu.transAxes, 
                    fontsize=11, fontweight='bold', c='k')
    ax_grunflu.tick_params(axis='both', which='major', labelsize=8)
    ax_grunflu.tick_params(axis='both', which='minor', labelsize=8)
    fig_grunflu.set_size_inches(1.75, 1.3)
    fig_grunflu.set_dpi(700)
    fig_grunflu.tight_layout(pad=0.1, w_pad=0.2, h_pad=0.1)
    fig_grunflu.canvas.draw()
    fig_grunflu.savefig(f"figs/{model_name}/{fname}/grunflu.pdf",
                   bbox_inches='tight', pad_inches=0.05, dpi=700)
    fig_grunflu.savefig(f"figs/{model_name}/{fname}/grunflu.png",
                   bbox_inches='tight', pad_inches=0.05, dpi=700, 
                   transparent=True)
    plt.show(block=True)
    

    # Unmixing 
    C_spoiae = spoiae.fast_img_unmix(spoiae_model, 
                                  spa, disp, 
                                  device=devices[0])
    
    # Normalize Unmixing ,1
    C_spoiae[..., :2] = C_spoiae[..., :2] / np.percentile(C_spoiae[..., :2], 99)
    
    # Calculate Fluence, Psi, etc. 
    mua = spoiae.fast_img_mua(spoiae_model, spa, disp, devices[0])
    mua_hat = spoiae.fast_img_muahat(spoiae_model, spa, disp, devices[0])
    mus = spoiae.fast_img_mus(spoiae_model, spa, disp, devices[0])
    flu = spoiae.fast_img_fluence(spoiae_model, spa, disp, devices[0])
    
    # Remix 
    rmx_spoiae = spoiae.fast_img_remix(spoiae_model, spa, disp, devices[0])
        
    # Plot US
    fig_us, ax_us = plt.subplots()
    fig_us, ax_us = pa_viz.US(fig_us, ax_us, us, xaxis, yaxis, lbl="US")
    fig_us.set_size_inches(1.5, 1)
    fig_us.set_dpi(700)
    fig_us.tight_layout()
    fig_us.canvas.draw()
    os.makedirs(os.path.dirname(f"figs/{model_name}/{fname}/us.pdf"), 
                exist_ok=True)
    fig_us.savefig(f"figs/{model_name}/{fname}/us.pdf", 
                    bbox_inches='tight', pad_inches=0.05, dpi=700)
    fig_us.savefig(f"figs/{model_name}/{fname}/us.png", 
                    bbox_inches='tight', pad_inches=0.05, dpi=700, 
                    transparent=True)
    plt.show(block=False)    
    
    # Plot DISP
    fig_disp, ax_disp = plt.subplots()
    fig_disp, ax_disp = pa_viz.disp(fig_disp, ax_disp, 
                                    us, disp, mask, xaxis, yaxis, 
                                    lbl=r"$\left|r\right|$", 
                                    make_cbar=True, cbar_loc="right")
    fig_disp.set_size_inches(1.5, 1)
    fig_disp.set_dpi(700)
    fig_disp.tight_layout()
    fig_disp.canvas.draw()
    fig_disp.savefig(f"figs/{model_name}/{fname}/disp.pdf", 
                     bbox_inches='tight', pad_inches=0.05, dpi=700)
    fig_disp.savefig(f"figs/{model_name}/{fname}/disp.png", 
                     bbox_inches='tight', pad_inches=0.05, dpi=700, 
                     transparent=True)
    plt.show(block=False)
    
    # Plot Input
    fig_in, axs_in = plt.subplots(5, 1,
                                  gridspec_kw={
                                      'height_ratios': [10, 1, 10, 1, 10]})
    
    
    fig_in, axs_in[0] = pa_viz.img(fig_in, axs_in[0], 
                                   us, spa, mask, xaxis, yaxis, 
                                   wls, plot_wls[0], 
                                   make_cbar=True, cbar_loc="right", 
                                   lbl=r"$\mathbf{P}$")
    fig_in, axs_in[2] = pa_viz.img(fig_in, axs_in[2], 
                                   us, spa, mask, xaxis, yaxis, 
                                   wls, plot_wls[1], 
                                   make_cbar=True, cbar_loc="right", 
                                   lbl=r"$\mathbf{P}$")
    fig_in, axs_in[4] = pa_viz.img(fig_in, axs_in[4], 
                                   us, spa, mask, xaxis, yaxis, 
                                   wls, plot_wls[2], 
                                   make_cbar=True, cbar_loc="right", 
                                   lbl=r"$\mathbf{P}$")
    
    axs_in[1].set_axis_off()
    axs_in[1].text(0.5, 0.5, "⋮", 
                   fontsize=16, fontweight="bold", c="k", 
                   ha='center', va='center')
    
    axs_in[3].set_axis_off()
    axs_in[3].text(0.5, 0.5, "⋮", 
                   fontsize=16, fontweight="bold", c="k", 
                   ha='center', va='center')
    
    fig_in.set_size_inches(1.5, 4)
    fig_in.set_dpi(700)
    fig_in.tight_layout(pad=0.1, w_pad=0.2, h_pad=0.1)
    fig_in.canvas.draw()
    fig_in.savefig(f"figs/{model_name}/{fname}/in.pdf", 
                   bbox_inches='tight', pad_inches=0.05, dpi=700)
    fig_in.savefig(f"figs/{model_name}/{fname}/in.png", 
                   bbox_inches='tight', pad_inches=0.05, dpi=700, 
                   transparent=True)
    plt.show(block=False)
    
    # Plot absorption coefficients
    fig_mua, axs_mua = plt.subplots(1, 3)
    
    fig_mua, axs_mua[0] = pa_viz.mua(fig_mua, axs_mua[0], 
                                     us, mua, mask, xaxis, yaxis, 
                                     wls, plot_wls[0], 
                                     make_cbar=True, cbar_loc="top", 
                                     lbl=r"$\mu_a$")
    fig_mua, axs_mua[1] = pa_viz.mua(fig_mua, axs_mua[1], 
                                     us, mua, mask, xaxis, yaxis, 
                                     wls, plot_wls[1], 
                                     make_cbar=True, cbar_loc="top", 
                                     lbl=r"$\mu_a$")
    fig_mua, axs_mua[2] = pa_viz.mua(fig_mua, axs_mua[2], 
                                     us, mua, mask, xaxis, yaxis, 
                                     wls, plot_wls[2], 
                                     make_cbar=True, cbar_loc="top", 
                                     lbl=r"$\mu_a$")
    
    fig_mua.set_size_inches(3,1.25)
    fig_mua.set_dpi(700)
    fig_mua.tight_layout(pad=0.1, w_pad=0.2, h_pad=0.1)
    fig_mua.canvas.draw()
    fig_mua.savefig(f"figs/{model_name}/{fname}/mua.pdf", 
                    bbox_inches='tight', pad_inches=0.05, dpi=700)
    fig_mua.savefig(f"figs/{model_name}/{fname}/mua.png", 
                    bbox_inches='tight', pad_inches=0.05, dpi=700, 
                    transparent=True)
    plt.show(block=False)
    
    # Plot reduced scattering coefficients
    fig_mus, axs_mus = plt.subplots(1, 3)
    
    fig_mus, axs_mus[0] = pa_viz.mus(fig_mus, axs_mus[0], 
                                     us, mus, mask, xaxis, yaxis, 
                                     wls, plot_wls[0], 
                                     make_cbar=True, cbar_loc="top", 
                                     lbl=r"$\mu_s'$")
    fig_mus, axs_mus[1] = pa_viz.mus(fig_mus, axs_mus[1], 
                                     us, mus, mask, xaxis, yaxis, 
                                     wls, plot_wls[1], 
                                     make_cbar=True, cbar_loc="top", 
                                     lbl=r"$\mu_s'$")
    fig_mus, axs_mus[2] = pa_viz.mus(fig_mus, axs_mus[2], 
                                     us, mus, mask, xaxis, yaxis, 
                                     wls, plot_wls[2], 
                                     make_cbar=True, cbar_loc="top", 
                                     lbl=r"$\mu_s'$")
    
    fig_mus.set_size_inches(3,1.25)
    fig_mus.set_dpi(700)
    fig_mus.tight_layout(pad=0.1, w_pad=0.2, h_pad=0.1)
    fig_mus.canvas.draw()
    fig_mus.savefig(f"figs/{model_name}/{fname}/mus.pdf", 
                    bbox_inches='tight', pad_inches=0.05, dpi=700)
    fig_mus.savefig(f"figs/{model_name}/{fname}/mus.png", 
                    bbox_inches='tight', pad_inches=0.05, dpi=700, 
                    transparent=True)
    plt.show(block=False)
    
    # Plot unmixing results
    fig_unm, axs_unm = plt.subplots(1, 3)
    
    fig_unm, axs_unm[0] = pa_viz.HbO2_con(fig_unm, axs_unm[0],
                                          us, C_spoiae, mask, xaxis, yaxis,
                                          fgmin=0, fgmax=1,
                                          make_cbar=True, cbar_loc="top", 
                                          lbl="HbO2")
    fig_unm, axs_unm[1] = pa_viz.Hb_con(fig_unm, axs_unm[1],
                                        us, C_spoiae, mask, xaxis, yaxis,
                                        fgmin=0, fgmax=1,
                                        make_cbar=True, cbar_loc="top", 
                                        lbl="Hb")
    fig_unm, axs_unm[2] = pa_viz.SO2(fig_unm, axs_unm[2],
                                     us, C_spoiae, mask, xaxis, yaxis,
                                     make_cbar=True, cbar_loc="top", 
                                     lbl="SO2")
    
    fig_unm.set_size_inches(3.75, 1.25)
    fig_unm.set_dpi(700)
    fig_unm.tight_layout(pad=0.1, w_pad=0.2, h_pad=0.1)
    fig_unm.canvas.draw()
    fig_unm.savefig(f"figs/{model_name}/{fname}/unmixing.pdf", 
                    bbox_inches='tight', pad_inches=0.05, dpi=700)
    fig_unm.savefig(f"figs/{model_name}/{fname}/unmixing.png", 
                    bbox_inches='tight', pad_inches=0.05, dpi=700, 
                    transparent=True)
    plt.show(block=False)
    
    # Plot estimated absorption coefficients
    fig_muahat, axs_muahat = plt.subplots(1, 3)
    
    fig_muahat, axs_muahat[0] = pa_viz.mua(fig_muahat, axs_muahat[0],
                                           us, mua_hat, mask, xaxis, yaxis,
                                           wls, plot_wls[0],
                                           make_cbar=True, cbar_loc="top",
                                           lbl=r"$\widehat{\mu}_a$")
    fig_muahat, axs_muahat[1] = pa_viz.mua(fig_muahat, axs_muahat[1],
                                           us, mua_hat, mask, xaxis, yaxis,
                                           wls, plot_wls[1],
                                           make_cbar=True, cbar_loc="top",
                                           lbl=r"$\widehat{\mu}_a$")
    fig_muahat, axs_muahat[2] = pa_viz.mua(fig_muahat, axs_muahat[2],
                                           us, mua_hat, mask, xaxis, yaxis,
                                           wls, plot_wls[2],
                                           make_cbar=True, cbar_loc="top",
                                           lbl=r"$\widehat{\mu}_a$")
    
    fig_muahat.set_size_inches(3,1.25)
    fig_muahat.set_dpi(700)
    fig_muahat.tight_layout(pad=0.1, w_pad=0.2, h_pad=0.1)
    fig_muahat.canvas.draw()
    fig_muahat.savefig(f"figs/{model_name}/{fname}/muahat.pdf",
                       bbox_inches='tight', pad_inches=0.05, dpi=700)
    fig_muahat.savefig(f"figs/{model_name}/{fname}/muahat.png",
                       bbox_inches='tight', pad_inches=0.05, dpi=700,
                       transparent=True)
    plt.show(block=False)
    
    # Plot Fluence
    fig_flu, axs_flu = plt.subplots(1, 3)
    
    fig_flu, axs_flu[0] = pa_viz.fluence(fig_flu, axs_flu[0],
                                         us, flu, mask, xaxis, yaxis,
                                         wls, plot_wls[0],
                                         make_cbar=True, cbar_loc="top", 
                                         lbl=r"$\Gamma\Phi$")
    fig_flu, axs_flu[1] = pa_viz.fluence(fig_flu, axs_flu[1],
                                         us, flu, mask, xaxis, yaxis,
                                         wls, plot_wls[1],
                                         make_cbar=True, cbar_loc="top", 
                                         lbl=r"$\Gamma\Phi$")
    fig_flu, axs_flu[2] = pa_viz.fluence(fig_flu, axs_flu[2],
                                         us, flu, mask, xaxis, yaxis,
                                         wls, plot_wls[2],
                                         make_cbar=True, cbar_loc="top", 
                                         lbl=r"$\Gamma\Phi$")

    fig_flu.set_size_inches(3, 1.25)
    fig_flu.set_dpi(700)
    fig_flu.tight_layout(pad=0.1, w_pad=0.2, h_pad=0.1)
    fig_flu.canvas.draw()
    fig_flu.savefig(f"figs/{model_name}/{fname}/fluence.pdf",
                    bbox_inches='tight', pad_inches=0.05, dpi=700)
    fig_flu.savefig(f"figs/{model_name}/{fname}/fluence.png",
                    bbox_inches='tight', pad_inches=0.05, dpi=700,
                    transparent=True)
    plt.show(block=False)
    
    # Plot Output
    fig_out, axs_out = plt.subplots(3,1)
    
    fig_out, axs_out[0] = pa_viz.img(fig_out, axs_out[0],
                                     us, rmx_spoiae, mask, xaxis, yaxis,
                                     wls, plot_wls[0],
                                     make_cbar=True, cbar_loc="right",
                                     lbl=r"$\widehat{\mathbf{P}}$")
    fig_out, axs_out[1] = pa_viz.img(fig_out, axs_out[1],
                                     us, rmx_spoiae, mask, xaxis, yaxis,
                                     wls, plot_wls[1],
                                     make_cbar=True, cbar_loc="right",
                                     lbl=r"$\widehat{\mathbf{P}}$")
    fig_out, axs_out[2] = pa_viz.img(fig_out, axs_out[2],
                                     us, rmx_spoiae, mask, xaxis, yaxis,
                                     wls, plot_wls[2],
                                     make_cbar=True, cbar_loc="right",
                                     lbl=r"$\widehat{\mathbf{P}}$")
    
    fig_out.set_size_inches(1.5, 3)
    fig_out.set_dpi(700)
    fig_out.tight_layout(pad=0.1, w_pad=0.2, h_pad=0.1)
    fig_out.canvas.draw()
    fig_out.savefig(f"figs/{model_name}/{fname}/out.pdf",
                    bbox_inches='tight', pad_inches=0.05, dpi=700)
    fig_out.savefig(f"figs/{model_name}/{fname}/out.png",
                    bbox_inches='tight', pad_inches=0.05, dpi=700,
                    transparent=True)
    plt.show(block=True) 
    

       

if __name__ == '__main__':
    main()
