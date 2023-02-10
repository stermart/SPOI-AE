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
from mpl_toolkits.axes_grid1 import make_axes_locatable, axes_size

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
    model2_name = "Model_5e0_NoAdapt"
    state_name = "final_state"

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
    wls = pa_util.generate_wls(680, 990, 2)
    def lit_unmix(x): return pa_util.nnls_unmix(E_lit, x)[0]
    def lit_remix(x): return E_lit @ lit_unmix(x)
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
    # ts, ts_r, ts_fnames = pa_util.load_data("data/Segmented_600_xv2.mat")
    ts *= tr_scale
    print("Done")

    # Calculate Seeded NMF
    snmf = pa_nmf.obj(k_NMF=2, seeded=True, max_iters=400, verbose=False)
    print("Training SNMF", flush=True)
    snmf = pa_nmf.train(snmf, tr, E0=E_lit, C0=lit_unmix(tr))
    print("Done with SNMF", flush=True)
    E_snmf = pa_nmf.get_spectra(snmf)
    def snmf_unmix(x): return pa_nmf.unmix(snmf, x)
    def snmf_remix(x): return E_snmf@snmf_unmix(x)

    # Load Model 1
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

    # Load Model 2
    model2_path = f"{os.getcwd()}/Trained Models/{model_dir}/{model2_name}"
    print(f"Current model path: {model2_path}")
    print("Loading model.......", end="", flush=True)
    spoiae2_model = torch.load(f"{model2_path}/model_arch.pt",
                            pickle_module=dill,
                            map_location=devices[0])
    spoiae2_final_state = torch.load(f"{model2_path}/{state_name}.pt",
                                  pickle_module=dill,
                                  map_location=devices[0])
    spoiae2_model.load_state_dict(spoiae2_final_state)

    E_spoiae2 = torch.relu(spoiae2_model.E).detach().cpu().numpy()
    spoiae2_model = spoiae.DataParallelPassthrough(spoiae2_model)
    spoiae2_model.eval()

    # Load Image
    spa, us, mask, disp, xaxis, yaxis = \
        pa_util.load_sPA_img(f"{fpath}/{fname}.mat")
    spa, _ = pa_util.norm_data(spa)

    # Unmixing
    C_spoiae = spoiae.fast_img_unmix(spoiae_model,
                                  spa, disp,
                                  device=devices[0])
    C_spoiae2 = spoiae.fast_img_unmix(spoiae2_model,
                                      spa, disp, 
                                      device=devices[0])
    C_snmf = pa_util.unmix_img(spa, snmf_unmix)
    C_lit = pa_util.unmix_img(spa, lit_unmix)

    # Normalize Unmixing
    C_spoiae[..., :2] = C_spoiae[..., :2] / np.percentile(C_spoiae[..., :2], 99)
    C_spoiae2[..., :2] = C_spoiae2[..., :2] / np.percentile(C_spoiae2[..., :2], 99)
    C_snmf[..., :2] = C_snmf[..., :2] / np.percentile(C_snmf[..., :2], 99)
    C_lit[..., :2] = C_lit[..., :2] / np.percentile(C_lit[..., :2], 99)

    # Calculate Optical Parameters

    mua_hat = spoiae.fast_img_muahat(spoiae_model, spa, disp, devices[0])
    mus = spoiae.fast_img_mus(spoiae_model, spa, disp, devices[0])

    # Plot US
    fig_us, ax_us = plt.subplots()
    fig_us, ax_us = pa_viz.US(fig_us, ax_us, us, xaxis, yaxis, lbl="US")
    fig_us.set_size_inches(1.5, 1)
    fig_us.set_dpi(700)
    fig_us.tight_layout()
    fig_us.canvas.draw()

    os.makedirs(os.path.dirname(
        f"figs/{model_name}/{fname}/us.pdf"), exist_ok=True)
    fig_us.savefig(f"figs/{model_name}/{fname}/us.pdf",
                   bbox_inches='tight', pad_inches=0.05, dpi=700)
    fig_us.savefig(f"figs/{model_name}/{fname}/us.png",
                   bbox_inches='tight', pad_inches=0.05, dpi=700)
    plt.show(block=False)

    # Plot SPAU-Net Unmixing
    fig_unm, axs_unm = plt.subplots(4, 3)

    fig_unm, axs_unm[0][0] = pa_viz.HbO2_con(fig_unm, axs_unm[0][0],
                                             us, C_spoiae, mask, xaxis, yaxis,
                                             fgmin=0, fgmax=1,
                                             make_cbar=False, lbl="A")
    fig_unm, axs_unm[0][1] = pa_viz.Hb_con(fig_unm, axs_unm[0][1],
                                           us, C_spoiae, mask, xaxis, yaxis,
                                           fgmin=0, fgmax=1,
                                           make_cbar=False, lbl="A")
    fig_unm, axs_unm[0][2] = pa_viz.SO2(fig_unm, axs_unm[0][2],
                                        us, C_spoiae, mask, xaxis, yaxis,
                                        make_cbar=False, lbl="A")

    fig_unm, axs_unm[1][0] = pa_viz.HbO2_con(fig_unm, axs_unm[1][0],
                                             us, C_spoiae2, mask, xaxis, yaxis,
                                             fgmin=0, fgmax=1,
                                             make_cbar=False, lbl="B")
    fig_unm, axs_unm[1][1] = pa_viz.Hb_con(fig_unm, axs_unm[1][1],
                                           us, C_spoiae2, mask, xaxis, yaxis,
                                           fgmin=0, fgmax=1,
                                           make_cbar=False, lbl="B")
    fig_unm, axs_unm[1][2] = pa_viz.SO2(fig_unm, axs_unm[1][2],
                                        us, C_spoiae2, mask, xaxis, yaxis,
                                        make_cbar=False, lbl="B")

    fig_unm, axs_unm[2][0] = pa_viz.HbO2_con(fig_unm, axs_unm[2][0],
                                             us, C_snmf, mask, xaxis, yaxis,
                                             fgmin=0, fgmax=1,
                                             make_cbar=False, lbl="C")
    fig_unm, axs_unm[2][1] = pa_viz.Hb_con(fig_unm, axs_unm[2][1],
                                           us, C_snmf, mask, xaxis, yaxis,
                                           fgmin=0, fgmax=1,
                                           make_cbar=False, lbl="C")
    fig_unm, axs_unm[2][2] = pa_viz.SO2(fig_unm, axs_unm[2][2],
                                        us, C_snmf, mask, xaxis, yaxis,
                                        make_cbar=False, lbl="C")

    fig_unm, axs_unm[3][0] = pa_viz.HbO2_con(fig_unm, axs_unm[3][0],
                                             us, C_lit, mask, xaxis, yaxis,
                                             fgmin=0, fgmax=1,
                                             make_cbar=False, lbl="D")
    fig_unm, axs_unm[3][1] = pa_viz.Hb_con(fig_unm, axs_unm[3][1],
                                           us, C_lit, mask, xaxis, yaxis,
                                           fgmin=0, fgmax=1,
                                           make_cbar=False, lbl="D")
    fig_unm, axs_unm[3][2] = pa_viz.SO2(fig_unm, axs_unm[3][2],
                                        us, C_lit, mask, xaxis, yaxis,
                                        make_cbar=False, lbl="D")

    fig_unm.set_size_inches(3.5, 5.1333)
    fig_unm.set_dpi(700)
    fig_unm.tight_layout(pad=0.1, w_pad=0.2, h_pad=1.0)

    cax_unm = [None, None, None]

    fig_unm, cax_unm[0] = pa_viz.horizontal_cbar(fig_unm, ax=axs_unm[:, 0],
                                                 fgmin=0, fgmax=1,
                                                 lbl="[HbO2] (a.u.)",
                                                 cmap='Reds')
    fig_unm, cax_unm[1] = pa_viz.horizontal_cbar(fig_unm, ax=axs_unm[:, 1],
                                                 fgmin=0, fgmax=1,
                                                 lbl="[HHb] (a.u.)",
                                                 cmap='Blues')
    fig_unm, cax_unm[2] = pa_viz.horizontal_cbar(fig_unm, ax=axs_unm[:, 2],
                                                 fgmin=0, fgmax=100,
                                                 lbl="SO2 (%)",
                                                 cmap='RdBu_r')

    fig_unm.canvas.draw()

    fig_unm.savefig(f"figs/{model_name}/{fname}/unmixing_comparison.pdf",
                    bbox_inches='tight', pad_inches=0.05, dpi=700)
    fig_unm.savefig(f"figs/{model_name}/{fname}/unmixing_comparison.png",
                    bbox_inches='tight', pad_inches=0.05, dpi=700)
    plt.show(block=False)

    # Plot SPAU-Net Optical Parameters
    plot_wls = [680, 760, 970]

    fig_opt, axs_opt = plt.subplots(3, 3)

    fig_opt, axs_opt[0][0] = pa_viz.img(fig_opt, axs_opt[0][0],
                                        us, spa, mask, xaxis, yaxis,
                                        wls, plot_wls[0],
                                        fgmax=np.percentile(spa, 99),
                                        make_cbar=False, lbl=r"$\mathbf{P}$")
    fig_opt, axs_opt[1][0] = pa_viz.img(fig_opt, axs_opt[1][0],
                                        us, spa, mask, xaxis, yaxis,
                                        wls, plot_wls[1],
                                        fgmax=np.percentile(spa, 99),
                                        make_cbar=False, lbl=r"$\mathbf{P}$")
    fig_opt, axs_opt[2][0] = pa_viz.img(fig_opt, axs_opt[2][0],
                                        us, spa, mask, xaxis, yaxis,
                                        wls, plot_wls[2],
                                        fgmax=np.percentile(spa, 99),
                                        make_cbar=False, lbl=r"$\mathbf{P}$")

    fig_opt, axs_opt[0][1] = pa_viz.mua(fig_opt, axs_opt[0][1],
                                        us, mua_hat, mask, xaxis, yaxis,
                                        wls, plot_wls[0],
                                        fgmax=np.percentile(mua_hat, 99),
                                        make_cbar=False,
                                        lbl=r"$\widehat{\mu}_a$")
    fig_opt, axs_opt[1][1] = pa_viz.mua(fig_opt, axs_opt[1][1],
                                        us, mua_hat, mask, xaxis, yaxis,
                                        wls, plot_wls[1],
                                        fgmax=np.percentile(mua_hat, 99),
                                        make_cbar=False,
                                        lbl=r"$\widehat{\mu}_a$")
    fig_opt, axs_opt[2][1] = pa_viz.mua(fig_opt, axs_opt[2][1],
                                        us, mua_hat, mask, xaxis, yaxis,
                                        wls, plot_wls[2],
                                        fgmax=np.percentile(mua_hat, 99),
                                        make_cbar=False,
                                        lbl=r"$\widehat{\mu}_a$")

    fig_opt, axs_opt[0][2] = pa_viz.mus(fig_opt, axs_opt[0][2],
                                        us, mus, mask, xaxis, yaxis,
                                        wls, plot_wls[0],
                                        fgmax=np.percentile(mus, 99),
                                        make_cbar=False,
                                        lbl=r"$\mu_s'$")
    fig_opt, axs_opt[1][2] = pa_viz.mus(fig_opt, axs_opt[1][2],
                                        us, mus, mask, xaxis, yaxis,
                                        wls, plot_wls[1],
                                        fgmax=np.percentile(mus, 99),
                                        make_cbar=False,
                                        lbl=r"$\mu_s'$")
    fig_opt, axs_opt[2][2] = pa_viz.mus(fig_opt, axs_opt[2][2],
                                        us, mus, mask, xaxis, yaxis,
                                        wls, plot_wls[2],
                                        fgmax=np.percentile(mus, 99),
                                        make_cbar=False,
                                        lbl=r"$\mu_s'$")

    fig_opt.set_size_inches(3.5, 3.85)
    fig_opt.set_dpi(700)
    fig_opt.tight_layout(pad=0.1, w_pad=0.2, h_pad=1.0)

    cax_opt = [None, None, None]

    fig_opt, cax_opt[0] = pa_viz.horizontal_cbar(fig_opt, ax=axs_opt[:, 0],
                                                 fgmin=0,
                                                 fgmax=np.percentile(spa, 99),
                                                 lbl="sPA Input (a.u.)",
                                                 cmap='plasma')
    fig_opt, cax_opt[1] = pa_viz.horizontal_cbar(fig_opt, ax=axs_opt[:, 1],
                                                 fgmin=0,
                                                 fgmax=np.percentile(
                                                     mua_hat, 99),
                                                 lbl="Absorption (1/m)",
                                                 cmap='cividis')
    fig_opt, cax_opt[2] = pa_viz.horizontal_cbar(fig_opt, ax=axs_opt[:, 2],
                                                 fgmin=0,
                                                 fgmax=np.percentile(mus, 99),
                                                 lbl="Scattering (1/m)",
                                                 cmap='cividis')

    fig_opt.canvas.draw()
    fig_opt.savefig(f"figs/{model_name}/{fname}/optical_parameters.pdf",
                    bbox_inches='tight', pad_inches=0.05, dpi=700)
    fig_opt.savefig(f"figs/{model_name}/{fname}/optical_parameters.png",
                    bbox_inches='tight', pad_inches=0.05, dpi=700)
    plt.show(block=True)


if __name__ == '__main__':
    main()
