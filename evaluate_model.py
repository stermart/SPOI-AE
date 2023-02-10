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

# my shit
import pa_nmf
import pa_util
import spoiae
import pa_viz

def main():
    
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
    
    # define model params
    model_dir = "Paper Models"
    # model_dir = "User Created"
    model_name = "Model_5e0_Adapt"
    state_name = "final_state"
    
    # Load Literature Endmembers
    print("Loading literature spectra......", end="")
    E_lit = pa_util.load_spectra("Hbspectra2.xlsx")
    wls = pa_util.generate_wls(680, 970, 2)
    lit_unmix = lambda x: pa_util.nnls_unmix(E_lit, x)[0]
    lit_remix = lambda x: E_lit @ lit_unmix(x)
    print("Done")
    
    # Load In Vivo Data
    print("Loading data......", end="")
    tr, tr_r, tr_fnames = pa_util.load_data("Data/Segmented_600_tr9.mat")
    xv, xv_r, xv_fnames = pa_util.load_data("Data/Segmented_600_xv2.mat")
    tr = np.concatenate((tr, xv), axis=1)
    tr_r = np.concatenate((tr_r, xv_r), axis=0)
    tr_fnames = tr_fnames + xv_fnames
    tr, tr_scale = pa_util.norm_data(tr)
    
    ts, ts_r, ts_fnames = pa_util.load_data("Data/Segmented_600_ts3.mat")
    ts *= tr_scale
    print("Done")
        
    fig_in, axs_in = plt.subplots(1, 2, sharex=True, sharey=True)
    
    fig_in, axs_in[0] = pa_viz.data_distr(fig_in, axs_in[0], 
                                          np.ravel(tr), 
                                          lbl="TR", c='k', 
                                          show_xlabel=True, show_ylabel=True)
    fig_in, axs_in[1] = pa_viz.data_distr(fig_in, axs_in[1], 
                                          np.ravel(ts), 
                                          lbl="TS", c='k', 
                                          show_xlabel=True, show_ylabel=False)

    fig_in.set_size_inches(3.5, 1.5)
    fig_in.set_dpi(700)
    fig_in.tight_layout(pad=0.1, w_pad=0.1, h_pad=0.1)
    os.makedirs(os.path.dirname(f"figs/{model_name}/ts_input.pdf"), exist_ok=True)
    fig_in.savefig(f"figs/{model_name}/ts_input.pdf",
                   bbox_inches='tight', pad_inches=0.05, dpi=700)
    fig_in.savefig(f"figs/{model_name}/ts_input.png",
                   bbox_inches='tight', pad_inches=0.05, dpi=700)
    fig_in.canvas.draw()
    plt.show(block=False) 

    # Load model, supporting variables
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
    print(spoiae_model.learnE, spoiae_loss.beta)
    
    # Calculate NMF and Seeded NMF
    snmf = pa_nmf.obj(k_NMF=2, seeded=True, max_iters=400, verbose=False)
    print("Training SNMF", flush=True)
    snmf = pa_nmf.train(snmf, tr, E0=E_lit, C0=lit_unmix(tr))
    print("Done with SNMF", flush=True)
    E_snmf = pa_nmf.get_spectra(snmf)
    snmf_unmix = lambda x: pa_nmf.unmix(snmf, x)
    snmf_remix = lambda x: E_snmf@snmf_unmix(x)
    
    # Plot Spectra
    fig_sp, axs_sp = pa_viz.Hb_spectra([E_spoiae[:, 0], E_snmf[:, 0], E_lit[:, 0]],
                              [E_spoiae[:, 1], E_snmf[:, 1], E_lit[:, 1]],
                              wls,
                              lbls=["HbO2", "HHb"],
                              names=legend_names,
                              c=colors, 
                              show_xlabel=True, show_ylabel=True)
    fig_sp.set_size_inches(3.5, 1.6)
    fig_sp.set_dpi(700)
    fig_sp.tight_layout(pad=0.1, w_pad=0.1, h_pad=0.1)
    fig_sp.canvas.draw()
    fig_sp.savefig(f"figs/{model_name}/spectra.pdf",
                   bbox_inches='tight', pad_inches=0.05, dpi=700)
    fig_sp.savefig(f"figs/{model_name}/spectra.png",
                   bbox_inches='tight', pad_inches=0.05, dpi=700)
    plt.show(block=False)
        
    # Extract Learning Rate History
    lr_hist = spoiae_curves["LR Hist"]
    del spoiae_curves["LR Hist"]   
    
    # Plot training curves with reference performance
    fig_tr, axs_tr = plt.subplots(1, 3, sharex=True, sharey=True)

    tr_snmf_loss = spoiae_loss(torch.from_numpy(tr).T, 
                             torch.from_numpy(snmf_remix(tr)).T)
    tr_snmf_mse = spoiae_loss._mse.item()
    tr_snmf_msad = spoiae_loss._msad.item()
    tr_snmf_loss = tr_snmf_loss.item()
    
    tr_lit_loss = spoiae_loss(torch.from_numpy(tr).T, 
                            torch.from_numpy(lit_remix(tr)).T)
    tr_lit_mse = spoiae_loss._mse.item()
    tr_lit_msad = spoiae_loss._msad.item()
    tr_lit_loss = tr_lit_loss.item()
    
    fig_tr, axs_tr[0] = pa_viz.learning_curve(fig_tr, axs_tr[0], 
                                           [spoiae_curves["Training"][:,0]], 
                                           [tr_snmf_mse, tr_lit_mse], 
                                           lbl="A", 
                                           names=legend_names, 
                                           c=colors, 
                                           show_xlabel=True, ylabel="Cost")
    fig_tr, axs_tr[1] = pa_viz.learning_curve(fig_tr, axs_tr[1], 
                                              [spoiae_curves["Training"][:,1]], 
                                              [tr_snmf_msad, tr_lit_msad], 
                                              lbl="B", 
                                              c=colors, 
                                              show_xlabel=True)
    fig_tr, axs_tr[2] = pa_viz.learning_curve(fig_tr, axs_tr[2], 
                                              [spoiae_curves["Training"][:,2]], 
                                              [tr_snmf_loss, tr_lit_loss],
                                              lbl="C",
                                              c=colors, 
                                              show_xlabel=True)
    
    print("---- TR ----")
    print(f'MSE: {spoiae_curves["Training"][-1,0]}, '
          f'{tr_snmf_mse}, {tr_lit_mse}')
    print(f'MSAD: {spoiae_curves["Training"][-1,1]}, '
          f'{tr_snmf_msad}, {tr_lit_msad}')
    print(f'Loss: {spoiae_curves["Training"][-1,2]}, '
          f'{tr_snmf_loss}, {tr_lit_loss}', 
          flush=True)
    
    with open(f"figs/{model_name}/ts_res.txt", "w") as fout:
        print("---- TR ----",
              file=fout)
        print(f'MSE: {spoiae_curves["Training"][-1,0]}, '
              f'{tr_snmf_mse}, {tr_lit_mse}',
              file=fout)
        print(f'MSAD: {spoiae_curves["Training"][-1,1]}, '
              f'{tr_snmf_msad}, {tr_lit_msad}',
              file=fout)
        print(f'Loss: {spoiae_curves["Training"][-1,2]}, '
              f'{tr_snmf_loss}, {tr_lit_loss}',
              file=fout,
              flush=True)
        
    fig_tr.set_size_inches(3.5, 1.5)
    fig_tr.set_dpi(700)
    fig_tr.tight_layout(pad=0.1, w_pad=0.1, h_pad=0.1)
    fig_tr.canvas.draw()
    fig_tr.savefig(f"figs/{model_name}/ts_tr_curves.pdf",
                bbox_inches='tight', pad_inches=0.05, dpi=700)
    fig_tr.savefig(f"figs/{model_name}/ts_tr_curves.png",
                bbox_inches='tight', pad_inches=0.05, dpi=700)
    plt.show(block=False)

    # Plot testing curves with reference performance
    fig_ts, axs_ts = plt.subplots(1, 3, sharex=True, sharey=True)
    
    ts_snmf_loss = spoiae_loss(torch.from_numpy(ts).T, 
                             torch.from_numpy(snmf_remix(ts)).T)
    ts_snmf_mse = spoiae_loss._mse.item()
    ts_snmf_msad = spoiae_loss._msad.item()
    ts_snmf_loss = ts_snmf_loss.item()
    
    ts_lit_loss = spoiae_loss(torch.from_numpy(ts).T, 
                            torch.from_numpy(lit_remix(ts)).T)
    ts_lit_mse = spoiae_loss._mse.item()
    ts_lit_msad = spoiae_loss._msad.item()
    ts_lit_loss = ts_lit_loss.item()
    
    fig_ts, axs_ts[0] = pa_viz.learning_curve(fig_ts, axs_ts[0],
                                              [spoiae_curves["Testing"][:, 0]],
                                              [ts_snmf_mse, ts_lit_mse],
                                              lbl="A",
                                              names=legend_names,
                                              c=colors, 
                                              show_xlabel=True, ylabel="Cost")
    fig_ts, axs_ts[1] = pa_viz.learning_curve(fig_ts, axs_ts[1], 
                                              [spoiae_curves["Testing"][:, 1]], 
                                              [ts_snmf_msad, ts_lit_msad], 
                                              lbl="B",
                                              c=colors, 
                                              show_xlabel=True)
    fig_ts, axs_ts[2] = pa_viz.learning_curve(fig_ts, axs_ts[2], 
                                              [spoiae_curves["Testing"][:, 2]], 
                                              [ts_snmf_loss, ts_lit_loss],
                                              lbl="C",
                                              c=colors, 
                                              show_xlabel=True)
    
    print("---- TS ----")
    print(f'MSE: {spoiae_curves["Testing"][-1,0]}, '
          f'{ts_snmf_mse}, {ts_lit_mse}')
    print(f'MSAD: {spoiae_curves["Testing"][-1,1]}, '
          f'{ts_snmf_msad}, {ts_lit_msad}')
    print(f'Loss: {spoiae_curves["Testing"][-1,2]}, '
          f'{ts_snmf_loss}, {ts_lit_loss}', 
          flush=True)  
    
    with open(f"figs/{model_name}/ts_res.txt", "a") as fout:
        print("---- TS ----", 
              file=fout)
        print(f'MSE: {spoiae_curves["Testing"][-1,0]}, '
              f'{ts_snmf_mse}, {ts_lit_mse}',
              file=fout)
        print(f'MSAD: {spoiae_curves["Testing"][-1,1]}, '
              f'{ts_snmf_msad}, {ts_lit_msad}', 
              file=fout)
        print(f'Loss: {spoiae_curves["Testing"][-1,2]}, '
              f'{ts_snmf_loss}, {ts_lit_loss}',
              file=fout,
              flush=True)    
      
    fig_ts.set_size_inches(3.5, 1.6)
    fig_ts.set_dpi(700)
    fig_ts.tight_layout(pad=0.1, w_pad=0.1, h_pad=0.1)
    fig_ts.canvas.draw()
    fig_ts.savefig(f"figs/{model_name}/ts_ts_curves.pdf",
                   bbox_inches='tight', pad_inches=0.05, dpi=700)
    fig_ts.savefig(f"figs/{model_name}/ts_ts_curves.png",
                   bbox_inches='tight', pad_inches=0.05, dpi=700)
    plt.show(block=False)
    
    # TR vs TS LC Log
    
    fig_vs, ax_vs = plt.subplots()
    
    fig_vs, ax_vs = pa_viz.log_learning_curve(fig_vs, ax_vs, 
                                              spoiae_curves["Training"][:,2],
                                              spoiae_curves["Testing"][:,2], 
                                              xv_mode=False, 
                                              show_xlabel=True, show_ylabel=True)    
    fig_vs.set_size_inches(3.5, 1.5)
    fig_vs.set_dpi(700)
    fig_vs.tight_layout()
    fig_vs.canvas.draw()
    fig_vs.savefig(f"figs/{model_name}/ts_vs_curves.pdf", 
                   bbox_inches='tight', pad_inches=0.05, dpi=700)
    fig_vs.savefig(f"figs/{model_name}/ts_vs_curves.png", 
                   bbox_inches='tight', pad_inches=0.05, dpi=700)
    plt.show(block=False)
        
    # R^2 analysis
    with torch.no_grad():
        tr_spoiae = spoiae_remix(tr.T, tr_r)
        print(tr_spoiae.shape, flush=True)
        ts_spoiae = spoiae_remix(ts.T, ts_r)
        print(ts_spoiae.shape, flush=True)
        
    # Training Data R^2
    tr_r2_lit = pa_util.R2(tr, lit_remix(tr))
    tr_r2_snmf = pa_util.R2(tr, snmf_remix(tr))
    tr_r2_spoiae = pa_util.R2(tr, tr_spoiae.T)
    
    ts_r2_lit = pa_util.R2(ts, lit_remix(ts))
    ts_r2_snmf = pa_util.R2(ts, snmf_remix(ts))
    ts_r2_spoiae = pa_util.R2(ts, ts_spoiae.T)
    
    fig_r2, axs_r2 = plt.subplots(1, 2, sharex=True, sharey=True)
    
    fig_r2, axs_r2[0] = pa_viz.R2(fig_r2, axs_r2[0], 
                                  [tr_r2_spoiae, tr_r2_snmf, tr_r2_lit], 
                                  wls, 
                                  lbl="TR", 
                                  names=legend_names, 
                                  c=colors, 
                                  show_xlabel=True, show_ylabel=True)
    
    print(", ".join(f"{np.mean(r2)=}" 
                    for r2 in [tr_r2_spoiae, tr_r2_snmf, tr_r2_lit]), 
          flush=True)
    print(", ".join(f"{np.std(r2)=}" 
                    for r2 in [tr_r2_spoiae, tr_r2_snmf, tr_r2_lit]), 
          flush=True)
    
    fig_r2, axs_r2[1] = pa_viz.R2(fig_r2, axs_r2[1], 
                                  [ts_r2_spoiae, ts_r2_snmf, ts_r2_lit], 
                                  wls, 
                                  lbl="TS", 
                                  c=colors, 
                                  show_xlabel=True, show_ylabel=False)
    
    print(", ".join(f"{np.mean(r2)=}" 
                    for r2 in [ts_r2_spoiae, ts_r2_snmf, ts_r2_lit]), 
          flush=True)
    print(", ".join(f"{np.std(r2)=}" 
                    for r2 in [ts_r2_spoiae, ts_r2_snmf, ts_r2_lit]), 
          flush=True)
    
    with open(f"figs/{model_name}/r2.txt", "w") as fout:
        print(", ".join(f"{np.mean(r2)=}"
                        for r2 in [tr_r2_spoiae, tr_r2_snmf, tr_r2_lit]),
              file=fout)
        print(", ".join(f"{np.std(r2)=}"
                        for r2 in [tr_r2_spoiae, tr_r2_snmf, tr_r2_lit]),
              file=fout)
        print(", ".join(f"{np.mean(r2)=}"
                        for r2 in [ts_r2_spoiae, ts_r2_snmf, ts_r2_lit]),
              file=fout)
        print(", ".join(f"{np.std(r2)=}"
                        for r2 in [ts_r2_spoiae, ts_r2_snmf, ts_r2_lit]),
              file=fout)    
    
    fig_r2.set_size_inches(3.5, 2)
    fig_r2.set_dpi(700)
    fig_r2.tight_layout(pad=0.1, w_pad=0.1, h_pad=0.1)
    fig_r2.canvas.draw()
    fig_r2.savefig(f"figs/{model_name}/ts_r2.pdf", 
                   bbox_inches='tight', pad_inches=0.05, dpi=700)    
    fig_r2.savefig(f"figs/{model_name}/ts_r2.png", 
                   bbox_inches='tight', pad_inches=0.05, dpi=700) 
    plt.show(block=False)
    
    # Plot Average Spectrum
    fig_avgsp, axs_avgsp = plt.subplots(1, 2, sharex=True, sharey=True)
    
    fig_avgsp, axs_avgsp[0] = pa_viz.avg_spectrum(fig_avgsp, axs_avgsp[0], 
                                                  tr, 
                                                  [tr_spoiae.T, 
                                                   snmf_remix(tr), 
                                                   lit_remix(tr)], 
                                                  wls, 
                                                  lbl="TR", 
                                                  names=legend_names, 
                                                  c=colors, 
                                                  show_xlabel=True, 
                                                  show_ylabel=True)
    fig_avgsp, axs_avgsp[1] = pa_viz.avg_spectrum(fig_avgsp, axs_avgsp[1], 
                                                  ts, 
                                                  [ts_spoiae.T, 
                                                   snmf_remix(ts), 
                                                   lit_remix(ts)], 
                                                  wls, 
                                                  lbl="TS", 
                                                  c=colors, 
                                                  show_xlabel=True, 
                                                  show_ylabel=False)

    fig_avgsp.set_size_inches(3.5, 1.75)
    fig_avgsp.set_dpi(700)
    fig_avgsp.tight_layout(pad=0.1, w_pad=0.1, h_pad=0.1)    
    fig_avgsp.canvas.draw()
    fig_avgsp.savefig(f"figs/{model_name}/ts_avgsp.pdf", 
                      bbox_inches='tight', pad_inches=0.05, dpi=700)    
    fig_avgsp.savefig(f"figs/{model_name}/ts_avgsp.png", 
                      bbox_inches='tight', pad_inches=0.05, dpi=700)  
    plt.show(block=True)
    
if __name__ == '__main__':
    main()    
    
    
