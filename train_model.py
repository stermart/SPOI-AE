# imports - numpy & scipy
import numpy as np

# imports - utils
from datetime import datetime   
from time import perf_counter
import os
import json
import dill

# imports - pytorch
import torch
from torch.utils.data import DataLoader
from torch import cuda

# imports - visualizations
from torchviz import make_dot

# spoiae libraries
import pa_util
import spoiae

def main():
    
    # CUDA
    print(f"CUDA Available? {cuda.is_available()}")
    if cuda.is_available():
        available_gpus = [(i, torch.cuda.get_device_name(torch.cuda.device(i))) 
                          for i in range(torch.cuda.device_count())]
        lbl = '\n - '
        print(f"Available GPUs: {lbl}{lbl.join(map(str, available_gpus))}")
    
    devices = []
    for i in range(max(cuda.device_count(), 1)):
        devices.append(
            torch.device(f"cuda:{i}") if cuda.is_available() else "cpu") 
    print(devices)
    
    # Load Literature Endmembers
    E_lit = pa_util.load_spectra("Hbspectra2.xlsx")
    wls = pa_util.generate_wls(680, 970, 2)
    
    # Load In Vivo Data
    
    tr, tr_r, tr_fnames = pa_util.load_data("Data/Segmented_600_tr9.mat")
    xv, xv_r, xv_fnames = pa_util.load_data("Data/Segmented_600_xv2.mat")
    tr = np.concatenate((tr, xv), axis=1)
    tr_r = np.concatenate((tr_r, xv_r), axis=0)
    tr_fnames = tr_fnames + xv_fnames
    tr, tr_scale = pa_util.norm_data(tr)
    
    ts, ts_r, ts_fnames = pa_util.load_data("Data/Segmented_600_ts3.mat")
    ts *= tr_scale    
    
    print("Training Dims", tr.shape, tr_r.shape)
    print("Testing Dims", ts.shape, ts_r.shape)
    
    lit_unmix = lambda y: pa_util.nnls_unmix(E_lit, y)[0]
    
    # SPOI-AE Parameters
    num_epochs = 500
    batch_size = 20_000
    learning_rate = 3e-4
    lr_decay = 1
    p_dropout = 0.2
    k_tot = 6
    k_fg = 2
    mua_dim = [200, 200, 200]
    mus_dim = [1000, 1000, 1200, 1200]
    adapt_spectra = False
    num_workers = 0
    weight_decay = 1e-6
    #use_nesterov = True
    #momentum = 0.99
    #dampening = 0
    min_lr = 3e-4
    max_lr = 4e-4
    lr_decay_dur = 150

    path = f"{os.getcwd()}/Trained Models/User Created/{datetime.now():%Y%m%d-%H%M%S}"
    os.makedirs(os.path.dirname(f"{path}/temp.txt"), exist_ok=False)
    print(f"Dumping to {path}")
    
    # SPOI-AE Loss Function Parameters
    alpha = 1e2
    beta = 0e0

    print(f"MSE weight: {alpha:.3g}, MSAD weight: {beta:.3g}", flush=True)
    
    # Definitions
    spoiae_model = spoiae.SPOI_AE(len(wls), k_tot=k_tot, k_NMF=k_fg, 
                                   mua_dim=mua_dim, mus_dim=mus_dim,
                                   p=p_dropout,
                                   E=E_lit, 
                                   learnE=adapt_spectra).float().to(devices[0])
    print(spoiae_model)
    print(spoiae_model.E, spoiae_model.E.device)
    print(spoiae_model.grun_flu)
    spoiae_parallel = spoiae.DataParallelPassthrough(spoiae_model)    
    
    final_criterion = spoiae.ComboError(alpha, beta)
    
    optimizer = torch.optim.AdamW(
        spoiae_model.parameters(),
        lr=1,
        weight_decay=weight_decay
    )

    def lr_fn(epoch, min_lr, max_lr, dur):
        if epoch < dur:
            return max_lr - ((max_lr - min_lr) / dur) * epoch
        else:
            return min_lr

    lr_scheduler = torch.optim.lr_scheduler.LambdaLR(
        optimizer=optimizer,
        lr_lambda=lambda epoch: lr_fn(epoch, min_lr, max_lr, lr_decay_dur)
    )
    
    training_dataset = spoiae.SPADataset(tr, tr_r)
    training_loader = DataLoader(training_dataset,
                                 batch_size=batch_size*len(devices),
                                 shuffle=True,
                                 num_workers=num_workers)
    
    eval_datasets = [spoiae.SPADataset(tr, tr_r), 
                     spoiae.SPADataset(ts, ts_r)]
    eval_loaders = list(map(
        lambda ds: DataLoader(ds, 
                              batch_size=batch_size*len(devices),
                              shuffle=True,
                              num_workers=num_workers),
        eval_datasets))
    
    # Get Literature Baseline Performance
    with torch.no_grad():
        lit_mse_tr = final_criterion.mse_criterion(
            torch.tensor(E_lit@lit_unmix(tr[:, :5*batch_size])).T, 
            torch.tensor(tr[:, :5*batch_size].T)).item()
        lit_mse_ts = final_criterion.mse_criterion(
            torch.tensor(E_lit@lit_unmix(ts)).T, 
            torch.tensor(ts.T)).item()
        print(f"Literature SPOI-AE MSE: {lit_mse_tr:.6g}, {lit_mse_ts:.6g}")
        lit_msad_tr = final_criterion.msad_criterion(
            torch.tensor(tr[:, :5*batch_size].T),
            torch.tensor(E_lit@lit_unmix(tr[:, :5*batch_size])).T).item()
        lit_msad_ts = final_criterion.msad_criterion(
            torch.tensor(ts.T),
            torch.tensor(E_lit@lit_unmix(ts)).T).item()
        print(f"Literatuve SPOI-AE MSAD: {lit_msad_tr:.6g}, {lit_msad_ts:.6g}")
        
    # dump model architecture
    dummy_tensor = torch.tensor(np.concatenate((tr[:, 0:5].T, tr_r[0:5]), 
                                               axis=1), 
                                dtype=torch.float, device=devices[0])
    dummy_eval = spoiae_model(dummy_tensor)
    make_dot(dummy_eval, params=dict(spoiae_model.named_parameters()), 
             show_attrs=True, 
             show_saved=True).render(f"{path}/spoiae_full", 
                                     format="pdf", 
                                     cleanup=True)
    make_dot(dummy_eval, params=dict(spoiae_model.named_parameters()), 
             show_attrs=False, 
             show_saved=False).render(f"{path}/spoiae_notfull", 
                                      format="pdf", 
                                      cleanup=True)    
        
    # dump model architecture
    torch.save(spoiae_model, f"{path}/model_arch.pt", pickle_module=dill)
    
    # dump loss function
    torch.save(final_criterion, f"{path}/loss_obj.pt", pickle_module=dill)
    
    # final training prep
    eval_hist = []
    torch.autograd.set_detect_anomaly(False)
    labels = ["MSE", "MSAD", "Final Loss", "Timing", "LR"]
    
    title_str = "|".join(map(lambda x: f"{x:^15}", labels))
    epoch_lbl = f"Epoch/{num_epochs:d}"
    print(f"{epoch_lbl:^18}|{title_str}")
    
    line_sep = "-"*len(f"{epoch_lbl:<12}|{title_str}")
    print(line_sep, flush=True)
   
    best_loss = np.inf

    # calculate pre-training error  
    spoiae_parallel.to(devices[0])  
    spoiae_parallel.eval()
    with torch.no_grad():
        t1 = perf_counter()
        epoch_eval_mat = []
        for loader in eval_loaders:
            eval_lossvec = []
            eval_sizevec = []
            for xr, x in loader:
                xr, x = xr.float().to(devices[0]), x.float().to(devices[0])
                xest = spoiae_parallel(xr)
                mse = final_criterion.mse_criterion(x, xest).item()
                msad = final_criterion.msad_criterion(xest, x).item()             
                loss = final_criterion(xr, xest).item()
                lossvec = [mse, msad, loss]
                eval_lossvec.append(lossvec)
                eval_sizevec.append(x.shape[0])
            eval_sizevec = np.array(eval_sizevec)
            eval_lossvec = np.array(eval_lossvec)
            eval_lossvec = np.sum(eval_lossvec * np.expand_dims(
                    eval_sizevec, axis=1), axis=0) \
                    / np.sum(eval_sizevec, axis=0)
            epoch_eval_mat.append(eval_lossvec)
        t1 = perf_counter() - t1
        eval_hist.append(epoch_eval_mat)
        epoch_eval_str_tr = "|".join(map(lambda x: f"{x:^15.6g}", 
                                       epoch_eval_mat[0]))
        epoch_eval_str_ts = "|".join(map(lambda x: f"{x:^15.6g}", 
                                       epoch_eval_mat[1]))
        epoch_lbl_str_tr = f"tr {0:>5d}/{num_epochs:d}"
        epoch_lbl_str_ts = f"ts {0:>5d}/{num_epochs:d}"
        print(f"{epoch_lbl_str_tr:^18}|{epoch_eval_str_tr}"
              f"|{t1:^15.6g}|{lr_scheduler.get_last_lr()[0]:^15.6g}", 
              flush=True)
        print(f"{epoch_lbl_str_ts:^18}|{epoch_eval_str_ts}", 
              flush=True)
        print(line_sep, flush=True)
        
        torch.save(spoiae_model.state_dict(), 
                   f"{path}/initial_state.pt", 
                   pickle_module=dill)
        
    # training loop
    for epoch in range(num_epochs):
        # training
        spoiae_parallel.train()
        
        optimizer.zero_grad()
        
        t1 = perf_counter()
        t2 = [perf_counter()]
        for n, (xr, x) in enumerate(training_loader):
            xr, x = xr.float().to(devices[0]), x.float().to(devices[0])
            xest = spoiae_parallel(xr)
            loss = final_criterion(xr, xest)
            loss.backward()
            optimizer.step()
            optimizer.zero_grad()
            t2[-1] = perf_counter() - t2[-1]
            
            lossvec = [f"{final_criterion._mse.item():.6g}", 
                       f"{final_criterion._msad.item():.6g}",
                       f"{loss.item():.6g}",
                       f"{t2[-1]:.6g}",
                       f"{lr_scheduler.get_last_lr()[0]:.6g}"]
            epoch_eval_str = "|".join(map(lambda x: f"{x:^15}", lossvec))

            epoch_lbl_str = f"{n+1:>5d}/{len(training_loader):d}"
            print(f"{epoch_lbl_str:^18}|{epoch_eval_str}", flush=True)
            
            t2.append(perf_counter())
        
        t2 = t2[:-1]
        t2 = np.mean(np.array(t2))
        
        # evaluating
        spoiae_parallel.eval()
        with torch.no_grad():
            epoch_eval_mat = []
            for loader in eval_loaders:
                eval_lossvec = []
                eval_sizevec = []
                for xr, x in loader:
                    xr, x = xr.float().to(devices[0]), x.float().to(devices[0])
                    xest = spoiae_parallel(xr)
                    mse = final_criterion.mse_criterion(x, xest).item()
                    msad = final_criterion.msad_criterion(xest, x).item()             
                    loss = final_criterion(xr, xest).item()
                    lossvec = [mse, msad, loss]
                    eval_lossvec.append(lossvec)
                    eval_sizevec.append(x.shape[0])
                eval_sizevec = np.array(eval_sizevec)
                eval_lossvec = np.array(eval_lossvec)
                eval_lossvec = np.sum(eval_lossvec * np.expand_dims(
                        eval_sizevec, axis=1), axis=0) \
                        / np.sum(eval_sizevec, axis=0)
                epoch_eval_mat.append(eval_lossvec)
            t1 = perf_counter() - t1
            eval_hist.append(epoch_eval_mat)
            print(line_sep)
            epoch_eval_str_tr = "|".join(map(lambda x: f"{x:^15.6g}", 
                                             epoch_eval_mat[0]))
            epoch_eval_str_ts = "|".join(map(lambda x: f"{x:^15.6g}", 
                                             epoch_eval_mat[1]))
            epoch_lbl_str_tr = f"tr {epoch+1:>5d}/{num_epochs:d}"
            epoch_lbl_str_ts = f"ts {epoch+1:>5d}/{num_epochs:d}"
            print(f"{epoch_lbl_str_tr:^18}|{epoch_eval_str_tr}"
                  f"|{t1:^15.6g}|{lr_scheduler.get_last_lr()[0]:^15.6g}", 
                  flush=True)
            print(f"{epoch_lbl_str_ts:^18}|{epoch_eval_str_ts}", 
                  flush=True)
            print(line_sep, flush=True)
            
            torch.save(spoiae_model.state_dict(), 
                       f"{path}/final_state.pt", 
                       pickle_module=dill)
            if epoch_eval_mat[0][-1] < best_loss:
                print(f"New Best Model, {best_loss:.3f} -> {epoch_eval_mat[0][-1]:.3f}", flush=True)
                best_loss = epoch_eval_mat[0][-1]
                torch.save(spoiae_model.state_dict(), 
                           f"{path}/best_state.pt", 
                           pickle_module=dill)

            if epoch % 10 == 0:
                print("Autosaving Model", flush=True)
                os.makedirs(os.path.dirname(f"{path}/hist/{epoch+1:d}_state.pt"), 
                            exist_ok=True)
                torch.save(spoiae_model.state_dict(), 
                           f"{path}/hist/{epoch+1:d}_state.pt", 
                           pickle_module=dill)
                
            # step lr
            lr_scheduler.step()
            
        eval_hist_np = np.array(eval_hist)
        eval_hist_np = np.swapaxes(eval_hist_np, 0, 1)
        
        tr_eval_hist = eval_hist_np[0, :]        
        ts_eval_hist = eval_hist_np[1, :]
        
        with open(f"{path}/learning_curves.json", "w") as fout:
            json.dump({
                "Training": tr_eval_hist.tolist(),
                "Testing": ts_eval_hist.tolist(), 
                "LR Hist": [lr_fn(i, min_lr, max_lr, lr_decay_dur) 
                            for i in range(epoch)]
            }, fout, indent=2)   
    
if __name__ == '__main__':
    main()
