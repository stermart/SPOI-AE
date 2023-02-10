## utility functions

import numpy as np
from scipy.io import loadmat, savemat
import matplotlib.pyplot as plt
from numpy import linalg as la
from scipy.optimize import nnls
import subprocess
import openpyxl as xl
from sklearn.metrics import r2_score
from sklearn.model_selection import KFold
import os

def nnls_unmix(E, Y):
    ret = la.lstsq(E, Y, rcond=None)[0]
    mask = np.any(ret < 0, axis=0)
        
    for i in np.where(mask)[0]:
        ret[:,i] = nnls(E, Y[:,i])[0]
    
    return ret, mask

def nnls_unmix_cube(E, sPA):
    Y = np.reshape(sPA, (np.prod(sPA.shape[0:2]), sPA.shape[2])).T
    C, _ = nnls_unmix(E, Y)
    C = C.T
    ret = np.reshape(C, (sPA.shape[0], sPA.shape[1], E.shape[1]))
    return ret

def sad(X, Y):
    num = np.sum(X * Y, axis=0)
    den = la.norm(X, ord=2, axis=0) * la.norm(Y, ord=2, axis=0)
    print(num.shape, den.shape)

    print(num.shape, den.shape)
    print(np.max(num/den))

    ret = 2 / np.pi * np.arccos(num/den)
    return ret

def get_gpu_memory_map():
    """Get the current gpu usage.

    Returns
    -------
    usage: dict
        Keys are device ids as integers.
        Values are memory usage as integers in MB.
    """
    result = subprocess.check_output(
        [
            'nvidia-smi', '--query-gpu=memory.used',
            '--format=csv,nounits,noheader'
        ], encoding='utf-8')
    # Convert lines into a dictionary
    gpu_memory = [int(x) for x in result.strip().split('\n')]
    gpu_memory_map = dict(zip(range(len(gpu_memory)), gpu_memory))
    return gpu_memory_map

def load_spectra(spectra_fname):
    spectra_book = xl.load_workbook(spectra_fname)
    ws = spectra_book['eps_a']

    A = [[i.value for i in j[1:3]] for j in ws.rows]
    E = np.array(A[1:])
    return E

def generate_wls(wl_start, wl_end, dwl):
    return np.arange(wl_start, wl_end+dwl, dwl)

def load_data(data_fname):
    fname_dict = loadmat(data_fname)
    data = fname_dict["data"].T.astype(np.double)
    disp = fname_dict["dist"].astype(np.double)
    fnames = [f.strip() for f in fname_dict["fnames"]]

    return data, disp, fnames

def norm_data(data):
    scale = 1 / (3*np.std(data))
    reg_data = scale * data
    return reg_data, scale

def split_data_3(data, disp, Nfolds):
    kf1 = KFold(n_splits=Nfolds, shuffle=True)
    kf2 = KFold(n_splits=Nfolds-1, shuffle=True)

    trxv_idx, ts_idx = next(kf1.split(data.T))
    trxv, trxv_r = data[:, trxv_idx], disp[trxv_idx]

    tr_idx, xv_idx = next(kf2.split(trxv.T))
    ts, ts_r = data[:, ts_idx], disp[ts_idx]
    tr, tr_r = data[:, tr_idx], disp[tr_idx]
    xv, xv_r = data[:, xv_idx], disp[xv_idx]

    return tr, tr_r, xv, xv_r, ts, ts_r

def split_data_2(data, disp):
    kf1 = KFold(n_splits=5, shuffle=True)
    
    tr_idx, ts_idx = next(kf1.split(data.T))
    ts, ts_r = data[:, ts_idx], disp[ts_idx]
    tr, tr_r = data[:, tr_idx], disp[tr_idx]
    
    return tr, tr_r, ts, ts_r

def load_sPA_img(fname):
    fname_dict = loadmat(fname)
    spa = fname_dict["spa"].astype(np.double)
    us = fname_dict["us"].astype(np.double)
    mask = fname_dict["roi_mask"].astype(np.double)
    disp = fname_dict["disp_mask"].astype(np.double)
    x_axis = np.around(np.squeeze(fname_dict["us_WidthAxis"].astype(np.double)), decimals=2)
    y_axis = np.around(np.squeeze(fname_dict["us_DepthAxis"].astype(np.double)), decimals=2)
    assert spa.shape[0:2] == us.shape == mask.shape, f"Invalid Data @ {fname}"
    return spa, us, mask, disp, x_axis, y_axis

def save_sPA_img(fname, spa, us, mask, disp, x_axis, y_axis):
    save_dict = {}
    save_dict["spa"] = spa
    save_dict["us"] = us
    save_dict["roi_mask"] = mask
    save_dict["disp_mask"] = disp
    save_dict["us_WidthAxis"] = x_axis
    save_dict["us_DepthAxis"] = y_axis
    os.makedirs(os.path.dirname(fname), exist_ok=True)
    savemat(fname, save_dict, do_compression=True)

def unmix_img(spa_img, unmix_fun_pw):
    Y = np.reshape(spa_img, (np.prod(spa_img.shape[0:2]), spa_img.shape[2])).T
    C = unmix_fun_pw(Y)
    unmix_img = np.reshape(C.T, (spa_img.shape[0], spa_img.shape[1], C.shape[0]))
    return unmix_img

def img_mask(abs_con_img, mask_img):
    SO2_img = abs_con_img[:,:,0] / np.sum(abs_con_img, axis=2) * 100
    mask_SO2_img = np.ma.masked_where(mask_img, SO2_img)
    return mask_SO2_img

def plot_mask(us_img, abs_con_img, mask_img, save_path=None):
    SO2_img = abs_con_img[:,:,0] / np.sum(abs_con_img, axis=2) * 100
    mask_SO2_img = np.ma.masked_where(mask_img, SO2_img)

    plt.figure()
    plt.imshow(us_img, cmap="gray")
    plt.imshow(mask_SO2_img, cmap="RdBu_r", vmin=0, vmax=100, alpha=1)
    plt.colorbar()

    plt.title("%SO2 Overlay")

    if save_path is not None:
        plt.savefig(save_path, dpi=300)

    plt.draw()

def plot_spectra(E, wls=None, name=None, c=None):
    plt.figure()

    if wls is None:
        for i in range(E.shape[1]):
            if c is not None:
                plt.plot(E[:,i], c=c[i])
            else:
                plt.plot(E[:, i])
    else:
        for i in range(E.shape[1]):
            if c is not None:
                plt.plot(wls, E[:,i], c=c[i])
            else:
                plt.plot(wls, E[:, i])
        plt.xlabel("Wavelength [nm]")

    plt.ylabel("Absorption")

    if name is None:
        name = ""

    plt.title(f"{name} Spectra")

    plt.legend([f"Abs. {_+1:d}" for _ in range(E.shape[1])])
    plt.draw()

def R2(data_true, data_est):
    return  r2_score(data_true.T, data_est.T, multioutput='raw_values')

def plot_R2(r2_list, wls=None, names=None, c=None, title=None):
    if names is not None:
        assert len(r2_list) == len(names), "Name each method"
    else:
        names = [f"Method {_:d}" for _ in range(len(r2_list))]
    
    fig, ax = plt.subplots(figsize=(6,4))
    if wls is None:
        for i in range(len(r2_list)):
            if c is not None:
                ax.plot(r2_list[i], marker='None', linestyle='-', c=c[i])
            else:
                ax.plot(r2_list[i], marker='None', linestyle='-')
    else:
        for i in range(len(r2_list)):
            if c is not None:
                ax.plot(wls, r2_list[i], marker='None', linestyle='-', c=c[i])
            else:
                ax.plot(wls, r2_list[i], marker='None', linestyle='-')
        plt.xlabel("Laser Wavelength (nm)")

    plt.ylabel("Voxel Reconstruction Accuracy ($\mathregular{R^2}$)")
    if title is not None:
        plt.title(title)

    plt.legend(names)
    plt.draw()

def plot_mag(data_true, data_est_list, wls=None, names=None, c=None, title=None):
    if names is not None:
        assert len(data_est_list) == len(names), "Name each method"
    else:
        names = [f"Method {_:d}" for _ in range(len(data_est_list))]

    plt.figure()
    
    means_true = np.mean(np.abs(data_true), axis=1)
    means = [np.mean(np.abs(data), axis=1) for data in data_est_list]
    means.append(means_true)

    if wls is None:
        for i in range(len(means)):
            if c is not None:
                plt.plot(means[i], c=c[i])
            else:
                plt.plot(means[i])
    else:
        for i in range(len(means)):
            if c is not None:
                plt.plot(wls, means[i], c=c[i])
            else:
                plt.plot(wls, means[i])
        plt.xlabel("Wavelength [nm]")
    
    plt.ylabel("Mean Intenstity")
    if title is not None:
        plt.title(title)
    else:
        plt.title("PA Magnitude")
    plt.legend(names + ["Input"])
    plt.draw()

def norm_spa_img(spa: np.ndarray):
    scale = 1 / (3*np.std(spa))
    return spa*scale, scale

