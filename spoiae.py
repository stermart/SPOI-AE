## Pytorch

from typing import Callable
import torch
import torch.nn as nn
import torch.nn.functional as F

from torch.utils.data import Dataset, DataLoader

from scipy.special import comb
import numpy as np
from numpy import linalg as la

class SPOI_AE(nn.Module):
    def __init__(self, N_wls, k_tot=16, k_NMF=2, 
                 mua_dim=[400], mus_dim=[1000], p=0.2, 
                 E=None, learnE=True):
        
        # super
        super(SPOI_AE, self).__init__()
        
        # store inputs
        self.L = N_wls
        self.k = k_tot
        self.top_k = k_NMF
        self.p = p
        self.mua_dim = mua_dim.copy()
        self.mus_dim = mus_dim.copy()
        self.E0 = E
        self.learnE = learnE
        
        # modify deep-NN dims
        self.mua_dim.insert(0, self.L)
        self.mua_dim.append(self.L)
        self.mus_dim.insert(0, self.L)
        self.mus_dim.append(self.L)
        
        # create mu_a deep-NN
        self.mua_net = []
        for i in range(len(self.mua_dim)-2):
            self.mua_net.append(nn.Linear(self.mua_dim[i], self.mua_dim[i+1]))
            self.mua_net.append(nn.LeakyReLU())
            self.mua_net.append(nn.BatchNorm1d(self.mua_dim[i+1], affine=True, eps=1e-8))
        self.mua_net.append(nn.Linear(self.mua_dim[-2], self.mua_dim[-1]))
        self.mua_net.append(nn.ReLU())
        self.mua_net = nn.Sequential(*self.mua_net)
        
        # create mu_s deep-NN
        self.mus_net = []
        for i in range(len(self.mus_dim)-2):
            self.mus_net.append(nn.Linear(self.mus_dim[i], self.mus_dim[i+1]))
            self.mus_net.append(nn.LeakyReLU())
            self.mus_net.append(nn.BatchNorm1d(self.mus_dim[i+1], affine=True, eps=1e-8))
        self.mus_net.append(nn.Linear(self.mus_dim[-2], self.mus_dim[-1]))
        self.mus_net.append(nn.ReLU())
        self.mus_net = nn.Sequential(*self.mus_net)
        
        # use parameterization for absorption spectra
        self.E = nn.Parameter(1e-3*torch.randn(self.L, self.k), requires_grad=self.learnE)
        if self.E0 is not None:
            self.E.data[:, :self.top_k] = torch.tensor(self.E0 / np.max(self.E0), dtype=self.E.dtype) # normalized
        self.c_dropout = nn.Dropout(p=self.p)
        
        # define Gruneisen * Wavelength-dependant Fluence parameter term        
        self.grun_flu = nn.Parameter(torch.ones(self.L))        

    def absorb_coeff(self, p):
        # so-called "mu_a"
        
        return self.mua_net(p)
    
    def est_absorb_coeff(self, p):
        # so-called "mu_a-hat"
        
        mu_a = self.absorb_coeff(p)
        
        c, c_prime = self.encode(mu_a)
        
        mu_a_hat = self.decode(c, c_prime)
        return mu_a_hat
    
    def reduced_scatter_coeff(self, p):
        # so-called "mu_s'"

        return self.mus_net(p)
        
    def encode(self, mu_a):
        # do fg and bg encode
        
        c_all = mu_a@F.relu(torch.pinverse(self.E).T)
        c = c_all[:, :self.top_k]
        c_prime = c_all[:, self.top_k:]
        return c, c_prime
    
    def encode_fg(self, mu_a):
        c, _ = self.encode(mu_a)
        return c
    
    def encode_bg(self, mu_a):
        _, c_prime = self.encode(mu_a)
        return c_prime
    
    def decode(self, c, c_prime):
        # do fg and bg decode
        
        c_all = torch.cat((c, self.c_dropout(c_prime)), dim=1)
        est_mu_a = F.relu(c_all @ self.E.T)
        return est_mu_a
    
    def eff_coeff(self, mu_a, reduced_mu_s):
        # so-called "mu_eff" 
        
        return torch.sqrt(3 * mu_a * (mu_a + reduced_mu_s) + 1e-10)
    
    def nonlinear_coeff(self, mu_eff, disp):
        # so-called "psi"
        
        return torch.exp(-(mu_eff * disp)) - 1
        
    def get_output(self, mu_a_hat, psi):
        # so-called "p-hat"
        
        l_part = mu_a_hat
        nl_part = psi * mu_a_hat
        p_hat = F.relu(self.grun_flu) * (l_part + nl_part)
        return p_hat
    
    def unmix(self, inputs):
        p = inputs[:, :-1] # TODO: fix to accept 1 by N_feature tensors
        disp = torch.unsqueeze(inputs[:, -1], dim=1)
        mu_a = self.absorb_coeff(p)
        c = self.encode_fg(mu_a)
        return c
    
    def fluence(self, inputs):
        p = inputs[:, :-1] # TODO: fix to accept 1 by N_feature tensors
        disp = torch.unsqueeze(inputs[:, -1], dim=1)
        
        mu_a = self.absorb_coeff(p)
        reduced_mu_s = self.reduced_scatter_coeff(p)
        c, c_prime = self.encode(mu_a)
        mu_a_hat = self.decode(c, c_prime)
        mu_eff = self.eff_coeff(mu_a_hat, reduced_mu_s)
        
        return F.relu(self.grun_flu) * torch.exp(-mu_eff * disp)        
        
    def forward(self, inputs):
        p = inputs[:, :-1] # TODO: fix to accept 1 by N_feature tensors
        disp = torch.unsqueeze(inputs[:, -1], dim=1)

        mu_a = self.absorb_coeff(p)
        reduced_mu_s = self.reduced_scatter_coeff(p)
        
        c, c_prime = self.encode(mu_a)
        mu_a_hat = self.decode(c, c_prime)
        
        mu_eff = self.eff_coeff(mu_a_hat, reduced_mu_s)
        psi = self.nonlinear_coeff(mu_eff, disp)
        
        p_hat = self.get_output(mu_a_hat, psi)
        
        return p_hat

def fast_img_psi(model: SPOI_AE, img: np.ndarray, disp_map: np.ndarray, device=None) -> np.ndarray:
    if device is None:
        device = torch.device("cpu")    
    
    height, width, channels = img.shape
    disp_height, disp_width = disp_map.shape
    
    assert height == disp_height, "PA Image and Displacment Map heights not equal"
    assert width == disp_width, "PA Image and Displacment Map widths not equal"
    
    img_flat = np.reshape(img, (height*width, channels))
    disp_flat = np.reshape(disp_map, (height*width, 1))
    
    psi_flat = fast_psi(model, img_flat, disp_flat, device)
        
    psi = np.reshape(psi_flat, (height, width, channels))
    return psi

def fast_psi(model: SPOI_AE, img_flat: np.ndarray, disp_flat: np.ndarray, device=None):
    if device is None:
        device = torch.device("cpu")   
    
    N, channels = img_flat.shape
    dispN = len(disp_flat)
    
    ret_flat = np.zeros((N, channels))
    
    img_ds = SPADataset(img_flat.T, disp_flat, use_idx=True)
    img_loader = DataLoader(img_ds, batch_size=25_000, shuffle=True)
    
    with torch.no_grad():    
        for i, (xr, x, idx) in enumerate(img_loader):
            xr, x = xr.float().to(device), x.float().to(device)
            print(f"Fast Psi {i+1:d}/{len(img_loader):d}......", end="", flush=True)
            disp = torch.tensor(disp_flat[idx], 
                                dtype=torch.float, 
                                device=device)
            mu_a = model.absorb_coeff(x)
            reduced_mu_s = model.reduced_scatter_coeff(x)
            c, c_prime = model.encode(mu_a)
            mu_a_hat = model.decode(c, c_prime)
            mu_eff = model.eff_coeff(mu_a_hat, reduced_mu_s)
            psi = model.nonlinear_coeff(mu_eff, disp)
            ret_flat[idx, :] = psi.detach().cpu().numpy()
            print("Done", flush=True)
            
    return ret_flat

def fast_muahat(model: SPOI_AE, img_flat: np.ndarray, disp_flat: np.ndarray, device=None):
    if device is None:
        device = torch.device("cpu")   
    
    N, channels = img_flat.shape
    dispN = len(disp_flat)
    
    ret_flat = np.zeros((N, channels))
    
    img_ds = SPADataset(img_flat.T, disp_flat, use_idx=True)
    img_loader = DataLoader(img_ds, batch_size=25_000, shuffle=True)
    
    with torch.no_grad():    
        for i, (xr, x, idx) in enumerate(img_loader):
            xr, x = xr.float().to(device), x.float().to(device)
            print(f"Fast Est Absortpion Coefficient {i+1:d}/{len(img_loader):d}......", end="", flush=True)
            ret_flat[idx, :] = model.est_absorb_coeff(x).detach().cpu().numpy()
            print("Done", flush=True)
            
    return ret_flat

def fast_img_muahat(model: SPOI_AE, img: np.ndarray, disp_map: np.ndarray, device=None) -> np.ndarray:
    if device is None:
        device = torch.device("cpu")    
    
    height, width, channels = img.shape
    disp_height, disp_width = disp_map.shape
    
    assert height == disp_height, "PA Image and Displacment Map heights not equal"
    assert width == disp_width, "PA Image and Displacment Map widths not equal"
    
    img_flat = np.reshape(img, (height*width, channels))
    disp_flat = np.reshape(disp_map, (height*width, 1))
    
    muahat_flat = fast_muahat(model, img_flat, disp_flat, device)
        
    muahat = np.reshape(muahat_flat, (height, width, channels))
    return muahat   

def fast_img_fluence(model: SPOI_AE, img: np.ndarray, disp_map: np.ndarray, device=None) -> np.ndarray:
    if device is None:
        device = torch.device("cpu")    
    
    height, width, channels = img.shape
    disp_height, disp_width = disp_map.shape
    
    assert height == disp_height, "PA Image and Displacment Map heights not equal"
    assert width == disp_width, "PA Image and Displacment Map widths not equal"
    
    img_flat = np.reshape(img, (height*width, channels))
    disp_flat = np.reshape(disp_map, (height*width, 1))
    
    flu_flat = fast_fluence(model, img_flat, disp_flat, device)
        
    flu = np.reshape(flu_flat, (height, width, channels))
    return flu

def fast_fluence(model: SPOI_AE, img_flat: np.ndarray, disp_flat: np.ndarray, device=None):
    if device is None:
        device = torch.device("cpu")   
    
    N, channels = img_flat.shape
    dispN = len(disp_flat)
    
    ret_flat = np.zeros((N, channels))
    
    img_ds = SPADataset(img_flat.T, disp_flat, use_idx=True)
    img_loader = DataLoader(img_ds, batch_size=25_000, shuffle=True)
    
    with torch.no_grad():    
        for i, (xr, x, idx) in enumerate(img_loader):
            xr, x = xr.float().to(device), x.float().to(device)
            print(f"Fast Fluence {i+1:d}/{len(img_loader):d}......", end="", flush=True)
            flu = model.fluence(xr)
            ret_flat[idx, :] = flu.detach().cpu().numpy()
            print("Done", flush=True)
            
    return ret_flat

def fast_mua(model: SPOI_AE, img_flat: np.ndarray, disp_flat: np.ndarray, device=None):
    if device is None:
        device = torch.device("cpu")   
    
    N, channels = img_flat.shape
    dispN = len(disp_flat)
    
    ret_flat = np.zeros((N, channels))
    
    img_ds = SPADataset(img_flat.T, disp_flat, use_idx=True)
    img_loader = DataLoader(img_ds, batch_size=25_000, shuffle=True)
    
    with torch.no_grad():    
        for i, (xr, x, idx) in enumerate(img_loader):
            xr, x = xr.float().to(device), x.float().to(device)
            print(f"Fast Absorption Coefficient {i+1:d}/{len(img_loader):d}......", end="", flush=True)
            ret_flat[idx, :] = model.absorb_coeff(x).detach().cpu().numpy()
            print("Done", flush=True)
            
    return ret_flat

def fast_img_mua(model: SPOI_AE, img: np.ndarray, disp_map: np.ndarray, device=None) -> np.ndarray:
    if device is None:
        device = torch.device("cpu")    
    
    height, width, channels = img.shape
    disp_height, disp_width = disp_map.shape
    
    assert height == disp_height, "PA Image and Displacment Map heights not equal"
    assert width == disp_width, "PA Image and Displacment Map widths not equal"
    
    img_flat = np.reshape(img, (height*width, channels))
    disp_flat = np.reshape(disp_map, (height*width, 1))
    
    mua_flat = fast_mua(model, img_flat, disp_flat, device)
        
    mua = np.reshape(mua_flat, (height, width, channels))
    
    return mua   

def fast_img_mus(model: SPOI_AE, img: np.ndarray, disp_map: np.ndarray, device=None) -> np.ndarray:
    if device is None:
        device = torch.device("cpu")    
    
    height, width, channels = img.shape
    disp_height, disp_width = disp_map.shape
    
    assert height == disp_height, "PA Image and Displacment Map heights not equal"
    assert width == disp_width, "PA Image and Displacment Map widths not equal"
    
    img_flat = np.reshape(img, (height*width, channels))
    disp_flat = np.reshape(disp_map, (height*width, 1))
    
    mus_flat = fast_mus(model, img_flat, disp_flat, device)
        
    mus = np.reshape(mus_flat, (height, width, channels))
    return mus

def fast_mus(model: SPOI_AE, img_flat: np.ndarray, disp_flat: np.ndarray, device=None):
    if device is None:
        device = torch.device("cpu")   
    
    N, channels = img_flat.shape
    dispN = len(disp_flat)
    
    ret_flat = np.zeros((N, channels))
    
    img_ds = SPADataset(img_flat.T, disp_flat, use_idx=True)
    img_loader = DataLoader(img_ds, batch_size=25_000, shuffle=True)
    
    with torch.no_grad():    
        for i, (xr, x, idx) in enumerate(img_loader):
            xr, x = xr.float().to(device), x.float().to(device)
            print(f"Fast Reduced Scattering Coefficient {i+1:d}/{len(img_loader):d}......", end="", flush=True)
            mus = model.reduced_scatter_coeff(x)
            ret_flat[idx, :] = mus.detach().cpu().numpy()
            print("Done", flush=True)
            
    return ret_flat

def fast_unmix(model: SPOI_AE, img_flat: np.ndarray, disp_flat: np.ndarray, device=None):
    if device is None:
        device = torch.device("cpu")   
    
    N, channels = img_flat.shape
    dispN = len(disp_flat)
    
    ret_flat = np.zeros((N, model.top_k))
    
    img_ds = SPADataset(img_flat.T, disp_flat, use_idx=True)
    img_loader = DataLoader(img_ds, batch_size=25_000, shuffle=True)
    
    with torch.no_grad():    
        for i, (xr, x, idx) in enumerate(img_loader):
            xr = xr.float().to(device)
            print(f"Fast Unmixing {i+1:d}/{len(img_loader):d}......", end="", flush=True)
            unmixed = model.unmix(xr)
            ret_flat[idx, :] = unmixed.detach().cpu().numpy()
            print("Done", flush=True)
            
    return ret_flat


def fast_img_unmix(model: SPOI_AE, img: np.ndarray, disp_map: np.ndarray, device=None) -> np.ndarray:
    if device is None:
        device = torch.device("cpu")    
    
    height, width, channels = img.shape
    disp_height, disp_width = disp_map.shape
    
    assert height == disp_height, "PA Image and Displacment Map heights not equal"
    assert width == disp_width, "PA Image and Displacment Map widths not equal"
    
    img_flat = np.reshape(img, (height*width, channels))
    disp_flat = np.reshape(disp_map, (height*width, 1))
    
    con_flat = fast_unmix(model, img_flat, disp_flat, device)
        
    con = np.reshape(con_flat, (height, width, model.top_k))
    return con   

def fast_remix(model: SPOI_AE, img_flat: np.ndarray, disp_flat: np.ndarray, device=None):
    if device is None:
        device = torch.device("cpu")    
    
    N, channels = img_flat.shape
    dispN = len(disp_flat)
    
    ret_flat = np.zeros((N, channels))
    
    img_ds = SPADataset(img_flat.T, disp_flat, use_idx=True)
    img_loader = DataLoader(img_ds, batch_size=25_000, shuffle=True)
    
    with torch.no_grad():    
        for i, (xr, x, idx) in enumerate(img_loader):
            xr = xr.float().to(device)
            print(f"Fast Remixing {i+1:d}/{len(img_loader):d}......", end="", flush=True)
            remixed = model(xr)
            ret_flat[idx, :] = remixed.detach().cpu().numpy()
            print("Done", flush=True)    
            
    return ret_flat


def fast_img_remix(model: SPOI_AE, img: np.ndarray, disp_map: np.ndarray, device=None) -> np.ndarray:
    if device is None:
        device = torch.device("cpu")   
    
    height, width, channels = img.shape
    disp_height, disp_width = disp_map.shape
    
    assert height == disp_height, "PA Image and Displacment Map heights not equal"
    assert width == disp_width, "PA Image and Displacment Map widths not equal"
    
    img_flat = np.reshape(img, (height*width, channels))
    disp_flat = np.reshape(disp_map, (height*width, 1))
    
    reimg_flat = fast_remix(model, img_flat, disp_flat, device)
        
    reimg = np.reshape(reimg_flat, (height, width, channels))
    return reimg   
    
# class SNV2Loss(nn.Module):
#     def __init__(self, l0, l1):
#         super(SNV2Loss, self).__init__()
        
#         self.l0 = l0
#         self.l1 = l1
        
#         self.mse_criterion = nn.MSELoss()
#         self.kl_criterion  = nn.KLDivLoss(reduction="batchmean")
        
#     def forward(self, input, target):
#         if input.shape[1] == target.shape[1]+1:
#             input = input[:, :-1]
#         loss = self.l0 * self.mse_criterion(input, target)
#         loss += self.l1 * self.kl_criterion(target, input)
#         return loss    
        
class DataParallelPassthrough(torch.nn.DataParallel):
    def __getattr__(self, name):
        try:
            return super().__getattr__(name)
        except AttributeError:
            return getattr(self.module, name)  

class ComboError(nn.Module):
    def __init__(self, alpha, beta, weight=None, size_average=True):
        super(ComboError, self).__init__()
        self.mse_criterion = nn.MSELoss()
        self.msad_criterion = MSADLoss()
        self.alpha = alpha
        self.beta = beta
        
    def forward(self, inputs, target):
        if inputs.shape[1] == target.shape[1]+1:
            inputs = inputs[:, :-1]
        self._mse = self.mse_criterion(inputs, target)
        self._msad = self.msad_criterion(inputs, target)
        combo_loss = self.alpha * self._mse \
            + self.beta * self._msad
        return combo_loss

# class ReducedScatteringError(nn.Module):
#     def __init__(self, weight=None, size_average=None):
#         super(ReducedScatteringError, self).__init__()
        
#     def forward(self, inputs):
#         ret = torch.mean(torch.linalg.norm(inputs, ord=2, dim=1))
#         return ret
    
# def decoderReg(model: SPOI_AE, sad: nn.Module):
#     A_dec1 = model.dec1_linear[0].weight # L by n
#     A_dec_prime = model.dec2_linear[0].weight # L by k
#     A_dec = torch.cat((A_dec1, A_dec_prime), dim=1) # L by n+k
    
#     reg = 1/comb(model.k, 2)
    
#     ret = 0
#     for i in range(A_dec.shape[1]-1):
#         for j in range(i+1, A_dec.shape[1]):
# #             print(i, A_dec.shape[1]-1, j,  A_dec.shape[1], 
# #                   torch.unsqueeze(A_dec[:,i], dim=0).shape, 
# #                   torch.unsqueeze(A_dec[:,j], dim=0).shape, 
# #                   sad(torch.unsqueeze(A_dec[:,i], dim=0), torch.unsqueeze(A_dec[:,j], dim=0)))
#             ret += torch.squeeze(sad(torch.unsqueeze(A_dec[:,i], dim=0), torch.unsqueeze(A_dec[:,j], dim=0)))
#     return 1 - reg*ret
    
# class SADLoss(nn.Module):
#     def __init__(self, weight=None, size_average=True):
#         super(SADLoss, self).__init__()
        
#     def forward(self, inputs, target):
#         dotprod = torch.sum(torch.mul(inputs, target), 1)
#         inputs_norm = torch.norm(inputs, p=2, dim=1)
#         target_norm = torch.norm(target, p=2, dim=1)
#         norm_prod = torch.mul(inputs_norm, target_norm) + 1e-10
#         sad_dist = 2/torch.pi * torch.acos(dotprod / norm_prod)
#         return sad_dist
    
# class SADDLoss(nn.Module):
#     def __init__(self, weight=None, size_average=True):
#         super(SADDLoss, self).__init__()
        
#     def forward(self, inputs, target):
#         dotprod = torch.sum(torch.mul(inputs, target), 1)
#         inputs_norm = torch.norm(inputs, p=2, dim=1)
#         target_norm = torch.norm(target, p=2, dim=1)
#         norm_prod = torch.mul(inputs_norm, target_norm) + 1e-10
#         sad_dist = 1 - 2/torch.pi * torch.acos(dotprod / norm_prod)
#         log_sad_dist = -torch.log2(sad_dist)
#         sadd_dist = torch.sum(log_sad_dist)
#         sadd_dist = sadd_dist / inputs.shape[1]
        
#         return sadd_dist
    
class MSADLoss(nn.Module):
    def __init__(self, weight=None, size_average=True):
        super(MSADLoss, self).__init__()
        
    def forward(self, inputs, target):
        dotprod = torch.sum(torch.mul(inputs, target), 1)
        inputs_norm = torch.norm(inputs, p=2, dim=1)
        target_norm = torch.norm(target, p=2, dim=1)
        norm_prod = torch.mul(inputs_norm, target_norm) + 1e-10
        sad_dist = 2/torch.pi * torch.acos(dotprod / norm_prod)
        msad = torch.mean(sad_dist, dim=0)
        return msad   

class SPADataset(Dataset):
    def __init__(self, data, disp, use_idx=False):
        super(SPADataset, self).__init__()
        
        self.data = np.concatenate((data, disp.T), axis=0).T.astype("float32")
        self.targets = data.T.astype("float32")
        self.use_idx = use_idx
        
    def __getitem__(self, index):
        if self.use_idx:
            return self.data[index], self.targets[index], index
        else:
            return self.data[index], self.targets[index]
            
    
    def __len__(self):
        return self.data.shape[0]

# class SNV1Loss(nn.Module):
#     def __init__(self, snv1_model: SPOI_AE, 
#                  e2e_criterion: nn.Module, ua_criterion: nn.Module, 
#                  us_criterion: nn.Module, Wdec_criterion: Callable, 
#                  l0=0, l1=0, l2=0, l3=0):
#         super(SNV1Loss, self).__init__()

#         self.snv1_model = snv1_model

#         self.e2e_criterion = e2e_criterion
#         self.ua_criterion = ua_criterion
#         self.us_criterion = us_criterion
#         self.Wdec_criterion = Wdec_criterion

#         self.l0 = l0
#         self.l1 = l1
#         self.l2 = l2
#         self.l3 = l3

#     def forward(self, xr, xest):
#         x = xr[:, :-1]
#         ret = self.l0 * self.e2e_criterion(xr, xest)
#         ret += self.l1 * self.ua_criterion(self.snv1_model.absorb_coeff(x), 
#                                            self.snv1_model.est_absorb_coeff(x))
#         ret += self.l2 * self.us_criterion(self.snv1_model.reduced_scatter_coeff(x))
#         ret += self.l3 * self.Wdec_criterion(self.snv1_model)
#         return ret
