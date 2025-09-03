
import torch
import numpy as np
import torch.nn as nn
import math
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import argparse
# custom functions imported from the utils.py file available within the package
from utils import *

DEFAULTS = {
    "n_of_modes": 10,
    "Planes": 7,
    "iterations": 300,
    # objective weights
    "alpha": 1.0,
    "beta": 2.0,
    "gamma": 0.0,
    # optimization schedule
    "first_n_iterations": 10,
    "delta_theta_1": 2*math.pi/255,  # usual step size
    "delta_theta_0": 10*(2*math.pi/255),  # bigger step size (default 10x)
    # sampling / optics
    "Nx": 512,
    "Ny": 512,
    "pixelSize": 8e-6,
    "wavelength": 1.57e-6,
    # propagation distances
    "d_in": 20e-3,
    "d": 2*9.7e-3,
    "d_out": 15e-3,
    # evaluation cadence / early stop scale
    "calc_perf_every_it": 10,
    # features
    "equalize_efficiency": 1,
    "plot_eff_distribution": 0,
    "smoothing_switch": 1,
    # smoothing strength
    "OffsetMultiplier": 0e-5,
    # extras
    "plot_results": 0,
    "do_padded_eval": 0,
}

def parse_cfg() -> dict:
    parser = argparse.ArgumentParser(add_help=True)
    # ints
    parser.add_argument("--n_of_modes", type=int, default=None)
    parser.add_argument("--Planes", type=int, default=None)
    parser.add_argument("--iterations", type=int, default=None)
    parser.add_argument("--first_n_iterations", type=int, default=None)
    parser.add_argument("--Nx", type=int, default=None)
    parser.add_argument("--Ny", type=int, default=None)
    parser.add_argument("--calc_perf_every_it", type=int, default=None)
    parser.add_argument("--equalize_efficiency", type=int, choices=[0,1], default=None)
    parser.add_argument("--plot_eff_distribution", type=int, choices=[0,1], default=None)
    parser.add_argument("--smoothing_switch", type=int, choices=[0,1], default=None)
    parser.add_argument("--plot_results", type=int, choices=[0,1], default=None)
    parser.add_argument("--do_padded_eval", type=int, choices=[0,1], default=None)
    # floats
    parser.add_argument("--alpha", type=float, default=None)
    parser.add_argument("--beta", type=float, default=None)
    parser.add_argument("--gamma", type=float, default=None)
    parser.add_argument("--delta_theta_1", type=float, default=None)
    parser.add_argument("--delta_theta_0", type=float, default=None)
    parser.add_argument("--pixelSize", type=float, default=None)
    parser.add_argument("--wavelength", type=float, default=None)
    parser.add_argument("--d_in", type=float, default=None)
    parser.add_argument("--d", type=float, default=None)
    parser.add_argument("--d_out", type=float, default=None)
    parser.add_argument("--OffsetMultiplier", type=float, default=None)

    try:
        args = parser.parse_args()
    except SystemExit:
        # in notebooks or if imported, ignore CLI parsing side-effect
        args = argparse.Namespace()
    cfg = DEFAULTS.copy()
    for k, v in vars(args).items() if hasattr(args, "__dict__") else []:
        if v is not None:
            cfg[k] = v
    return cfg

CFG = parse_cfg()

# concise explicit unpacking (friendly to linters and readers)
(n_of_modes, Planes, iterations,
 alpha, beta, gamma,
 first_n_iterations, delta_theta_1, delta_theta_0,
 Nx, Ny, pixelSize, wavelength,
 d_in, d, d_out,
 calc_perf_every_it,
 equalize_efficiency, plot_eff_distribution, smoothing_switch, OffsetMultiplier) = (
     CFG["n_of_modes"], CFG["Planes"], CFG["iterations"],
     CFG["alpha"], CFG["beta"], CFG["gamma"],
     CFG["first_n_iterations"], CFG["delta_theta_1"], CFG["delta_theta_0"],
     CFG["Nx"], CFG["Ny"], CFG["pixelSize"], CFG["wavelength"],
     CFG["d_in"], CFG["d"], CFG["d_out"],
     CFG["calc_perf_every_it"],
     CFG["equalize_efficiency"], CFG["plot_eff_distribution"], CFG["smoothing_switch"], CFG["OffsetMultiplier"])

# Select device (prefer CUDA)
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"[MPLC2] Using device: {DEVICE}")

# derived parameters
reprW, reprH = Nx * pixelSize, Ny * pixelSize
crs_delta = 0.0001 * calc_perf_every_it
maskOffset = OffsetMultiplier * np.sqrt(1e-3 / (Nx * Ny * n_of_modes))

# wavelength-independent grids
nx_m = pixelSize*np.linspace(-(Nx-1)/2, (Nx-1)/2, num=Nx)
ny_m = pixelSize*np.linspace(-(Ny-1)/2, (Ny-1)/2, num=Ny)
X,Y = np.meshgrid(nx_m,ny_m)
X_torch = torch.from_numpy(X).to(DEVICE)
Y_torch = torch.from_numpy(Y).to(DEVICE)

nx = np.linspace(-(Nx-1)/2, (Nx-1)/2, num=Nx)
ny = np.linspace(-(Ny-1)/2, (Ny-1)/2, num=Ny)
kx, ky = np.meshgrid(2*np.pi*nx/(Nx*pixelSize),2*np.pi*ny/(Ny*pixelSize))


lambda_list = np.array([1.53e-6, 1.55e-6, 1.57e-6, 1.59e-6, 1.61e-6, 1.625e-6], dtype=np.float64)
lambda_c = 1.57e-6

# 读取LP模式和高斯输出（多波长）
lp_data = np.load('modes_lp_10.npz')
lp_modes = lp_data['profiles']  # 形状: (L, 10, 512, 512)
gauss_data = np.load('gauss_5x2_custom.npz')
gauss_modes = gauss_data['profiles']  # 形状: (L, 10, 512, 512)

L = min(lp_modes.shape[0], gauss_modes.shape[0], len(lambda_list))
lambda_list = lambda_list[:L]

Speckle_basis = lp_modes[:L, 0:n_of_modes, :, :]
Gaussian_basis = gauss_modes[:L, 0:n_of_modes, :, :]
Speckle_basis_torch = torch.from_numpy(Speckle_basis).to(torch.cdouble).to(DEVICE)
Gaussian_basis_torch = torch.from_numpy(Gaussian_basis).to(torch.cdouble).to(DEVICE)

# 生成多波长高斯mask
Gaussian_Masks = np.zeros_like(Gaussian_basis, dtype=np.float64)
for l in range(L):
    for m in range(n_of_modes):
        inten = np.abs(Gaussian_basis[l, m, :, :]) ** 2
        thr = 0.05 * np.max(inten)
        Gaussian_Masks[l, m, :, :] = inten > thr
Gaussian_Masks_torch = torch.from_numpy(Gaussian_Masks).to(torch.double).to(DEVICE)

# 若需要pad
if (Nx > 512) or (Ny > 512):
    pad_x = int((Nx-512)/2)
    pad_y = int((Ny-512)/2)
    Speckle_basis_torch = nn.functional.pad(Speckle_basis_torch, (pad_x, Nx-512-pad_x, pad_y, Ny-512-pad_y), mode='constant', value=0.+0.j)
    Gaussian_basis_torch = nn.functional.pad(Gaussian_basis_torch, (pad_x, Nx-512-pad_x, pad_y, Ny-512-pad_y), mode='constant', value=0.+0.j)
    Gaussian_Masks_torch = nn.functional.pad(Gaussian_Masks_torch, (pad_x, Nx-512-pad_x, pad_y, Ny-512-pad_y), mode='constant', value=0.0)

# 多波长下的 phi_bk 与 phi_cr
phi_bk = torch.ones((Gaussian_Masks_torch.shape[0], Ny, Nx), dtype=torch.double, device=DEVICE) - torch.sum(Gaussian_Masks_torch, axis = 1)
phi_cr = torch.zeros((Gaussian_Masks_torch.shape[0], n_of_modes, Ny, Nx), dtype = torch.double, device=DEVICE)
for l in range(Gaussian_Masks_torch.shape[0]):
    for i in range(n_of_modes):
        phi_cr[l,i,:,:] = torch.sum(Gaussian_Masks_torch[l], axis = 0) - Gaussian_Masks_torch[l,i,:,:]

phi = Gaussian_basis_torch

# # visualize one of the input modes, a set of Gaussians on the outputs and a binary mask outlining the backgroud region
# # brightness = amplitude, colour = phase
# plt.title("One of the input modes - $\chi_{0}$")
# complim(Speckle_basis_torch[0, :, :])

# plt.title("Sum of the output modes - $\sum\phi_{i}$")
# complim(torch.sum(phi, axis = 0))

# plt.title("$\phi^{bk}$")
# complim(phi_bk)

# plt.title("$\phi_{0}^{cr}$")
# complim(phi_cr[0,:,:])




Masks = torch.zeros((Planes,Ny,Nx), dtype=torch.double, device=DEVICE) # use zero phases as starting guesses for the phase masks
Masks_complex = torch.exp(1j*Masks) # complex representation of the phase masks with amplitude = 1 everywhere

# create placeholder arrays to store every input and every output field in each plane
L = Gaussian_Masks_torch.shape[0]
Modes_in = torch.zeros((L, Planes, n_of_modes, Ny, Nx), dtype = torch.cdouble, device=DEVICE)
Modes_out = torch.zeros((L, Planes, n_of_modes, Ny, Nx), dtype = torch.cdouble, device=DEVICE)

overlap = torch.zeros((n_of_modes), dtype = torch.cdouble, device=DEVICE)
eff_distribution = torch.ones((n_of_modes), dtype = torch.double, device=DEVICE)
dFdpsi = torch.zeros((L, Planes, n_of_modes, Ny, Nx), dtype = torch.cdouble, device=DEVICE)
crs_array_convergence = torch.zeros((iterations//calc_perf_every_it), dtype = torch.double, device=DEVICE)
conv_count = 0

# 每个波长的 kz，初始化 Modes_in/Out
kz_torch_list = []
for l in range(L):
    k_l = (2*np.pi)/lambda_list[l]
    kz_l = np.sqrt(k_l**2 - (kx**2 + ky**2))
    kz_torch_list.append(torch.from_numpy(kz_l.astype(np.cdouble)).to(DEVICE))
    Modes_in[l, 0, :, :, :] = propagate_HK(Speckle_basis_torch[l], kz_torch_list[l], d_in)
    # 目标场定义在输出面（距最后一面 d_out 处），用于反向传播到最后一面
    Modes_out[l, Planes-1, :, :, :] = propagate_HK(phi[l], kz_torch_list[l], -d_out)

# iterate 
for i in range(1, iterations+1):

    # change the step size depending on the current iteration number
    if i < first_n_iterations:
        delta_theta = delta_theta_0
    else:
        delta_theta = delta_theta_1

    # update all the phase masks on this iteration in an ascending order
    for mask_ind in range(Planes):

        # 多波长：按 λ 比例缩放相位并分别前后传播
        for l in range(L):
            scale_l = lambda_c / lambda_list[l]
            modes = torch.zeros((n_of_modes, Ny, Nx), dtype = torch.cdouble, device=DEVICE)
            for pl in range(Planes-1):
                mask_cmplx_l = torch.exp(1j*(Masks[pl, :, :]*scale_l))
                modes = Modes_in[l, pl, :, :, :] * mask_cmplx_l
                modes = propagate_HK(modes, kz_torch_list[l], d)
                Modes_in[l, pl+1, :, :, :] = modes
            modes_forw_last_plane = Modes_in[l, Planes-1, :, :, :] * torch.exp(1j*(Masks[Planes-1, :, :]*scale_l))
            # 从最后一面向前传播到真实输出面 z_out
            eout_l = propagate_HK(modes_forw_last_plane, kz_torch_list[l], d_out)

            for j in range(n_of_modes):
                overlap = torch.sum(torch.squeeze(eout_l[j,:,:]) * torch.conj(torch.squeeze(phi[l,j,:,:])))
                a = (phi[l, j, :, :]) * overlap
                psi_cr_l = (torch.squeeze(eout_l[j,:,:])) * torch.squeeze(phi_cr[l,j,:,:])
                psi_bk_l = (torch.squeeze(eout_l[j,:,:])) * phi_bk[l]
                dFdpsi[l, Planes-1, j, :, :] = - alpha*a + (beta*psi_cr_l - gamma*psi_bk_l)*0.5

            # 将输出面上的梯度场反向传播回最后一面
            dFdpsi[l, Planes-1, :, :, :] = propagate_HK(dFdpsi[l, Planes-1, :, :, :], kz_torch_list[l], -d_out)

            for pl in range(Planes-1, mask_ind, -1):
                mask_cmplx_l = torch.exp(1j*(Masks[pl, :, :]*scale_l))
                dFdpsi_prop = dFdpsi[l, pl, :, :, :] * torch.conj(mask_cmplx_l)
                dFdpsi_prop = propagate_HK(dFdpsi_prop, kz_torch_list[l], -d)
                dFdpsi[l, pl-1, :, :, :] = dFdpsi_prop

                phi_prop = Modes_out[l, pl, :, :, :] * torch.conj(mask_cmplx_l)
                phi_prop = propagate_HK(phi_prop, kz_torch_list[l], -d)
                Modes_out[l, pl-1, :, :, :] = phi_prop

        # if equalize_efficiency is on, make a sum in (1) a weighted sum, where the weights are 1/(relative_efficiency_i) for each particular mode            
        if equalize_efficiency == 1:
            total_term = torch.zeros((Ny,Nx), dtype=torch.cdouble, device=DEVICE)
            for l in range(L):
                scale_l = lambda_c / lambda_list[l]
                mask_cmplx_l = torch.exp(1j*(Masks[mask_ind, :, :]*scale_l))
                weighted_overlaps = torch.zeros((Ny,Nx), dtype=torch.cdouble, device=DEVICE)
                for mode in range(n_of_modes):
                    weighted_overlaps = weighted_overlaps + (1/eff_distribution[mode]) * torch.squeeze(Modes_in[l, mask_ind, mode, :, :]) * torch.conj(torch.squeeze(dFdpsi[l, mask_ind, mode, :, :]))
                total_term = total_term + mask_cmplx_l * weighted_overlaps
            delta_P = delta_theta*torch.sign(torch.imag(total_term))
        else:
            total_term = torch.zeros((Ny,Nx), dtype=torch.cdouble, device=DEVICE)
            for l in range(L):
                scale_l = lambda_c / lambda_list[l]
                mask_cmplx_l = torch.exp(1j*(Masks[mask_ind, :, :]*scale_l))
                overlaps = torch.sum(torch.squeeze(Modes_in[l, mask_ind, :, :, :]) * torch.conj(torch.squeeze(dFdpsi[l, mask_ind, :, :, :])), axis = 0)
                total_term = total_term + mask_cmplx_l * overlaps
            delta_P = delta_theta*torch.sign(torch.imag(total_term))
        
        #  if smoothing_switch is on, mask the regions of the phase masks where there is almost no incedent light, based on the overlap of input and output modes at this plane
        if smoothing_switch == 1:
                ov_sum = torch.zeros((Ny, Nx), dtype=torch.double, device=DEVICE)
                for l in range(L):
                    ov_sum = ov_sum + torch.abs(torch.sum(torch.squeeze(Modes_in[l, mask_ind, :, :, :]*torch.conj(Modes_out[l, mask_ind, :, :, :])), axis = 0))
                ovrlp_in_out = ov_sum / L
                mask_cmplx = ovrlp_in_out*torch.exp(1j*(Masks[mask_ind, :, :] + delta_P)) 
                mask_cmplx = mask_cmplx + maskOffset
                Masks[mask_ind, :, :] = torch.angle(mask_cmplx)
        #  if smoothing_switch is off, just add phase delta_P to a current guess of the certain phase mask
        else:
            Masks[mask_ind, :, :] = Masks[mask_ind, :, :] + delta_P

        # store the resulting current guess of the phase mask as a complex array, with amplitude = 1 everywhere
        Masks_complex[mask_ind, :, :] = torch.exp(1j*torch.squeeze(Masks[mask_ind, :, :]))


    # calculate and print out sorter's performance after every iteration (or every K iterations to save time)
    if i % calc_perf_every_it == 0:
        fids = []
        crss = []
        effs = []
        for l in range(L):
            scale_l = lambda_c / lambda_list[l]
            for pl in range(Planes-1):
                mask_cmplx_l = torch.exp(1j*(Masks[pl, :, :]*scale_l))
                modes = Modes_in[l, pl, :, :, :]*mask_cmplx_l
                modes = propagate_HK(modes, kz_torch_list[l], d)
                Modes_in[l, pl+1, :, :, :] = modes
            modes = modes*torch.exp(1j*(Masks[Planes-1, :, :]*scale_l))
            eout = propagate_HK(modes, kz_torch_list[l], d_out)
            eout_int_only = (torch.abs(eout))**2
            fid, _ = performance_loc_fidelity(eout, Gaussian_Masks_torch[l], phi[l]) 
            crs, _, _ = performance_crosstalk(eout_int_only, Gaussian_Masks_torch[l]) 
            eff, eff_list = performance_efficiency(eout_int_only, Gaussian_Masks_torch[l])
            fids.append(fid); crss.append(crs); effs.append(eff)

        fid = torch.stack(fids).mean(); crs = torch.stack(crss).mean(); eff = torch.stack(effs).mean()
        print('iteration', i, ': loc. fidelity =', round(fid.detach().cpu().numpy().item(),2), ', crosstalk =', round(crs.detach().cpu().numpy().item(),2), ', efficiency =', round(eff.detach().cpu().numpy().item(),2))
        crs_array_convergence[conv_count] = crs # store calculated cross-talk to an array to then plot it against the number of iterations
        
        # stop iterating if the algorithm is no longer improving cross-talk by more than a certain value after a certain iteration
        if i > (iterations/3) and (crs_array_convergence[conv_count-1] - crs_array_convergence[conv_count]) < crs_delta:
            break
        conv_count = conv_count + 1

        # store a list of a relative efficiency of every output on the current iteration to try to equalize them on the next run
        if equalize_efficiency == 1:
            eff_distribution = eff_list/torch.max(eff_list)
            # plot efficiency distribution if plot_eff_distribution is on
            if plot_eff_distribution == 1:                    
                plt.plot(eff_distribution)
                plt.title('efficiency distribution')
                plt.ylim((0,1))
                plt.show()
        
fids = []; crss = []; effs = []
for l in range(L):
    scale_l = lambda_c / lambda_list[l]
    for pl in range(Planes-1):
        modes = Modes_in[l, pl, :, :, :]*torch.exp(1j*(Masks[pl, :, :]*scale_l))
        modes = propagate_HK(modes, kz_torch_list[l], d)
        Modes_in[l, pl+1, :, :, :] = modes
    modes = modes*torch.exp(1j*(Masks[Planes-1, :, :]*scale_l))
    eout = propagate_HK(modes, kz_torch_list[l], d_out)
    eout_int_only = (torch.abs(eout))**2
    fid, _ = performance_loc_fidelity(eout, Gaussian_Masks_torch[l], phi[l])
    crs, _, _ = performance_crosstalk(eout_int_only, Gaussian_Masks_torch[l])
    eff, _ = performance_efficiency(eout_int_only, Gaussian_Masks_torch[l])
    fids.append(fid); crss.append(crs); effs.append(eff)
fid = torch.stack(fids).mean(); crs = torch.stack(crss).mean(); eff = torch.stack(effs).mean()
print('Final performance (avg over λ): loc. fidelity =', round(fid.detach().cpu().numpy().item(),3), ', crosstalk =', round(crs.detach().cpu().numpy().item(),3), ', efficiency =', round(eff.detach().cpu().numpy().item(),3))

if CFG.get("plot_results", 0) == 1:
    # 展示相位面
    for i in range(Planes):
        plt.title("Phase mask %s" %(i+1))
        _ = plot_in_GS(Masks[i,:,:])

# 逐波长性能打印
for idx, (f_i, c_i, e_i) in enumerate(zip(fids, crss, effs)):
    print(f"λ={lambda_list[idx]*1e6:.3f} µm -> fidelity={float(f_i.detach().cpu().numpy()):.3f}, crosstalk={float(c_i.detach().cpu().numpy()):.3f}, efficiency={float(e_i.detach().cpu().numpy()):.3f}")

if CFG.get("do_padded_eval", 0) == 1:
    newNx = Nx + 400
    newNy = Ny + 400
    l_c = 2 if L >= 3 else 0
    Modes_in_wide = torch.zeros((Planes,n_of_modes,newNx,newNy), dtype=torch.cdouble)
    Modes_in_wide[0,:,200:200+Nx,200:200+Ny] = Modes_in[l_c,0,:,:,:]
    Masks_wide = torch.zeros((Planes,newNy,newNx), dtype = torch.double)
    Masks_complex_wide = torch.exp(1j*Masks_wide)
    Masks_complex_wide[:,200:200+Nx,200:200+Ny] = Masks_complex
    nx_wide = np.linspace(-(newNx-1)/2, (newNx-1)/2, num=newNx)
    ny_wide = np.linspace(-(newNy-1)/2, (newNy-1)/2, num=newNy)
    kx_wide, ky_wide = np.meshgrid(2*np.pi*nx_wide/(newNx*pixelSize),2*np.pi*ny_wide/(newNy*pixelSize))
    kz_wide = np.sqrt((2*np.pi/lambda_c)**2 - (kx_wide**2 + ky_wide**2)).astype(np.cdouble)
    kz_torch_wide = torch.from_numpy(kz_wide)
    for pl in range(Planes-1):
        modes = Modes_in_wide[pl, :, :, :]*Masks_complex_wide[pl, :, :]
        modes = propagate_HK(modes, kz_torch_wide, d)
        Modes_in_wide[pl+1, :, :, :] = modes
    modes = modes*Masks_complex_wide[Planes-1,:,:]
    modes_cropped = modes[:,200:200+Nx,200:200+Ny]
    # 在宽域上从最后一面传播到输出面，再裁剪评估
    eout_wide = propagate_HK(modes, kz_torch_wide, d_out)
    eout_cropped = eout_wide[:,200:200+Nx,200:200+Ny]
    eout_cropped_int_only = (torch.abs(eout_cropped))**2
    fid_wide, _ = performance_loc_fidelity(eout_cropped, Gaussian_Masks_torch[l_c], phi[l_c])
    crs_wide, _, _ = performance_crosstalk(eout_cropped_int_only, Gaussian_Masks_torch[l_c])
    eff_wide, _ = performance_efficiency(eout_cropped_int_only, Gaussian_Masks_torch[l_c])
    print('performance padded (λc): loc. fidelity =', round(fid_wide.detach().numpy().item(),3), ', crosstalk =', round(crs_wide.detach().numpy().item(),3), ', efficiency =', round(eff_wide.detach().numpy().item(),3))

    plt.plot(crs_array_convergence)
    plt.ylabel('avg. crosstalk (avg over λ)')
    plt.xlabel('iterations/(calc_perf_every_it)')
    plt.axis([0, iterations//calc_perf_every_it, 0, 20])
    plt.show()


# ==========================================
# Visualization: λ=1.57 μm 前/后向“相位前”分布与相位图
# - 前向快照: z=0, p0..p6 的 pre-phase (传播到该面, 未乘该面相位), 以及 z_out
# - 后向快照: z_out, p6..p0 的 pre-phase (从后向传播到该面, 未乘该面相位), 以及 z=0
# ==========================================
import os
os.makedirs('results', exist_ok=True)

with torch.no_grad():
    # 选择 λ=1.57 μm 的索引
    l_idx = int(np.argmin(np.abs(lambda_list - lambda_c)))
    kz_l = kz_torch_list[l_idx]
    scale_l = lambda_c / lambda_list[l_idx]

    # 前向: 收集相位前快照（总强度=所有模式强度求和）
    fwd_titles = []
    fwd_maps = []
    # z=0
    modes = Speckle_basis_torch[l_idx].clone()
    fwd_maps.append(torch.sum(torch.abs(modes) ** 2, dim=0))
    fwd_titles.append('z=0')
    # 传播到 p0 (pre-phase)
    modes = propagate_HK(modes, kz_l, d_in)
    fwd_maps.append(torch.sum(torch.abs(modes) ** 2, dim=0))
    fwd_titles.append('p0 pre')
    # 依次到 p1..p6 的 pre-phase
    for pl in range(0, Planes-1):  # 到 p1..p6 的pre，需要先在上一面乘相位再传播
        mask_cmplx = torch.exp(1j * (Masks[pl] * scale_l))
        modes = modes * mask_cmplx
        modes = propagate_HK(modes, kz_l, d)
        fwd_maps.append(torch.sum(torch.abs(modes) ** 2, dim=0))
        fwd_titles.append(f'p{pl+1} pre')
    # 输出面 z_out (在 p6 乘相位后传播 d_out)
    modes = modes * torch.exp(1j * (Masks[Planes-1] * scale_l))
    modes_out = propagate_HK(modes, kz_l, d_out)
    fwd_maps.append(torch.sum(torch.abs(modes_out) ** 2, dim=0))
    fwd_titles.append('z_out')

    # 后向: 从目标输出面场出发，收集各面的 pre-phase
    bwd_titles = []
    bwd_maps = []
    # z_out（目标场）
    modes_b = phi[l_idx].clone()
    bwd_maps.append(torch.sum(torch.abs(modes_b) ** 2, dim=0))
    bwd_titles.append('z_out')
    # 到 p6 pre：先 -d_out 到 p6 的后相位(post)，再乘 conj(mask6) 得 pre
    modes_b = propagate_HK(modes_b, kz_l, -d_out)
    mask6 = torch.exp(1j * (Masks[Planes-1] * scale_l))
    modes_b = modes_b * torch.conj(mask6)
    bwd_maps.append(torch.sum(torch.abs(modes_b) ** 2, dim=0))
    bwd_titles.append('p6 pre')
    # 依次到 p5..p0 的 pre：每步先 -d 到达上一面的 post，再乘对应 conj(mask) 得 pre
    for pl in range(Planes-2, -1, -1):  # from p5 down to p0
        modes_b = propagate_HK(modes_b, kz_l, -d)
        mask_cmplx = torch.exp(1j * (Masks[pl] * scale_l))
        modes_b = modes_b * torch.conj(mask_cmplx)
        bwd_maps.append(torch.sum(torch.abs(modes_b) ** 2, dim=0))
        bwd_titles.append(f'p{pl} pre')
    # 最后到 z=0：-d_in 传播
    modes_b = propagate_HK(modes_b, kz_l, -d_in)
    bwd_maps.append(torch.sum(torch.abs(modes_b) ** 2, dim=0))
    bwd_titles.append('z=0')

    # 画图：前向 9 幅（z=0, p0..p6, z_out）
    import matplotlib.pyplot as plt
    fig1, axes1 = plt.subplots(3, 3, figsize=(12, 10))
    for idx, ax in enumerate(axes1.ravel()):
        if idx < len(fwd_maps):
            im = ax.imshow(fwd_maps[idx].detach().cpu().numpy(), cmap='inferno', origin='lower')
            ax.set_title(fwd_titles[idx])
            ax.axis('off')
        else:
            ax.axis('off')
    fig1.suptitle('Forward pre-phase intensity (λ=1.57 μm)')
    fig1.tight_layout()
    fig1.savefig('results/forward_prephase_1p57.png', dpi=150)

    # 画图：后向 9 幅（z_out, p6..p0, z=0）
    fig2, axes2 = plt.subplots(3, 3, figsize=(12, 10))
    for idx, ax in enumerate(axes2.ravel()):
        if idx < len(bwd_maps):
            im = ax.imshow(bwd_maps[idx].detach().cpu().numpy(), cmap='inferno', origin='lower')
            ax.set_title(bwd_titles[idx])
            ax.axis('off')
        else:
            ax.axis('off')
    fig2.suptitle('Backward pre-phase intensity (λ=1.57 μm)')
    fig2.tight_layout()
    fig2.savefig('results/backward_prephase_1p57.png', dpi=150)

    # 相位图：7 个相位面
    fig3, axes3 = plt.subplots(2, 4, figsize=(12, 6))
    for p in range(Planes):
        r = p // 4
        c = p % 4
        ax = axes3[r, c]
        ax.imshow(Masks[p].detach().cpu().numpy(), cmap='twilight', origin='lower')
        ax.set_title(f'Mask p{p}')
        ax.axis('off')
    # 关掉多余子图
    if Planes < 8:
        for k in range(Planes, 8):
            r = k // 4
            c = k % 4
            axes3[r, c].axis('off')
    fig3.suptitle('Phase masks (radians)')
    fig3.tight_layout()
    fig3.savefig('results/masks_phase_maps.png', dpi=150)

    plt.show()


# ==========================================
# Subplot: 六个波长的耦合矩阵 + 指标 (IL, MDL, XTs_avg_dB, fidelity/crosstalk/efficiency)
# - 耦合矩阵基于输出面复场与目标复场的归一化内积 C_{m,j}=
#   <E_out_m, Phi_j>/sqrt(<E_out_m,E_out_m><Phi_j,Phi_j>)
# - IL=10*log10(mean(s^2)), MDL=10*log10(max(s^2)/min(s^2)), XTs_avg_dB=10*log10(mean(((sum|C|^2 - diag|C|^2)/diag|C|^2)))
# ==========================================
with torch.no_grad():
    Nl = len(lambda_list)
    modeCount = n_of_modes
    ILs = np.zeros(Nl)
    MDLs = np.zeros(Nl)
    XTs_avg_dB = np.zeros(Nl)
    fids_l = np.zeros(Nl)
    crss_l = np.zeros(Nl)
    effs_l = np.zeros(Nl)

    # 预创建子图
    nrows, ncols = 2, 3
    fig_cm, axes_cm = plt.subplots(nrows, ncols, figsize=(12, 8))

    for l in range(Nl):
        kz_l = kz_torch_list[l]
        scale_l = lambda_c / lambda_list[l]

        # 前向到输出面
        modes = propagate_HK(Speckle_basis_torch[l], kz_l, d_in)
        for pl in range(Planes-1):
            modes = modes * torch.exp(1j * (Masks[pl] * scale_l))
            modes = propagate_HK(modes, kz_l, d)
        modes = modes * torch.exp(1j * (Masks[Planes-1] * scale_l))
        eout = propagate_HK(modes, kz_l, d_out)  # (M, Ny, Nx)

        # 基于复内积构建耦合矩阵 C (M×M)
        E = eout.reshape(modeCount, -1)
        P = phi[l].reshape(modeCount, -1)
        num = E @ torch.conj(P).T  # (M,M)
        normE = torch.sum(torch.abs(E)**2, dim=1)  # (M,)
        normP = torch.sum(torch.abs(P)**2, dim=1)  # (M,)
        denom = torch.sqrt(normE[:, None] * normP[None, :]) + 1e-12
        C = num / denom

        # IL / MDL from SVD of C
        C_np = C.detach().cpu().numpy()
        s = np.linalg.svd(C_np, compute_uv=False)  # singular values
        s2 = s**2
        ILs[l] = 10 * np.log10(np.mean(s2))
        MDLs[l] = 10 * np.log10(np.max(s2) / (np.min(s2) + 1e-15))

        # XTs (per mode) and XTs_avg_dB
        C2 = np.abs(C_np)**2
        totalPower = np.sum(C2, axis=1)
        signalPower = np.clip(np.diag(C2), 1e-15, None)
        XTs_modes = (totalPower - signalPower) / signalPower
        XTs_avg_dB[l] = 10 * np.log10(np.mean(XTs_modes))

        # 同时计算 fidelity/crosstalk/efficiency（基于 mask 的原函数）
        eout_int = (torch.abs(eout))**2
        fid_l, _ = performance_loc_fidelity(eout, Gaussian_Masks_torch[l], phi[l])
        crs_l, _, _ = performance_crosstalk(eout_int, Gaussian_Masks_torch[l])
        eff_l, _ = performance_efficiency(eout_int, Gaussian_Masks_torch[l])
        fids_l[l] = float(fid_l.detach().cpu().numpy())
        crss_l[l] = float(crs_l.detach().cpu().numpy())
        effs_l[l] = float(eff_l.detach().cpu().numpy())

        # 绘制该波长的耦合矩阵（功率 |C|^2）
        r = l // ncols
        c = l % ncols
        ax = axes_cm[r, c]
        im = ax.imshow(C2, cmap='magma', origin='lower', aspect='equal')
        ax.set_title(f'λ={lambda_list[l]*1e6:.3f} μm')
        ax.set_xlabel('target idx')
        ax.set_ylabel('input mode')

    fig_cm.suptitle('Coupling matrices |C|^2 across wavelengths')
    fig_cm.tight_layout()
    fig_cm.savefig('results/coupling_matrices_6wls.png', dpi=150)
    plt.show()

    # 打印表格型结果（简洁版）
    print('Wavelengths (μm):', [f'{wl*1e6:.3f}' for wl in lambda_list])
    print('IL (dB):         ', [f'{v:.3f}' for v in ILs])
    print('MDL (dB):        ', [f'{v:.3f}' for v in MDLs])
    print('XTs_avg (dB):    ', [f'{v:.3f}' for v in XTs_avg_dB])
    print('fidelity:        ', [f'{v:.3f}' for v in fids_l])
    print('crosstalk:       ', [f'{v:.3f}' for v in crss_l])
    print('efficiency:      ', [f'{v:.3f}' for v in effs_l])


# ==========================================
# 追加可视化：λ=1.57 μm 时，10 个模式反向传播到 z=0 的强度图
# 结果保存：results/backward_z0_modes_1p57.png
# ==========================================
with torch.no_grad():
    import os
    os.makedirs('results', exist_ok=True)

    # 选择 λ=1.57 μm 对应索引与缩放
    l_idx = int(np.argmin(np.abs(lambda_list - lambda_c)))
    kz_l = kz_torch_list[l_idx]
    scale_l = lambda_c / lambda_list[l_idx]

    # 从目标面出发，逐面反向传播到 z=0（逐模式并行）
    modes_b = phi[l_idx].clone()  # (M, Ny, Nx)
    modes_b = propagate_HK(modes_b, kz_l, -d_out)
    modes_b = modes_b * torch.conj(torch.exp(1j * (Masks[Planes-1] * scale_l)))
    for pl in range(Planes-2, -1, -1):
        modes_b = propagate_HK(modes_b, kz_l, -d)
        modes_b = modes_b * torch.conj(torch.exp(1j * (Masks[pl] * scale_l)))
    modes_b = propagate_HK(modes_b, kz_l, -d_in)  # at z=0

    # 仅取前 10 个模式（或 n_of_modes 更小者）并绘制强度
    M = min(10, modes_b.shape[0], n_of_modes)
    rows, cols = 2, 5
    fig, axes = plt.subplots(rows, cols, figsize=(14, 6))
    axes = axes.ravel()
    for j in range(rows * cols):
        if j < M:
            inten = torch.abs(modes_b[j]) ** 2
            axes[j].imshow(inten.detach().cpu().numpy(), cmap='inferno', origin='lower')
            axes[j].set_title(f'mode {j} @ z=0')
            axes[j].axis('off')
        else:
            axes[j].axis('off')
    fig.suptitle('Backward to z=0 per-mode intensity (λ=1.57 μm)')
    fig.tight_layout()
    fig.savefig('results/backward_z0_modes_1p57.png', dpi=150)
    plt.show()


