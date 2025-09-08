"""
根据用户需求：从文件中直接读取Gaussian_Masks。
"""
import torch
import numpy as np
import torch.nn as nn
import math
import matplotlib
import os
import json
import random
import datetime
# matplotlib.use("Agg")
import matplotlib.pyplot as plt
# 从文件中读取Gaussian_Masks，无需生成
import argparse
# custom functions imported from the utils.py file available within the package
from utils import *

DEFAULTS = {
    "n_of_modes": 10,
    "Planes": 9,
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
    "OffsetMultiplier": 1e-5,
    # extras
    "plot_results": 0,
    "do_padded_eval": 1,
    # acceleration: AMP removed; always use complex64 for stability
    "seed": 42,
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
    # AMP removed; no CLI option
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
    parser.add_argument("--seed", type=int, default=None)

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

# reproducibility: set seeds and deterministic flags
def seed_everything(seed: int = 42):
    try:
        np.random.seed(seed)
        random.seed(seed)
        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
    except Exception as _e:
        print(f"[MPLC2] Seed setup warning: {_e}")

seed_everything(CFG.get("seed", 42))

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
if DEVICE.type == "cuda":
    try:
        name = torch.cuda.get_device_name(0)
        cap = torch.cuda.get_device_capability(0)
        print(f"[MPLC2] GPU: {name}, capability={cap}, torch_cuda={getattr(torch.version, 'cuda', None)}")
    except Exception as e:
        print(f"[MPLC2] CUDA detected but failed to query device info: {e}")
else:
    print("[MPLC2] CUDA 不可用：将使用 CPU 运行。若期望使用 GPU，请安装 CUDA 版 PyTorch 并确保驱动正确。")

# 简要环境与配置记录
os.makedirs('results', exist_ok=True)
run_meta = {
    "timestamp": datetime.datetime.now().isoformat(),
    "device": str(DEVICE),
    "torch_version": getattr(torch, "__version__", None),
    "torch_cuda": getattr(torch.version, "cuda", None),
    "numpy_version": np.__version__,
    "cfg": CFG,
}
try:
    with open(os.path.join('results', 'run_meta.json'), 'w', encoding='utf-8') as f:
        json.dump(run_meta, f, ensure_ascii=False, indent=2)
    print("[MPLC2] 运行元信息已保存到 results/run_meta.json")
except Exception as e:
    print(f"[MPLC2] 保存运行元信息失败: {e}")

# AMP context removed per request

# derived parameters
crs_delta = 0.0001 * calc_perf_every_it
maskOffset = OffsetMultiplier * np.sqrt(1e-3 / (Nx * Ny * n_of_modes))
nx_m = pixelSize*np.linspace(-(Nx-1)/2, (Nx-1)/2, num=Nx)
ny_m = pixelSize*np.linspace(-(Ny-1)/2, (Ny-1)/2, num=Ny)
X,Y = np.meshgrid(nx_m,ny_m)
# X_torch/Y_torch 未使用，避免不必要的内存占用

# 使用 PyTorch 在目标设备上直接构建 k 空间网格，避免 CPU→GPU 传输
nx_t = torch.linspace(-(Nx-1)/2, (Nx-1)/2, steps=Nx, device=DEVICE, dtype=torch.float32)
ny_t = torch.linspace(-(Ny-1)/2, (Ny-1)/2, steps=Ny, device=DEVICE, dtype=torch.float32)
kx_1d = (2*math.pi) * nx_t / (Nx * pixelSize)
ky_1d = (2*math.pi) * ny_t / (Ny * pixelSize)
# 与 numpy 的 meshgrid(default 'xy') 对齐，得到 (Ny, Nx) 形状
kx_t, ky_t = torch.meshgrid(kx_1d, ky_1d, indexing='xy')
# 横向波数平方 (Ny, Nx)，float32 在设备上
k_t2_torch = kx_t**2 + ky_t**2


lambda_list = np.array([1.53e-6, 1.55e-6, 1.57e-6, 1.59e-6, 1.61e-6, 1.625e-6], dtype=np.float64)
lambda_c = 1.57e-6

# 读取LP模式和高斯输出（多波长）
lp_data = np.load('lp_out.npz')  # 从lp_out.npz读取LP模式
lp_modes = lp_data['profiles']  # shape=(6, 10, 512, 512)
gauss_data = np.load('gauss_5x2_4d.npz')  # 从gauss_5x2_4d.npz读取高斯模式和掩膜
gauss_modes = gauss_data['Gaussian_basis']  # shape=(6, 10, 512, 512)
gauss_masks = gauss_data['Gaussian_Masks']  # Gaussian_Masks

L = min(lp_modes.shape[0], gauss_modes.shape[0], len(lambda_list))
lambda_list = lambda_list[:L]

Speckle_basis = lp_modes[:L, 0:n_of_modes, :, :].astype(np.complex64)
Gaussian_basis = gauss_modes[:L, 0:n_of_modes, :, :].astype(np.complex64)
Speckle_basis_torch = torch.from_numpy(Speckle_basis).to(DEVICE)
Gaussian_basis_torch = torch.from_numpy(Gaussian_basis).to(DEVICE)
"""
根据用户需求：高斯阵列中各束相同，掩膜采用“1/e^2 半径 + 25 像素”的统一圆形区域。
步骤：
1) 以第一个波长、第一模式的强度作为模板，找到峰值中心。
2) 用 Ith=I0*exp(-2) 的阈值在中心附近(限制半径 r_cap)估计 1/e^2 半径 w。
3) 掩膜半径 r_mask = w + 25。
4) 将该圆形掩膜复制到所有 λ 与模式。
"""
Gaussian_Masks = gauss_masks[:L, 0:n_of_modes, :, :].astype(np.float32)
Gaussian_Masks_torch = torch.from_numpy(Gaussian_Masks).to(torch.float32).to(DEVICE)

# 若需要pad
if (Nx > 512) or (Ny > 512):
    pad_x = int((Nx-512)/2)
    pad_y = int((Ny-512)/2)
    Speckle_basis_torch = nn.functional.pad(Speckle_basis_torch, (pad_x, Nx-512-pad_x, pad_y, Ny-512-pad_y), mode='constant', value=0.+0.j)
    Gaussian_basis_torch = nn.functional.pad(Gaussian_basis_torch, (pad_x, Nx-512-pad_x, pad_y, Ny-512-pad_y), mode='constant', value=0.+0.j)
    Gaussian_Masks_torch = nn.functional.pad(Gaussian_Masks_torch, (pad_x, Nx-512-pad_x, pad_y, Ny-512-pad_y), mode='constant', value=0.0)

# 多波长下的 phi_bk 与 phi_cr（二值并集/补集定义，避免负值；在掩膜互斥时等价且更稳健）
sum_masks = torch.sum(Gaussian_Masks_torch, dim=1)  # (L, Ny, Nx)
phi_bk = (sum_masks == 0).to(torch.float32)  # 背景：未被任何通道覆盖
phi_cr = ((sum_masks.unsqueeze(1) - Gaussian_Masks_torch) > 0).to(torch.float32)  # 交叉区域：其他通道的并集

phi = Gaussian_basis_torch

# 简单预览：修复为单一 λ 与单一模式的可视化（仅在 plot_results=1 时运行）
if CFG.get("plot_results", 0) == 1:
    try:
        l0 = 0  # 选择第一个波长作演示
        # plt.figure(); plt.title("One of the input modes - $\\chi_{0}$"); complim(Speckle_basis_torch[l0, 0, :, :])
        # plt.figure(); plt.title("Sum of the output modes - $\\sum\\phi_{i}$"); complim(torch.sum(phi[l0], dim=0))
        # 修复：在高斯区域显示高斯场，在背景区域显示背景掩膜
        gauss_sum = torch.sum(phi[l0], dim=0)
        gauss_mask = torch.sum(Gaussian_Masks_torch[l0], dim=0) > 0  # 高斯区域 (Ny, Nx)
        # 创建组合场：高斯区域用原高斯场，背景区域用弱复数信号显示背景
        combined_field = torch.where(gauss_mask, gauss_sum, 0.1 * phi_bk[l0].to(torch.complex64))
        plt.figure(); plt.title("Gaussian field + background mask"); complim(combined_field)
        plt.figure(); plt.title("$\\phi^{bk}$"); complim(phi_bk[l0])
        plt.figure(); plt.title("$\\phi_{0}^{cr}$"); complim(phi_cr[l0, 0])
    except Exception as e:
        print("[MPLC2] Preview plots skipped:", e)



Masks = torch.zeros((Planes, Ny, Nx), dtype=torch.float32, device=DEVICE)  # use zero phases as starting guesses for the phase masks
Masks_complex = torch.exp(1j * Masks)  # complex representation of the phase masks with amplitude = 1 everywhere

# create placeholder arrays to store every input and every output field in each plane
L = Gaussian_Masks_torch.shape[0]
Modes_in = torch.zeros((L, Planes, n_of_modes, Ny, Nx), dtype=torch.complex64, device=DEVICE)
Phi_bwd = torch.zeros((L, Planes, n_of_modes, Ny, Nx), dtype=torch.complex64, device=DEVICE)
eff_distribution = torch.ones((n_of_modes), dtype=torch.float32, device=DEVICE)
dFdpsi = torch.zeros((L, Planes, n_of_modes, Ny, Nx), dtype=torch.complex64, device=DEVICE)
crs_array_convergence = torch.zeros((iterations//calc_perf_every_it), dtype = torch.double, device=DEVICE)
conv_count = 0

# 每个波长的 kz，初始化 Modes_in/Phi_bwd
kz_torch_list = []
for l in range(L):
    # 标量波数（float32）
    k_l_val = float((2*math.pi) / float(lambda_list[l]))
    # 重要：在复数域开方，保证倏逝波对应 imag(kz) != 0
    kz_sq = (k_l_val**2) - k_t2_torch  # real float32 tensor on DEVICE
    kz_c = torch.sqrt(kz_sq.to(torch.complex64))  # complex64 on DEVICE
    kz_torch_list.append(kz_c)
    Modes_in[l, 0] = propagate_HK(Speckle_basis_torch[l], kz_torch_list[l], d_in)
    # 目标场定义在输出面（距最后一面 d_out 处），用于反向传播到最后一面
    Phi_bwd[l, Planes-1] = propagate_HK(phi[l], kz_torch_list[l], -d_out)

# 额外缓存：每个波长的输入在 z=p0 处（从 z=0 传播 d_in）
modes_in0 = [propagate_HK(Speckle_basis_torch[l], kz_torch_list[l], d_in) for l in range(L)]

# 预计算各波长的相位缩放因子 scl_l = λc/λl
scls = [float(lambda_c / float(lambda_list[l])) for l in range(L)]

# iterate 
for i in range(1, iterations+1):

    # change the step size depending on the current iteration number
    if i < first_n_iterations:
        delta_theta = delta_theta_0
    else:
        delta_theta = delta_theta_1

    # update all the phase masks on this iteration in an ascending order
    # 每次迭代：构建一次完整的相位掩膜缓存（各波长×各平面），后续仅在单面更新后局部刷新
    mask_cache_per_lambda = [
        [torch.exp(1j * (Masks[pl] * scls[l])) for pl in range(Planes)]
        for l in range(L)
    ]

    for mask_ind in range(Planes):

        # 多波长：按 λ 比例缩放相位并分别前后传播
        for l in range(L):
            kz_l = kz_torch_list[l]
            mask_cmplx_all = mask_cache_per_lambda[l]
            # 从输入完整前向传播，同时刷新 Modes_in（d_in 部分使用预缓存）
            modes = modes_in0[l]
            Modes_in[l, 0] = modes
            for pl in range(Planes-1):
                modes = modes * mask_cmplx_all[pl]
                modes = propagate_HK(modes, kz_l, d)
                Modes_in[l, pl+1] = modes
            modes = modes * mask_cmplx_all[Planes-1]
            eout_l = propagate_HK(modes, kz_l, d_out)

            for j in range(n_of_modes):
                ovlp = torch.sum(torch.squeeze(eout_l[j,:,:]) * torch.conj(torch.squeeze(phi[l,j,:,:])))
                a = (phi[l, j, :, :]) * ovlp
                psi_cr_l = (torch.squeeze(eout_l[j,:,:])) * torch.squeeze(phi_cr[l,j,:,:])
                psi_bk_l = (torch.squeeze(eout_l[j,:,:])) * phi_bk[l]
                dFdpsi[l, Planes-1, j, :, :] = - alpha*a + (beta*psi_cr_l - gamma*psi_bk_l)*0.5

            # 将输出面上的梯度场反向传播回最后一面
            dFdpsi[l, Planes-1, :, :, :] = propagate_HK(dFdpsi[l, Planes-1, :, :, :], kz_torch_list[l], -d_out)

            for pl in range(Planes-1, mask_ind, -1):
                dFdpsi_prop = dFdpsi[l, pl, :, :, :] * torch.conj(mask_cmplx_all[pl])
                dFdpsi_prop = propagate_HK(dFdpsi_prop, kz_torch_list[l], -d)
                dFdpsi[l, pl-1, :, :, :] = dFdpsi_prop

                phi_prop = Phi_bwd[l, pl, :, :, :] * torch.conj(mask_cmplx_all[pl])
                phi_prop = propagate_HK(phi_prop, kz_torch_list[l], -d)
                Phi_bwd[l, pl-1, :, :, :] = phi_prop

        # if equalize_efficiency is on, make a sum in (1) a weighted sum, where the weights are 1/(relative_efficiency_i) for each particular mode            
        if equalize_efficiency == 1:
            total_term = torch.zeros((Ny, Nx), dtype=torch.complex64, device=DEVICE)
            # keep on device with consistent dtype
            inv_eff = (1.0 / eff_distribution.to(device=DEVICE)).view(n_of_modes, 1, 1)  # (M,1,1)
            for l in range(L):
                mask_cmplx_l = mask_cache_per_lambda[l][mask_ind]
                Mi = Modes_in[l, mask_ind]  # (M, Ny, Nx)
                Gi = dFdpsi[l, mask_ind]    # (M, Ny, Nx)
                weighted_overlaps = torch.sum(inv_eff * Mi * torch.conj(Gi), dim=0)  # (Ny, Nx)
                total_term = total_term + mask_cmplx_l * weighted_overlaps
            delta_P = delta_theta*torch.sign(torch.imag(total_term))
        else:
            total_term = torch.zeros((Ny, Nx), dtype=torch.complex64, device=DEVICE)
            for l in range(L):
                mask_cmplx_l = mask_cache_per_lambda[l][mask_ind]
                Mi = Modes_in[l, mask_ind]   # (M, Ny, Nx)
                Gi = dFdpsi[l, mask_ind]     # (M, Ny, Nx)
                overlaps = torch.sum(Mi * torch.conj(Gi), dim=0)
                total_term = total_term + mask_cmplx_l * overlaps
            delta_P = delta_theta*torch.sign(torch.imag(total_term))
        
        #  if smoothing_switch is on, mask the regions of the phase masks where there is almost no incedent light, based on the overlap of input and output modes at this plane
        if smoothing_switch == 1:
                ov_sum = torch.zeros((Ny, Nx), dtype=torch.float32, device=DEVICE)
                for l in range(L):
                    ov_sum = ov_sum + torch.abs(torch.sum(torch.squeeze(Modes_in[l, mask_ind, :, :, :] * torch.conj(Phi_bwd[l, mask_ind, :, :, :])), dim=0))
                # 归一化到 [0,1]
                ov_max = torch.amax(ov_sum)
                ovrlp_in_out = ov_sum / (ov_max + 1e-6)
                mask_cmplx = ovrlp_in_out * torch.exp(1j * (Masks[mask_ind, :, :] + delta_P))
                # add a tiny real offset in a dtype/device-safe way (optional smoothing bias)
                if maskOffset != 0:
                    mask_cmplx = mask_cmplx + torch.tensor(maskOffset, dtype=torch.float32, device=DEVICE)
                Masks[mask_ind, :, :] = torch.angle(mask_cmplx)
        #  if smoothing_switch is off, just add phase delta_P to a current guess of the certain phase mask
        else:
            Masks[mask_ind, :, :] = Masks[mask_ind, :, :] + delta_P

        # store the resulting current guess of the phase mask as a complex array, with amplitude = 1 everywhere
        Masks_complex[mask_ind, :, :] = torch.exp(1j * torch.squeeze(Masks[mask_ind, :, :]))
        # 刷新缓存：仅更新当前平面的相位（对所有波长），避免整套重算
        for l in range(L):
            mask_cache_per_lambda[l][mask_ind] = torch.exp(1j * (Masks[mask_ind] * scls[l]))


    # calculate and print out sorter's performance after every iteration (or every K iterations to save time)
    if i % calc_perf_every_it == 0:
        fids = []
        crss = []
        effs = []
        eff_lists = []
        for l in range(L):
            # 统一评估管线：从输入完整传播到输出
            kz_l = kz_torch_list[l]
            scl = lambda_c / lambda_list[l]
            modes = propagate_HK(Speckle_basis_torch[l], kz_l, d_in)
            for pl in range(Planes-1):
                modes = modes * torch.exp(1j*(Masks[pl]*scl))
                modes = propagate_HK(modes, kz_l, d)
            modes = modes * torch.exp(1j*(Masks[Planes-1]*scl))
            eout = propagate_HK(modes, kz_l, d_out)
            eout_int_only = (torch.abs(eout))**2
            fid, _ = performance_loc_fidelity(eout, Gaussian_Masks_torch[l], phi[l]) 
            crs, _, _ = performance_crosstalk(eout_int_only, Gaussian_Masks_torch[l]) 
            eff, eff_list = performance_efficiency(eout_int_only, Gaussian_Masks_torch[l])
            fids.append(fid); crss.append(crs); effs.append(eff)
            eff_lists.append(eff_list)

        fid = torch.stack(fids).mean(); crs = torch.stack(crss).mean(); eff = torch.stack(effs).mean()
        print('iteration', i, ': loc. fidelity =', round(fid.detach().cpu().numpy().item(),2), ', crosstalk =', round(crs.detach().cpu().numpy().item(),2), ', efficiency =', round(eff.detach().cpu().numpy().item(),2))
        crs_array_convergence[conv_count] = crs # store calculated cross-talk to an array to then plot it against the number of iterations
        
        # stop iterating if the algorithm is no longer improving cross-talk by more than a certain value after a certain iteration
        if (conv_count > 0) and (i > (iterations/3)) and ((crs_array_convergence[conv_count-1] - crs_array_convergence[conv_count]) < crs_delta):
            break
        conv_count = conv_count + 1

        # store a list of a relative efficiency of every output on the current iteration to try to equalize them on the next run
        if equalize_efficiency == 1:
            # 跨波长聚合（中位数），提升鲁棒性
            eff_stack = torch.stack(eff_lists, dim=0)  # (L, M)
            eff_med = torch.median(eff_stack, dim=0).values
            eff_distribution = torch.clamp(eff_med / torch.max(eff_med), min=1e-6)
            # plot efficiency distribution if plot_eff_distribution is on
            if plot_eff_distribution == 1:                    
                plt.plot(eff_distribution.detach().cpu().numpy())
                plt.title('efficiency distribution')
                plt.ylim((0,1))
                plt.show()
        
fids = []; crss = []; effs = []
for l in range(L):
    # 统一评估管线：从输入完整传播到输出
    kz_l = kz_torch_list[l]
    scl = lambda_c / lambda_list[l]
    modes = propagate_HK(Speckle_basis_torch[l], kz_l, d_in)
    for pl in range(Planes-1):
        modes = modes * torch.exp(1j*(Masks[pl]*scl))
        modes = propagate_HK(modes, kz_l, d)
    modes = modes * torch.exp(1j*(Masks[Planes-1]*scl))
    eout = propagate_HK(modes, kz_l, d_out)
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

