
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
lp_data = np.load('modes_lp_10.npz')
lp_modes = lp_data['profiles']  # 形状: (L, 10, 512, 512)
gauss_data = np.load('gauss_5x2_custom.npz')
gauss_modes = gauss_data['profiles']  # 形状: (L, 10, 512, 512)

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
Gaussian_Masks = np.zeros_like(Gaussian_basis, dtype=np.float32)
yy, xx = np.indices((Ny, Nx))
# 模板：第一个 λ、第一个模式
template = np.abs(Gaussian_basis[0, 0])**2
idx_max = int(np.argmax(template)) if template.size > 0 else 0
cy, cx = np.unravel_index(idx_max, template.shape) if template.size > 0 else (Ny//2, Nx//2)
I0 = float(template[cy, cx]) if template.size > 0 else 0.0
if I0 > 0:
    Ith = I0 * np.exp(-2.0)
    r_map = np.sqrt((yy - cy)**2 + (xx - cx)**2)
    # 限制搜索半径，避免阵列中其他峰影响估计
    r_cap = 0.2 * float(min(Ny, Nx))
    region = (template >= Ith) & (r_map <= r_cap)
    if np.any(region):
        w_est = float(r_map[region].max())
    else:
        w_est = 20.0
else:
    w_est = 20.0
r_mask = w_est + 7.0
# 复制到所有 λ 与模式，但中心对齐各自模式的峰值位置
for l in range(L):
    for m in range(n_of_modes):
        inten_lm = np.abs(Gaussian_basis[l, m])**2
        idx_max_lm = int(np.argmax(inten_lm)) if inten_lm.size > 0 else 0
        cy_m, cx_m = np.unravel_index(idx_max_lm, inten_lm.shape) if inten_lm.size > 0 else (Ny//2, Nx//2)
        r_map_m = np.sqrt((yy - cy_m)**2 + (xx - cx_m)**2)
        Gaussian_Masks[l, m] = (r_map_m <= r_mask).astype(np.float32)

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

