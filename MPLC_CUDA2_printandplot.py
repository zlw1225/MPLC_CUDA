"""
MPLC 相位掩膜后处理脚本
=======================
功能：
1. 加载训练好的相位掩膜
2. 计算光学性能指标（插入损耗、串扰、保真度等）
3. 对比完整尺寸 vs 裁剪尺寸的性能差异
4. 生成可视化图像

用法示例：
  # 只评估完整尺寸
  python MPLC_CUDA2_printandplot.py --nx_crop 0 --output_folder results
  
  # 对比完整尺寸和裁剪尺寸（512×256）
  python MPLC_CUDA2_printandplot.py --nx_crop 256 --ny_crop 512 --output_folder results
  
  # 应用量化（5-bit = 32级）
  python MPLC_CUDA2_printandplot.py --nx_crop 0 --quantize_pow 5 --output_folder results
"""

import math
import torch
import numpy as np
import os
import json
import argparse
from utils import propagate_HK, performance_loc_fidelity, performance_crosstalk, performance_efficiency

# ============================================================
# 第1步：解析命令行参数
# ============================================================
parser = argparse.ArgumentParser(description='MPLC 后处理工具：评估相位掩膜性能')
parser.add_argument('--nx_crop', type=int, default=256, 
                    help='裁剪宽度（像素）。设为0则禁用裁剪，只评估完整尺寸')
parser.add_argument('--ny_crop', type=int, default=512, 
                    help='裁剪高度（像素）')
parser.add_argument('--output_folder', type=str, default='results', 
                    help='结果文件夹路径')
parser.add_argument('--quantize_pow', type=int, default=0, 
                    help='相位量化位数（2^pow 级）。0表示不量化')
parser.add_argument('--masks_file', type=str, default='masks_full.pt', 
                    help='要加载的掩膜文件名')
args = parser.parse_args()

# 提取参数
Nx_crop = args.nx_crop              # 裁剪宽度
Ny_crop = args.ny_crop              # 裁剪高度
output_folder = args.output_folder  # 输出目录
quantize_pow = max(0, args.quantize_pow)  # 量化级数（非负）
masks_file = args.masks_file        # 掩膜文件名
crop_enabled = (Nx_crop > 0 and Ny_crop > 0)  # 是否启用裁剪功能

# 确保输出目录存在
if not os.path.isdir(output_folder):
    os.makedirs(output_folder, exist_ok=True)

# ============================================================
# 第2步：定义工具函数
# ============================================================

def quantize_masks(tensor, pow_level):
    """
    量化相位掩膜到 2^pow_level 个离散级别
    
    参数:
        tensor: 相位张量（弧度）
        pow_level: 量化位数（例如 5 表示 32 级）
    
    返回:
        量化后的相位张量
    
    示例:
        pow_level=5 → 32 级，步长 = 2π/32 ≈ 11.25°
    """
    if pow_level <= 0:
        return tensor  # 不量化，直接返回
    
    levels = 1 << pow_level  # 2^pow_level (位运算，等同于 2**pow_level)
    step = (2 * math.pi) / float(levels)  # 量化步长
    
    # 将相位包裹到 [-π, π] 范围内
    wrapped = torch.remainder(tensor + math.pi, 2 * math.pi) - math.pi
    
    # 四舍五入到最近的量化级别
    return torch.round(wrapped / step) * step

# ============================================================
# 第3步：加载或初始化必要的数据和参数
# ============================================================
# 说明：脚本可以在两种模式下运行：
#   1. 独立模式：从磁盘加载所有数据
#   2. 交互模式：从内存中获取训练时的变量（例如 Jupyter notebook）

g = globals()  # 获取当前全局变量字典

# 定义必需的变量列表
required_vars = [
    'Masks',                # 相位掩膜张量
    'LP_basis_torch',       # 输入模式（LP模式）
    'phi',                  # 目标输出模式（高斯模式）
    'Gaussian_Masks_torch', # 高斯掩膜（用于性能评估）
    'lambda_list',          # 波长列表
    'lambda_c',             # 中心波长
    'Planes',               # 相位平面数量
    'n_of_modes',           # 模式数量
    'pixelSize',            # 像素尺寸（米）
    'd_in',                 # 输入传播距离
    'd',                    # 平面间距离
    'd_out',                # 输出传播距离
    'DEVICE'                # 计算设备（CPU/GPU）
]

# 检查哪些变量缺失
missing = [v for v in required_vars if v not in g]

if missing:
    # 需要从磁盘加载数据
    # 3.1 设置计算设备
    if 'DEVICE' not in g:
        DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    # 3.2 加载配置文件（run_meta.json）
    try:
        meta_path = os.path.join(output_folder, 'run_meta.json')
        with open(meta_path, 'r', encoding='utf-8') as f:
            meta = json.load(f)
            cfg_loaded = meta.get('cfg', {})
            
            # 提取物理参数（带默认值）
            pixelSize = cfg_loaded.get('pixelSize', 8e-6)      # 像素尺寸 [m]
            Planes = cfg_loaded.get('Planes', 9)               # 相位平面数
            n_of_modes = cfg_loaded.get('n_of_modes', 10)      # 模式数
            d_in = cfg_loaded.get('d_in', 20e-3)               # 输入距离 [m]
            d = cfg_loaded.get('d', 2 * 9.7e-3)                # 平面间距 [m]
            d_out = cfg_loaded.get('d_out', 15e-3)             # 输出距离 [m]
            
    except Exception as e:
        print(f'[Warning] 无法加载配置文件: {e}，使用默认值')
        pass
    
    # 3.3 设置波长参数
    if 'lambda_list' not in g:
        # 6 个测试波长（微米 → 米）
        lambda_list = np.array([1.53e-6, 1.55e-6, 1.57e-6, 1.59e-6, 1.61e-6, 1.625e-6], dtype=np.float64)
    if 'lambda_c' not in g:
        lambda_c = 1.57e-6  # 中心波长 [m]
    
    # 3.4 加载相位掩膜
    if 'Masks' not in g:
        mask_path = os.path.join(output_folder, masks_file)
        Masks = torch.load(mask_path, map_location=DEVICE, weights_only=True)
        print(f'[Info] 加载相位掩膜: {mask_path}，形状: {Masks.shape}')
    # 3.5 加载输入/输出模式数据
    Ny_full, Nx_full = Masks.shape[-2], Masks.shape[-1]  # 掩膜的完整尺寸
    
    if 'LP_basis_torch' not in g or 'phi' not in g or 'Gaussian_Masks_torch' not in g:
        
        # 从 .npz 文件加载（这些文件由数据生成脚本创建）
        lp_data = np.load('lp_out_140.npz')      # LP 模式数据
        gauss_data = np.load('gauss_10x1_70.npz')  # 高斯模式数据
        
        # 提取前 n_of_modes 个模式
        lp_modes = lp_data['profiles'][:, 0:n_of_modes]           # 输入 LP 模式
        gauss_modes = gauss_data['Gaussian_basis'][:, 0:n_of_modes]  # 目标高斯模式
        gauss_masks = gauss_data['Gaussian_Masks'][:, 0:n_of_modes]  # 高斯掩膜
        
        # 转换为 PyTorch 张量并移至计算设备
        LP_basis_torch = torch.from_numpy(lp_modes.astype(np.complex64)).to(DEVICE)
        phi = torch.from_numpy(gauss_modes.astype(np.complex64)).to(DEVICE)
        Gaussian_Masks_torch = torch.from_numpy(gauss_masks.astype(np.float32)).to(DEVICE)
        
        data_size = lp_modes.shape[-1]  # 原始数据的尺寸（通常是 140×140）
        
        # 3.6 调整模式尺寸以匹配掩膜
        # 说明：训练时掩膜和模式可能是不同尺寸，需要统一
        
        def pad_complex(t, pad_tuple):
            """填充复数张量（分别处理实部和虚部）"""
            r = torch.nn.functional.pad(t.real, pad_tuple)
            im = torch.nn.functional.pad(t.imag, pad_tuple)
            return torch.complex(r, im)
        
        if (Nx_full > data_size) or (Ny_full > data_size):
            # 情况1：掩膜比数据大 → 需要填充数据
            pad_x = int((Nx_full - data_size) / 2)  # 左右填充量
            pad_y = int((Ny_full - data_size) / 2)  # 上下填充量
            pad_tuple = (pad_x, Nx_full - data_size - pad_x, 
                        pad_y, Ny_full - data_size - pad_y)
            
            LP_basis_torch = pad_complex(LP_basis_torch, pad_tuple)
            phi = pad_complex(phi, pad_tuple)
            Gaussian_Masks_torch = torch.nn.functional.pad(Gaussian_Masks_torch, pad_tuple)
            print(f'[Info] 输入输出填充模式: {data_size}×{data_size} → {Ny_full}×{Nx_full}')
        
        elif (Nx_full < data_size) or (Ny_full < data_size):
            # 情况2：掩膜比数据小 → 需要裁剪数据（中心裁剪）
            crop_x = int((data_size - Nx_full) / 2)  # 左侧裁剪量
            crop_y = int((data_size - Ny_full) / 2)  # 上侧裁剪量
            
            LP_basis_torch = LP_basis_torch[:, :, crop_y:crop_y+Ny_full, crop_x:crop_x+Nx_full]
            phi = phi[:, :, crop_y:crop_y+Ny_full, crop_x:crop_x+Nx_full]
            Gaussian_Masks_torch = Gaussian_Masks_torch[:, :, crop_y:crop_y+Ny_full, crop_x:crop_x+Nx_full]
            print(f'[Info] 输入输出裁剪模式: {data_size}×{data_size} → {Ny_full}×{Nx_full}')
        
        else:
            # 情况3：尺寸已匹配
            print(f'[Info] 模式尺寸已匹配: {Ny_full}×{Nx_full}')


if 'Masks' not in globals():
    raise RuntimeError('Masks tensor is required but could not be located.')

Masks = quantize_masks(globals()['Masks'], quantize_pow)
globals()['Masks'] = Masks
if quantize_pow > 0:
    unique_levels = torch.unique(torch.remainder(Masks + math.pi, 2 * math.pi)).numel()
    levels_count = 1 << quantize_pow
    print(f'[Quantize] Applied {levels_count}-level phase quantization (pow={quantize_pow}); unique levels observed: {unique_levels}.')


"""后处理：输出 full 与裁剪版本指标 + Overview 图。
约定：训练脚本已保存 results/masks_full.pt (full 尺寸)。
这里按用户需求：
  - baseline 使用内存中的 Masks (full)
  - crop 尺寸可通过命令行参数 --nx_crop 和 --ny_crop 控制 (默认 256x512 中心裁剪)
  - 打印 IL/MDL/XTs/fidelity/crosstalk/efficiency，两组以及 ΔIL / ΔXTs
  - 生成 overview_full.png 与 overview_cropped.png
"""

def central_crop(tensor, crop_h, crop_w):
    H = tensor.shape[-2]; W = tensor.shape[-1]
    top = (H - crop_h)//2; left = (W - crop_w)//2
    return tensor[..., top:top+crop_h, left:left+crop_w]

def build_kz(Ny_loc, Nx_loc, wavelength_val):
    nx_l = torch.linspace(-(Nx_loc-1)/2, (Nx_loc-1)/2, steps=Nx_loc, device=DEVICE, dtype=torch.float32)
    ny_l = torch.linspace(-(Ny_loc-1)/2, (Ny_loc-1)/2, steps=Ny_loc, device=DEVICE, dtype=torch.float32)
    kx1d_l = (2*math.pi) * nx_l / (Nx_loc * pixelSize)
    ky1d_l = (2*math.pi) * ny_l / (Ny_loc * pixelSize)
    kx_l, ky_l = torch.meshgrid(kx1d_l, ky1d_l, indexing='xy')
    kperp2 = kx_l**2 + ky_l**2
    k0 = (2*math.pi) / float(wavelength_val)
    return torch.sqrt((k0**2 - kperp2).to(torch.complex64))

def eval_metrics(masks_ph, LP_basis_ref, phi_ref, masks_target):
    Nl = len(lambda_list)
    ILs = np.zeros(Nl); MDLs = np.zeros(Nl); XTs_avg = np.zeros(Nl)
    fid_arr = np.zeros(Nl); crs_arr = np.zeros(Nl); eff_arr = np.zeros(Nl)
    for l in range(Nl):
        kz_loc = build_kz(LP_basis_ref.shape[-2], LP_basis_ref.shape[-1], lambda_list[l])
        scl = lambda_c / lambda_list[l]
        modes = propagate_HK(LP_basis_ref[l], kz_loc, d_in)
        for pl in range(Planes-1):
            modes = modes * torch.exp(1j * (masks_ph[pl] * scl))
            modes = propagate_HK(modes, kz_loc, d)
        modes = modes * torch.exp(1j * (masks_ph[Planes-1] * scl))
        eout = propagate_HK(modes, kz_loc, d_out)
        # 耦合矩阵
        E = eout.reshape(n_of_modes, -1)
        P = phi_ref[l].reshape(n_of_modes, -1)
        num = E @ torch.conj(P).T
        normE = torch.sum(torch.abs(E)**2, dim=1)
        normP = torch.sum(torch.abs(P)**2, dim=1)
        denom = torch.sqrt(normE[:, None] * normP[None, :]) + 1e-12
        C = num / denom
        s = np.linalg.svd(C.detach().cpu().numpy(), compute_uv=False)
        s2 = s**2
        ILs[l] = 10 * np.log10(np.mean(s2))
        MDLs[l] = 10 * np.log10(np.max(s2) / (np.min(s2)+1e-15))
        C2 = np.abs(C.detach().cpu().numpy())**2
        totalPower = np.sum(C2, axis=1)
        signalPower = np.clip(np.diag(C2), 1e-15, None)
        XTs_avg[l] = 10 * np.log10(np.mean((totalPower - signalPower)/signalPower))
        # 原函数指标
        eout_int = (torch.abs(eout))**2
        fid_l, _ = performance_loc_fidelity(eout, masks_target[l], phi_ref[l])
        crs_l, _, _ = performance_crosstalk(eout_int, masks_target[l])
        eff_l, _ = performance_efficiency(eout_int, masks_target[l])
        fid_arr[l] = float(fid_l.detach().cpu().numpy())
        crs_arr[l] = float(crs_l.detach().cpu().numpy())
        eff_arr[l] = float(eff_l.detach().cpu().numpy())
    return {
        'IL': ILs, 'MDL': MDLs, 'XTs': XTs_avg,
        'fid': fid_arr, 'crs': crs_arr, 'eff': eff_arr
    }

def make_overview(masks_ph, LP_basis_ref, phi_ref, fname):
    with torch.no_grad():
        l_idx = int(np.argmin(np.abs(lambda_list - lambda_c)))
        kz_l = build_kz(LP_basis_ref.shape[-2], LP_basis_ref.shape[-1], lambda_list[l_idx])
        scl = lambda_c / lambda_list[l_idx]
        # forward snapshots
        fwd_titles=[]; fwd_maps=[]
        modes = LP_basis_ref[l_idx].clone()
        fwd_maps.append(torch.sum(torch.abs(modes)**2, dim=0)); fwd_titles.append('z=0')
        modes = propagate_HK(modes, kz_l, d_in)
        fwd_maps.append(torch.sum(torch.abs(modes)**2, dim=0)); fwd_titles.append('p0 pre')
        for pl in range(Planes-1):
            modes = modes * torch.exp(1j*(masks_ph[pl]*scl))
            modes = propagate_HK(modes, kz_l, d)
            fwd_maps.append(torch.sum(torch.abs(modes)**2, dim=0)); fwd_titles.append(f'p{pl+1} pre')
        modes = modes * torch.exp(1j*(masks_ph[Planes-1]*scl))
        eout = propagate_HK(modes, kz_l, d_out)
        fwd_maps.append(torch.sum(torch.abs(eout)**2, dim=0)); fwd_titles.append('z_out')
        # backward snapshots
        bwd_maps=[]; modes_b = phi_ref[l_idx].clone()
        bwd_maps.append(torch.sum(torch.abs(modes_b)**2, dim=0))
        modes_b = propagate_HK(modes_b, kz_l, -d_out)
        modes_b = modes_b * torch.conj(torch.exp(1j*(masks_ph[Planes-1]*scl)))
        bwd_maps.append(torch.sum(torch.abs(modes_b)**2, dim=0))
        for pl in range(Planes-2, -1, -1):
            modes_b = propagate_HK(modes_b, kz_l, -d)
            modes_b = modes_b * torch.conj(torch.exp(1j*(masks_ph[pl]*scl)))
            bwd_maps.append(torch.sum(torch.abs(modes_b)**2, dim=0))
        modes_b = propagate_HK(modes_b, kz_l, -d_in)
        bwd_maps.append(torch.sum(torch.abs(modes_b)**2, dim=0))
        # plot
        import matplotlib.pyplot as plt
        ncols_ovr = len(fwd_maps)
        fig, axes = plt.subplots(3, ncols_ovr, figsize=(3*ncols_ovr, 9))
        for c in range(ncols_ovr):
            ax = axes[0,c]; ax.imshow(fwd_maps[c].detach().cpu().numpy(), cmap='viridis', origin='lower'); ax.set_title(fwd_titles[c]); ax.axis('off')
        bwd_aligned = list(reversed(bwd_maps))
        for c in range(min(ncols_ovr, len(bwd_aligned))):
            ax = axes[1,c]; ax.imshow(bwd_aligned[c].detach().cpu().numpy(), cmap='viridis', origin='lower'); ax.axis('off')
        start = 1 if ncols_ovr >=2 else 0
        for p in range(Planes):
            c = start + p
            if c < ncols_ovr:
                ax = axes[2,c]; ax.imshow(masks_ph[p].detach().cpu().numpy(), cmap='RdBu_r', origin='lower', vmin=-math.pi, vmax=math.pi); ax.axis('off')
        fig.suptitle(fname)
        fig.tight_layout()
        fig.savefig(os.path.join(output_folder, f'{fname}.png'), dpi=150)
        plt.close(fig)

# ============================================================
# 第6步：定义详细可视化函数（独立于裁剪逻辑）
# ============================================================

def generate_detailed_plots(masks_ph, LP_basis_ref, phi_ref, Gmask_ref, prefix=''):
    """
    生成详细的分析图像
    
    参数:
        masks_ph: 相位掩膜张量
        LP_basis_ref: 输入模式
        phi_ref: 目标输出模式
        Gmask_ref: 高斯掩膜
        prefix: 文件名前缀（例如 'full_' 或 'cropped_'）
    """
    import matplotlib.pyplot as plt
    
    # 1) 各平面相位图 (masks_phase_maps.png)
    ncols_phase = 4
    nrows_phase = math.ceil(Planes / ncols_phase)
    fig_phase_maps, axes_phase_maps = plt.subplots(nrows_phase, ncols_phase, 
                                                     figsize=(3*ncols_phase, 3*nrows_phase))
    axes_phase_flat = np.array(axes_phase_maps).ravel() if isinstance(axes_phase_maps, np.ndarray) else np.array([axes_phase_maps])
    
    for p in range(Planes):
        ax = axes_phase_flat[p]
        ax.imshow(masks_ph[p].detach().cpu().numpy(), cmap='twilight', 
                 origin='lower', vmin=-math.pi, vmax=math.pi)
        ax.set_title(f'Plane {p}')
        ax.axis('off')
    for k in range(Planes, nrows_phase*ncols_phase):
        axes_phase_flat[k].axis('off')
    
    fig_phase_maps.suptitle(f'{prefix}Phase Masks (radians)')
    fig_phase_maps.tight_layout()
    fig_phase_maps.savefig(os.path.join(output_folder, f'{prefix}masks_phase_maps.png'), dpi=150)
    plt.close(fig_phase_maps)
    
    # 2) 六个波长耦合矩阵 (coupling_matrices_6wls.png)
    with torch.no_grad():
        Nl = len(lambda_list)
        ncols_cpl = 3
        nrows_cpl = math.ceil(Nl / ncols_cpl)
        fig_cpl, axes_cpl = plt.subplots(nrows_cpl, ncols_cpl, 
                                         figsize=(4*ncols_cpl, 3.5*nrows_cpl))
        axes_cpl_flat = np.array(axes_cpl).ravel() if isinstance(axes_cpl, np.ndarray) else np.array([axes_cpl])
        
        for idx in range(nrows_cpl * ncols_cpl):
            ax = axes_cpl_flat[idx]
            if idx >= Nl:
                ax.axis('off')
                continue
            
            l = idx
            kz_l = build_kz(LP_basis_ref.shape[-2], LP_basis_ref.shape[-1], lambda_list[l])
            scl_l = lambda_c / lambda_list[l]
            modes = propagate_HK(LP_basis_ref[l], kz_l, d_in)
            
            for pl in range(Planes-1):
                modes = modes * torch.exp(1j * (masks_ph[pl] * scl_l))
                modes = propagate_HK(modes, kz_l, d)
            modes = modes * torch.exp(1j * (masks_ph[Planes-1] * scl_l))
            eout = propagate_HK(modes, kz_l, d_out)
            
            E = eout.reshape(n_of_modes, -1)
            P = phi_ref[l].reshape(n_of_modes, -1)
            num = E @ torch.conj(P).T
            normE = torch.sum(torch.abs(E)**2, dim=1)
            normP = torch.sum(torch.abs(P)**2, dim=1)
            denom = torch.sqrt(normE[:, None] * normP[None, :]) + 1e-12
            C = num / denom
            C2_plot = np.flip(np.abs(C.detach().cpu().numpy())**2, axis=1)
            
            im = ax.imshow(C2_plot, cmap='magma', origin='lower', aspect='equal', vmin=0.0, vmax=1.0)
            ax.set_title(f'λ={lambda_list[l]*1e6:.3f} μm')
            ax.axis('off')
            if idx == 0:
                fig_cpl.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
        
        for k in range(Nl, nrows_cpl*ncols_cpl):
            axes_cpl_flat[k].axis('off')
        
        fig_cpl.suptitle(f'{prefix}Coupling Matrices |C|²')
        fig_cpl.tight_layout()
        fig_cpl.savefig(os.path.join(output_folder, f'{prefix}coupling_matrices_6wls.png'), dpi=150)
        plt.close(fig_cpl)
    
    # 3) & 4) λ=1.57 μm 反向传播到 z=0 的每模式强度与相位
    with torch.no_grad():
        l_idx = int(np.argmin(np.abs(lambda_list - lambda_c)))
        kz_l = build_kz(LP_basis_ref.shape[-2], LP_basis_ref.shape[-1], lambda_list[l_idx])
        scl_l = lambda_c / lambda_list[l_idx]
        modes_b = phi_ref[l_idx].clone()
        modes_b = propagate_HK(modes_b, kz_l, -d_out)
        modes_b = modes_b * torch.conj(torch.exp(1j * (masks_ph[Planes-1] * scl_l)))
        
        for pl in range(Planes-2, -1, -1):
            modes_b = propagate_HK(modes_b, kz_l, -d)
            modes_b = modes_b * torch.conj(torch.exp(1j * (masks_ph[pl] * scl_l)))
        modes_b = propagate_HK(modes_b, kz_l, -d_in)
        
        Mloc = min(modes_b.shape[0], n_of_modes)
        ncols_bw = min(5, Mloc) if Mloc>0 else 1
        nrows_bw = math.ceil(Mloc / ncols_bw) if Mloc>0 else 1
        
        # 强度图
        fig_bw_I, axes_bw_I = plt.subplots(nrows_bw, ncols_bw, figsize=(3*ncols_bw, 3*nrows_bw))
        axes_bw_I_flat = np.array(axes_bw_I).ravel() if isinstance(axes_bw_I, np.ndarray) else np.array([axes_bw_I])
        vmax_I = max([float(torch.max(torch.abs(modes_b[j])**2).detach().cpu().numpy()) for j in range(Mloc)]) if Mloc>0 else 1.0
        
        for j in range(Mloc):
            inten = torch.abs(modes_b[j])**2
            axes_bw_I_flat[j].imshow(inten.detach().cpu().numpy(), cmap='inferno', 
                                     origin='lower', vmin=0.0, vmax=vmax_I)
            axes_bw_I_flat[j].set_title(f'mode {j}')
            axes_bw_I_flat[j].axis('off')
        for k in range(Mloc, nrows_bw*ncols_bw):
            axes_bw_I_flat[k].axis('off')
        
        fig_bw_I.suptitle(f'{prefix}Backward z=0 Intensity (λ=1.57 μm)')
        fig_bw_I.tight_layout()
        fig_bw_I.savefig(os.path.join(output_folder, f'{prefix}backward_z0_modes_1p57.png'), dpi=150)
        plt.close(fig_bw_I)
        
        # 相位图
        fig_bw_P, axes_bw_P = plt.subplots(nrows_bw, ncols_bw, figsize=(3*ncols_bw, 3*nrows_bw))
        axes_bw_P_flat = np.array(axes_bw_P).ravel() if isinstance(axes_bw_P, np.ndarray) else np.array([axes_bw_P])
        
        for j in range(Mloc):
            ph = torch.angle(modes_b[j])
            axes_bw_P_flat[j].imshow(ph.detach().cpu().numpy(), cmap='twilight', 
                                     origin='lower', vmin=-math.pi, vmax=math.pi)
            axes_bw_P_flat[j].set_title(f'mode {j} phase')
            axes_bw_P_flat[j].axis('off')
        for k in range(Mloc, nrows_bw*ncols_bw):
            axes_bw_P_flat[k].axis('off')
        
        fig_bw_P.suptitle(f'{prefix}Backward z=0 Phase (λ=1.57 μm)')
        fig_bw_P.tight_layout()
        fig_bw_P.savefig(os.path.join(output_folder, f'{prefix}backward_z0_modes_phase_1p57.png'), dpi=150)
        plt.close(fig_bw_P)

# ============================================================
# 第7步：评估完整尺寸性能
# ============================================================

metrics_full = eval_metrics(Masks, LP_basis_torch, phi, Gaussian_Masks_torch)
make_overview(Masks, LP_basis_torch, phi, 'overview_full')

# 生成完整尺寸的详细图像
generate_detailed_plots(Masks, LP_basis_torch, phi, Gaussian_Masks_torch, prefix='full_')

# 打印性能指标
Ny_full, Nx_full = Masks.shape[-2], Masks.shape[-1]
wl_list = [f'{wl*1e6:.3f}' for wl in lambda_list]
def _fmt(arr):
    return [f'{v:.3f}' for v in arr]

print('Wavelengths (μm):', wl_list)
print('[Full]    IL (dB):', _fmt(metrics_full['IL']))
print('[Full]    XTs (dB):', _fmt(metrics_full['XTs']))
print('[Full]    MDL (dB):', _fmt(metrics_full['MDL']))
print('[Full]    fidelity :', _fmt(metrics_full['fid']))
print('[Full]    crosstalk:', _fmt(metrics_full['crs']))
print('[Full]    efficiency:', _fmt(metrics_full['eff']))

# ============================================================
# 第8步：评估裁剪尺寸性能（如果启用）
# ============================================================
if crop_enabled and Ny_full >= Ny_crop and Nx_full >= Nx_crop:
    print('\n' + '='*60)
    print(f'开始评估裁剪尺寸性能 ({Ny_crop}×{Nx_crop})')
    print('='*60)
    
    # 裁剪所有相关张量
    Masks_crop = central_crop(Masks, Ny_crop, Nx_crop)
    LP_crop = central_crop(LP_basis_torch, Ny_crop, Nx_crop)
    phi_crop = central_crop(phi, Ny_crop, Nx_crop)
    Gmask_crop = central_crop(Gaussian_Masks_torch, Ny_crop, Nx_crop)
    
    # 评估性能
    metrics_crop = eval_metrics(Masks_crop, LP_crop, phi_crop, Gmask_crop)
    make_overview(Masks_crop, LP_crop, phi_crop, 'overview_cropped')
    
    # 生成裁剪尺寸的详细图像
    generate_detailed_plots(Masks_crop, LP_crop, phi_crop, Gmask_crop, prefix='cropped_')
    
    # 打印性能指标
    print('\n' + '-'*60)
    print('裁剪尺寸性能指标')
    print('-'*60)
    print('[Cropped] IL (dB):', _fmt(metrics_crop['IL']))
    print('[Cropped] XTs (dB):', _fmt(metrics_crop['XTs']))
    print('[Cropped] MDL (dB):', _fmt(metrics_crop['MDL']))
    print('[Cropped] fidelity :', _fmt(metrics_crop['fid']))
    print('[Cropped] crosstalk:', _fmt(metrics_crop['crs']))
    print('[Cropped] efficiency:', _fmt(metrics_crop['eff']))
    
    # 计算并打印差值（Cropped - Full）
    print('\n' + '-'*60)
    print('性能差异分析 (Cropped - Full)')
    print('-'*60)
    dIL = metrics_crop['IL'] - metrics_full['IL']
    dXT = metrics_crop['XTs'] - metrics_full['XTs']
    print('ΔIL (crop-full)  (dB):', _fmt(dIL))
    print('ΔXTs (crop-full) (dB):', _fmt(dXT))
    mean_dIL = np.mean(dIL)
    mean_dXT = np.mean(dXT)
    print(f'Mean ΔIL={mean_dIL:.3f} dB, Mean ΔXTs={mean_dXT:.3f} dB')
    
    # 判断裁剪影响
    if abs(mean_dIL) < 0.1 and abs(mean_dXT) < 0.2:
        print('✅ 裁剪影响可忽略 (ΔIL<0.1dB & ΔXTs<0.2dB)')
    else:
        print('⚠️  裁剪有显著影响，需要评估是否可接受')
    
    # 保存裁剪相位
    torch.save(Masks_crop.detach().cpu(), os.path.join(output_folder, 'masks_cropped.pt'))
    print(f'[Save] 裁剪后的相位掩膜已保存到 {output_folder}/masks_cropped.pt')

elif crop_enabled:
    print(f'\n⚠️  警告：裁剪尺寸 {Ny_crop}×{Nx_crop} 大于完整尺寸 {Ny_full}×{Nx_full}，跳过裁剪评估')
else:
    print(f'\n[Info] 掩膜裁剪功能已禁用 (nx_crop={Nx_crop}, ny_crop={Ny_crop})')

