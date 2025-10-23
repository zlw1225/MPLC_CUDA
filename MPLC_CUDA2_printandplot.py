import math, torch, numpy as np, os, json, argparse
from utils import propagate_HK, performance_loc_fidelity, performance_crosstalk, performance_efficiency

# 解析命令行参数
parser = argparse.ArgumentParser(description='MPLC post-processing with configurable crop size')
parser.add_argument('--nx_crop', type=int, default=256, help='Crop width (default: 256)')
parser.add_argument('--ny_crop', type=int, default=512, help='Crop height (default: 512)')
parser.add_argument('--output_folder', type=str, default='results', help='Output folder for results (default: results)')
parser.add_argument('--quantize_pow', type=int, default=0, help='Quantize phase masks to 2^pow levels (0 disables quantization)')
parser.add_argument('--masks_file', type=str, default='masks_full.pt', help='Mask file name to load (default: masks_full.pt)')
args = parser.parse_args()

Nx_crop = args.nx_crop
Ny_crop = args.ny_crop
output_folder = args.output_folder
quantize_pow = max(0, args.quantize_pow)
masks_file = args.masks_file
crop_enabled = (Nx_crop > 0 and Ny_crop > 0)

if not os.path.isdir(output_folder):
    os.makedirs(output_folder, exist_ok=True)

def quantize_masks(tensor, pow_level):
    if pow_level <= 0:
        return tensor
    levels = 1 << pow_level
    step = (2 * math.pi) / float(levels)
    wrapped = torch.remainder(tensor + math.pi, 2 * math.pi) - math.pi
    return torch.round(wrapped / step) * step

# 尝试从当前全局命名空间获取训练阶段对象；若不存在则从磁盘加载必需内容
g = globals()
required_vars = ['Masks','LP_basis_torch','phi','Gaussian_Masks_torch','lambda_list','lambda_c','Planes','n_of_modes','pixelSize','d_in','d','d_out','DEVICE']
missing = [v for v in required_vars if v not in g]
if missing:
    # 最低限：加载 run_meta 以获取配置
    if 'DEVICE' not in g:
        DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    try:
        with open(os.path.join(output_folder, 'run_meta.json'), 'r', encoding='utf-8') as f:
            meta = json.load(f)
            cfg_loaded = meta.get('cfg', {})
            pixelSize = cfg_loaded.get('pixelSize', 8e-6)
            Planes = cfg_loaded.get('Planes', 9)
            n_of_modes = cfg_loaded.get('n_of_modes', 10)
            d_in = cfg_loaded.get('d_in', 20e-3)
            d = cfg_loaded.get('d', 2 * 9.7e-3)
            d_out = cfg_loaded.get('d_out', 15e-3)
    except Exception:
        pass
    # 基础波长数组（若缺失）
    if 'lambda_list' not in g:
        lambda_list = np.array([1.53e-6,1.55e-6,1.57e-6,1.59e-6,1.61e-6,1.625e-6],dtype=np.float64)
    if 'lambda_c' not in g:
        lambda_c = 1.57e-6
    # 加载相位
    if 'Masks' not in g:
        Masks = torch.load(os.path.join(output_folder, masks_file), map_location=DEVICE, weights_only=True)
    # 若模式与掩膜缺失：尝试加载原 .npz 并按训练脚本逻辑 pad 到当前 Masks 尺寸
    Ny_full, Nx_full = Masks.shape[-2], Masks.shape[-1]
    if 'LP_basis_torch' not in g or 'phi' not in g or 'Gaussian_Masks_torch' not in g:
        # 复用训练使用的数据文件名
        lp_data = np.load('lp_out_140.npz')
        gauss_data = np.load('gauss_10x1_70.npz')
        lp_modes = lp_data['profiles'][:,0:n_of_modes]
        gauss_modes = gauss_data['Gaussian_basis'][:,0:n_of_modes]
        gauss_masks = gauss_data['Gaussian_Masks'][:,0:n_of_modes]
        LP_basis_torch = torch.from_numpy(lp_modes.astype(np.complex64)).to(DEVICE)
        phi = torch.from_numpy(gauss_modes.astype(np.complex64)).to(DEVICE)
        Gaussian_Masks_torch = torch.from_numpy(gauss_masks.astype(np.float32)).to(DEVICE)
        data_size = lp_modes.shape[-1]
        if (Nx_full>data_size) or (Ny_full>data_size):
            pad_x = int((Nx_full-data_size)/2); pad_y = int((Ny_full-data_size)/2)
            pad_tuple = (pad_x, Nx_full-data_size-pad_x, pad_y, Ny_full-data_size-pad_y)
            def pad_complex(t):
                r = torch.nn.functional.pad(t.real, pad_tuple); im = torch.nn.functional.pad(t.imag, pad_tuple)
                return torch.complex(r,im)
            LP_basis_torch = pad_complex(LP_basis_torch)
            phi = pad_complex(phi)
            Gaussian_Masks_torch = torch.nn.functional.pad(Gaussian_Masks_torch, pad_tuple)


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

# ========== Full size metrics ==========
metrics_full = eval_metrics(Masks, LP_basis_torch, phi, Gaussian_Masks_torch)
make_overview(Masks, LP_basis_torch, phi, 'overview_full')

# ========== Cropped metrics ==========
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

if crop_enabled and Ny_full >= Ny_crop and Nx_full >= Nx_crop:
    Masks_crop = central_crop(Masks, Ny_crop, Nx_crop)
    LP_crop = central_crop(LP_basis_torch, Ny_crop, Nx_crop)
    phi_crop = central_crop(phi, Ny_crop, Nx_crop)
    Gmask_crop = central_crop(Gaussian_Masks_torch, Ny_crop, Nx_crop)
    metrics_crop = eval_metrics(Masks_crop, LP_crop, phi_crop, Gmask_crop)
    make_overview(Masks_crop, LP_crop, phi_crop, 'overview_cropped')
    # 打印
    print('[Cropped] IL (dB):', _fmt(metrics_crop['IL']))
    print('[Cropped] XTs (dB):', _fmt(metrics_crop['XTs']))
    print('[Cropped] MDL (dB):', _fmt(metrics_crop['MDL']))
    print('[Cropped] fidelity :', _fmt(metrics_crop['fid']))
    print('[Cropped] crosstalk:', _fmt(metrics_crop['crs']))
    print('[Cropped] efficiency:', _fmt(metrics_crop['eff']))
    # 差值（Cropped - Full）
    dIL = metrics_crop['IL'] - metrics_full['IL']
    dXT = metrics_crop['XTs'] - metrics_full['XTs']
    print('ΔIL (crop-full)  (dB):', _fmt(dIL))
    print('ΔXTs (crop-full) (dB):', _fmt(dXT))
    mean_dIL = np.mean(dIL); mean_dXT = np.mean(dXT)
    print(f'Mean ΔIL={mean_dIL:.3f} dB, Mean ΔXTs={mean_dXT:.3f} dB')
    if abs(mean_dIL) < 0.1 and abs(mean_dXT) < 0.2:
        print('[Info] Crop impact negligible (ΔIL<0.1dB & ΔXTs<0.2dB).')
    # 保存裁剪相位
    torch.save(Masks_crop.detach().cpu(), os.path.join(output_folder, 'masks_cropped.pt'))
    print(f'[Save] Cropped phase masks saved to {output_folder}/masks_cropped.pt')

    # ================== 仅对裁剪结果输出指定四类图像 ==================
    import matplotlib.pyplot as plt
    # 1) 裁剪后各平面相位图 (masks_phase_maps.png)
    ncols_phase = 4
    nrows_phase = math.ceil(Planes / ncols_phase)
    fig_phase_maps, axes_phase_maps = plt.subplots(nrows_phase, ncols_phase, figsize=(3*ncols_phase, 3*nrows_phase))
    axes_phase_flat = np.array(axes_phase_maps).ravel() if isinstance(axes_phase_maps, np.ndarray) else np.array([axes_phase_maps])
    for p in range(Planes):
        ax = axes_phase_flat[p]
        ax.imshow(Masks_crop[p].detach().cpu().numpy(), cmap='twilight', origin='lower', vmin=-math.pi, vmax=math.pi)
        ax.set_title(f'Mask p{p}')
        ax.axis('off')
    for k in range(Planes, nrows_phase*ncols_phase):
        axes_phase_flat[k].axis('off')
    fig_phase_maps.suptitle('Cropped phase masks (radians)')
    fig_phase_maps.tight_layout()
    fig_phase_maps.savefig(os.path.join(output_folder, 'masks_phase_maps.png'), dpi=150)
    plt.close(fig_phase_maps)

    # 2) 六个波长耦合矩阵 (coupling_matrices_6wls.png)
    with torch.no_grad():
        Nl = len(lambda_list)
        ncols_cpl = 3
        nrows_cpl = math.ceil(Nl / ncols_cpl)
        fig_cpl, axes_cpl = plt.subplots(nrows_cpl, ncols_cpl, figsize=(4*ncols_cpl, 3.5*nrows_cpl))
        axes_cpl_flat = np.array(axes_cpl).ravel() if isinstance(axes_cpl, np.ndarray) else np.array([axes_cpl])
        for idx in range(nrows_cpl * ncols_cpl):
            ax = axes_cpl_flat[idx]
            if idx >= Nl:
                ax.axis('off')
                continue
            l = idx
            kz_l = build_kz(LP_crop.shape[-2], LP_crop.shape[-1], lambda_list[l])
            scl_l = lambda_c / lambda_list[l]
            modes = propagate_HK(LP_crop[l], kz_l, d_in)
            for pl in range(Planes-1):
                modes = modes * torch.exp(1j * (Masks_crop[pl] * scl_l))
                modes = propagate_HK(modes, kz_l, d)
            modes = modes * torch.exp(1j * (Masks_crop[Planes-1] * scl_l))
            eout = propagate_HK(modes, kz_l, d_out)
            E = eout.reshape(n_of_modes, -1)
            P = phi_crop[l].reshape(n_of_modes, -1)
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
        fig_cpl.suptitle('Cropped coupling matrices |C|^2')
        fig_cpl.tight_layout()
        fig_cpl.savefig(os.path.join(output_folder, 'coupling_matrices_6wls.png'), dpi=150)
        plt.close(fig_cpl)

    # 3) & 4) λ=1.57 μm 反向传播到 z=0 的每模式强度与相位
    with torch.no_grad():
        l_idx = int(np.argmin(np.abs(lambda_list - lambda_c)))
        kz_l = build_kz(LP_crop.shape[-2], LP_crop.shape[-1], lambda_list[l_idx])
        scl_l = lambda_c / lambda_list[l_idx]
        modes_b = phi_crop[l_idx].clone()
        modes_b = propagate_HK(modes_b, kz_l, -d_out)
        modes_b = modes_b * torch.conj(torch.exp(1j * (Masks_crop[Planes-1] * scl_l)))
        for pl in range(Planes-2, -1, -1):
            modes_b = propagate_HK(modes_b, kz_l, -d)
            modes_b = modes_b * torch.conj(torch.exp(1j * (Masks_crop[pl] * scl_l)))
        modes_b = propagate_HK(modes_b, kz_l, -d_in)
        Mloc = min(modes_b.shape[0], n_of_modes)
        ncols_bw = min(5, Mloc) if Mloc>0 else 1
        nrows_bw = math.ceil(Mloc / ncols_bw) if Mloc>0 else 1
        # 强度
        fig_bw_I, axes_bw_I = plt.subplots(nrows_bw, ncols_bw, figsize=(3*ncols_bw, 3*nrows_bw))
        axes_bw_I_flat = np.array(axes_bw_I).ravel() if isinstance(axes_bw_I, np.ndarray) else np.array([axes_bw_I])
        vmax_I = max([float(torch.max(torch.abs(modes_b[j])**2).detach().cpu().numpy()) for j in range(Mloc)]) if Mloc>0 else 1.0
        for j in range(Mloc):
            inten = torch.abs(modes_b[j])**2
            axes_bw_I_flat[j].imshow(inten.detach().cpu().numpy(), cmap='inferno', origin='lower', vmin=0.0, vmax=vmax_I)
            axes_bw_I_flat[j].set_title(f'mode {j} @ z=0')
            axes_bw_I_flat[j].axis('off')
        for k in range(Mloc, nrows_bw*ncols_bw):
            axes_bw_I_flat[k].axis('off')
        fig_bw_I.suptitle('Cropped backward z0 intensity (λ=1.57 μm)')
        fig_bw_I.tight_layout()
        fig_bw_I.savefig(os.path.join(output_folder, 'backward_z0_modes_1p57.png'), dpi=150)
        plt.close(fig_bw_I)
        # 相位
        fig_bw_P, axes_bw_P = plt.subplots(nrows_bw, ncols_bw, figsize=(3*ncols_bw, 3*nrows_bw))
        axes_bw_P_flat = np.array(axes_bw_P).ravel() if isinstance(axes_bw_P, np.ndarray) else np.array([axes_bw_P])
        for j in range(Mloc):
            ph = torch.angle(modes_b[j])
            axes_bw_P_flat[j].imshow(ph.detach().cpu().numpy(), cmap='twilight', origin='lower', vmin=-math.pi, vmax=math.pi)
            axes_bw_P_flat[j].set_title(f'mode {j} phase @ z=0')
            axes_bw_P_flat[j].axis('off')
        for k in range(Mloc, nrows_bw*ncols_bw):
            axes_bw_P_flat[k].axis('off')
        fig_bw_P.suptitle('Cropped backward z0 phase (λ=1.57 μm)')
        fig_bw_P.tight_layout()
        fig_bw_P.savefig(os.path.join(output_folder, 'backward_z0_modes_phase_1p57.png'), dpi=150)
        plt.close(fig_bw_P)


elif crop_enabled:
    print(f'[Warn] Crop size {Ny_crop}x{Nx_crop} larger than full size {Ny_full}x{Nx_full}; skipping crop metrics.')
else:
    print(f'[Info] Crop metrics disabled (nx_crop={Nx_crop}, ny_crop={Ny_crop}).')

# 结束