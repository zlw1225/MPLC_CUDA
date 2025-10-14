

"""
MPLC_CUDA2 模块实现了基于 CUDA 加速的多平面光学转换 (MPLC) 优化流程。

主要流程概览：
- 解析配置与运行时常量，确定平面数量、频域网格以及 GPU 设备信息；
- 从预生成的 LP 与高斯模式数据中加载并调整输入/输出场分布，同时构建背景/交叉区域掩模；
- 通过多波长前向/反向传播初始化状态，并在主优化循环中迭代更新各平面相位掩模；
- 定期评估定位保真度、串扰和效率，执行效率均衡与平滑处理，最终保存最优相位掩模及指标报表。

在实现中，重要环节包括 `optimize_phase_masks` 中的内外循环传播、梯度近似与掩模更新逻辑，以及 `evaluate_performance` 用于多波长性能统计的流程。
"""

from __future__ import annotations

import argparse
import datetime
import json
import math
import os
import sys
from types import SimpleNamespace
from typing import List

import matplotlib
# matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn

from utils import (
    complim,
    performance_crosstalk,
    performance_efficiency,
    performance_loc_fidelity,
    plot_in_GS,
    propagate_HK,
)


def parse_cfg() -> SimpleNamespace:
    """解析命令行参数并返回包含默认值配置的命名空间。"""
    parser = argparse.ArgumentParser(add_help=True)
    parser.add_argument("--n_of_modes", type=int, default=10)
    parser.add_argument("--Planes", type=int, default=9)
    parser.add_argument("--iterations", type=int, default=500)
    parser.add_argument("--first_n_iterations", type=int, default=10)
    parser.add_argument("--Nx", type=int, default=512)
    parser.add_argument("--Ny", type=int, default=1024)
    parser.add_argument("--calc_perf_every_it", type=int, default=10)
    parser.add_argument("--equalize_efficiency", type=int, choices=[0, 1], default=1)
    parser.add_argument("--plot_eff_distribution", type=int, choices=[0, 1], default=0)
    parser.add_argument("--smoothing_switch", type=int, choices=[0, 1], default=1)
    parser.add_argument("--plot_results", type=int, choices=[0, 1], default=0)
    parser.add_argument("--preview_inputs", type=int, choices=[0, 1], default=1)
    parser.add_argument("--do_padded_eval", type=int, choices=[0, 1], default=0)
    parser.add_argument("--alpha", type=float, default=1.0)
    parser.add_argument("--beta", type=float, default=2.0)
    parser.add_argument("--gamma", type=float, default=1.0)
    parser.add_argument("--delta_theta_1", type=float, default=2 * math.pi / 255)
    parser.add_argument("--delta_theta_0", type=float, default=10 * (2 * math.pi / 255))
    parser.add_argument("--pixelSize", type=float, default=8e-6)
    parser.add_argument("--wavelength", type=float, default=1.57e-6)
    parser.add_argument("--d_in", type=float, default=20e-3)
    parser.add_argument("--d", type=float, default=2 * 9.7e-3)
    parser.add_argument("--d_out", type=float, default=15e-3)
    parser.add_argument("--OffsetMultiplier", type=float, default=1.0)

    if "ipykernel" in sys.modules or "__file__" not in globals():
        args = parser.parse_args([])
    else:
        args = parser.parse_args()

    return SimpleNamespace(**vars(args))


def select_device() -> torch.device:
    """选择 GPU 或 CPU 设备并打印硬件信息，优先使用可用的 CUDA。"""
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[MPLC2] Using device: {device}")
    if device.type == "cuda":
        name = torch.cuda.get_device_name(0)
        cap = torch.cuda.get_device_capability(0)
        print(
            f"[MPLC2] GPU: {name}, capability={cap}, torch_cuda={getattr(torch.version, 'cuda', None)}"
        )
    return device


def ensure_results_dir() -> str:
    """确保结果输出目录存在，若不存在则创建并返回路径。"""
    results_dir = os.path.join("results")
    os.makedirs(results_dir, exist_ok=True)
    return results_dir


def write_run_meta(cfg: SimpleNamespace, device: torch.device, results_dir: str) -> None:
    """将运行时间、环境信息与配置写入结果目录下的 `run_meta.json`。"""
    run_meta = {
        "timestamp": datetime.datetime.now().isoformat(),
        "device": str(device),
        "torch_version": getattr(torch, "__version__", None),
        "torch_cuda": getattr(torch.version, "cuda", None),
        "numpy_version": np.__version__,
        "cfg": {k: getattr(cfg, k) for k in vars(cfg)},
    }
    meta_path = os.path.join(results_dir, "run_meta.json")
    with open(meta_path, "w", encoding="utf-8") as f:
        json.dump(run_meta, f, ensure_ascii=False, indent=2)
    print(f"[MPLC2] 运行元信息已保存到 {meta_path}")


def runtime_constants(cfg: SimpleNamespace) -> SimpleNamespace:
    """根据配置计算串扰收敛阈值与掩模偏移常量，用于优化过程。"""
    crs_delta = 1e-13 * cfg.calc_perf_every_it
    mask_offset = cfg.OffsetMultiplier * math.sqrt(1e-3 / (cfg.Nx * cfg.Ny * cfg.n_of_modes))
    return SimpleNamespace(crs_delta=crs_delta, mask_offset=mask_offset)


def build_frequency_grids(cfg: SimpleNamespace, device: torch.device) -> SimpleNamespace:
    """构建频域网格与平方波数张量，以供角谱传播使用。"""
    nx_t = torch.linspace(-(cfg.Nx - 1) / 2, (cfg.Nx - 1) / 2, steps=cfg.Nx, device=device, dtype=torch.float32)
    ny_t = torch.linspace(-(cfg.Ny - 1) / 2, (cfg.Ny - 1) / 2, steps=cfg.Ny, device=device, dtype=torch.float32)
    kx_1d = (2 * math.pi) * nx_t / (cfg.Nx * cfg.pixelSize)
    ky_1d = (2 * math.pi) * ny_t / (cfg.Ny * cfg.pixelSize)
    kx_t, ky_t = torch.meshgrid(kx_1d, ky_1d, indexing="xy")
    k_sq = kx_t ** 2 + ky_t ** 2
    return SimpleNamespace(nx=nx_t, ny=ny_t, k_sq=k_sq)


def load_field_data(cfg: SimpleNamespace, device: torch.device) -> SimpleNamespace:
    """加载 LP 与高斯模数据，生成多波长字段及背景/交叉区域掩模。"""
    lambda_candidates = np.array([1.53e-6, 1.55e-6, 1.57e-6, 1.59e-6, 1.61e-6, 1.625e-6], dtype=np.float64)
    lambda_c = 1.57e-6

    lp_data = np.load("lp_out_140.npz")
    lp_modes = lp_data["profiles"].astype(np.complex64)

    gauss_data = np.load("gauss_10x1_70.npz")
    gauss_basis = gauss_data["Gaussian_basis"].astype(np.complex64)
    gauss_masks = gauss_data["Gaussian_Masks"].astype(np.float32)

    num_lambda_available = min(lp_modes.shape[0], gauss_basis.shape[0])
    if num_lambda_available == 0:
        raise RuntimeError("Gaussian/LP 数据的波长维度为空，无法继续运行。")

    if num_lambda_available <= len(lambda_candidates):
        lambda_list = lambda_candidates[:num_lambda_available]
    else:
        lambda_list = np.linspace(lambda_candidates[0], lambda_candidates[-1], num_lambda_available, dtype=np.float64)

    if lp_modes.shape[0] != gauss_basis.shape[0]:
        print(
            f"[MPLC2][WARN] 数据文件的波长数量不一致：LP={lp_modes.shape[0]}, Gaussian={gauss_basis.shape[0]}。"
            f" 使用共同的最小值 {num_lambda_available}。"
        )

    lp_basis = torch.from_numpy(lp_modes[:num_lambda_available, 0 : cfg.n_of_modes]).to(device)
    gaussian_basis = torch.from_numpy(gauss_basis[:num_lambda_available, 0 : cfg.n_of_modes]).to(device)
    gaussian_masks = torch.from_numpy(gauss_masks[:num_lambda_available, 0 : cfg.n_of_modes]).to(torch.float32).to(device)

    lp_basis, gaussian_basis, gaussian_masks = adjust_resolution(cfg, lp_basis, gaussian_basis, gaussian_masks)
    phi_bk, phi_cr = compute_region_masks(gaussian_masks)

    return SimpleNamespace(
        lambda_list=lambda_list,
        lambda_c=lambda_c,
        lp=lp_basis,
        gaussian=gaussian_basis,
        gaussian_masks=gaussian_masks,
        phi=gaussian_basis,
        phi_bk=phi_bk,
        phi_cr=phi_cr,
    )


def adjust_resolution(
    cfg: SimpleNamespace,
    lp_basis: torch.Tensor,
    gaussian_basis: torch.Tensor,
    gaussian_masks: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """根据配置尺寸对输入/输出模式进行填充或裁剪以匹配模拟网格。"""
    data_size = lp_basis.shape[-1]
    if (cfg.Nx > data_size) or (cfg.Ny > data_size):
        pad_x = int((cfg.Nx - data_size) / 2)
        pad_y = int((cfg.Ny - data_size) / 2)
        pad_tuple = (pad_x, cfg.Nx - data_size - pad_x, pad_y, cfg.Ny - data_size - pad_y)
        lp_padded = pad_complex_tensor(lp_basis, pad_tuple)
        gaussian_padded = pad_complex_tensor(gaussian_basis, pad_tuple)
        gaussian_masks_padded = nn.functional.pad(gaussian_masks, pad_tuple, mode="constant", value=0.0)
        return lp_padded, gaussian_padded, gaussian_masks_padded
    if (cfg.Nx < data_size) or (cfg.Ny < data_size):
        crop_x = int((data_size - cfg.Nx) / 2)
        crop_y = int((data_size - cfg.Ny) / 2)
        lp_cropped = lp_basis[:, :, crop_y : crop_y + cfg.Ny, crop_x : crop_x + cfg.Nx]
        gaussian_cropped = gaussian_basis[:, :, crop_y : crop_y + cfg.Ny, crop_x : crop_x + cfg.Nx]
        gaussian_masks_cropped = gaussian_masks[:, :, crop_y : crop_y + cfg.Ny, crop_x : crop_x + cfg.Nx]
        return lp_cropped, gaussian_cropped, gaussian_masks_cropped
    return lp_basis, gaussian_basis, gaussian_masks


def pad_complex_tensor(tensor: torch.Tensor, pad_tuple: tuple[int, int, int, int]) -> torch.Tensor:
    """对复数张量进行对称零填充，分别处理实部与虚部。"""
    real = nn.functional.pad(tensor.real, pad_tuple, mode="constant", value=0.0)
    imag = nn.functional.pad(tensor.imag, pad_tuple, mode="constant", value=0.0)
    return torch.complex(real, imag)


def compute_region_masks(gaussian_masks: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    """根据高斯掩模生成背景区域与交叉区域掩模张量。"""
    sum_masks = torch.sum(gaussian_masks, dim=1)
    phi_bk = (sum_masks == 0).to(torch.float32)
    phi_cr = ((sum_masks.unsqueeze(1) - gaussian_masks) > 0).to(torch.float32)
    return phi_bk, phi_cr


def preview_inputs(cfg: SimpleNamespace, data: SimpleNamespace) -> None:
    """可选地绘制输入模式与掩模预览，帮助快速检查数据正确性。"""
    if not getattr(cfg, "preview_inputs", 1):
        return
    try:
        l0 = 0

        def complim_on_axis(ax, field) -> None:
            """将复场可视化为 RGB 复合图并显示在指定坐标轴上。"""
            if isinstance(field, torch.Tensor):
                arr = field.detach().cpu().numpy()
            else:
                arr = np.asarray(field)
            if arr.ndim == 3:
                # for batches, take first slice
                arr = arr[0]
            max_val = np.max(np.abs(arr))
            if not np.isfinite(max_val) or max_val == 0:
                max_val = 1.0
            M = arr / max_val
            A = np.abs(M)
            A = np.clip(A, 0.0, 1.0)
            P = np.angle(M)

            R = A * ((np.cos(P - 2 * math.pi / 3) / 2) + 0.5)
            G = A * ((np.cos(P) / 2) + 0.5)
            B = A * ((np.cos(P + 2 * math.pi / 3) / 2) + 0.5)
            C = np.stack((R, G, B), axis=-1)
            ax.imshow(C)
            # ax.axis("off")

        gauss_sum = torch.sum(data.phi[l0], dim=0)
        gauss_mask = torch.sum(data.gaussian_masks[l0], dim=0) > 0
        combined_field = torch.where(gauss_mask, gauss_sum, 0.1 * data.phi_bk[l0].to(torch.complex64))

        fields = [
            (data.lp[l0, 0], "One of the input modes - $\\chi_{0}$"),
            (torch.sum(data.phi[l0], dim=0), "Sum of the output modes - $\\sum\\phi_{i}$"),
            (combined_field, "Gaussian field + background mask"),
            (data.phi_bk[l0], "$\\phi^{bk}$"),
            (data.phi_cr[l0, 0], "$\\phi_{0}^{cr}$"),
        ]

        fig, axes = plt.subplots(2, 3, figsize=(15, 10))
        axes_positions = [axes[0, 0], axes[0, 1], axes[1, 0], axes[1, 1], axes[1, 2]]
        for ax, (field, title) in zip(axes_positions, fields):
            complim_on_axis(ax, field)
            ax.set_title(title)
        axes[0, 2].axis("off")
        fig.suptitle("Input / mask preview", fontsize=16)
        fig.tight_layout(rect=[0, 0, 1, 0.95])
        plt.show()
        plt.close(fig)
    except Exception as exc:
        print("[MPLC2] Preview plots skipped:", exc)


def initialize_state(
    cfg: SimpleNamespace,
    device: torch.device,
    data: SimpleNamespace,
    grids: SimpleNamespace,
    constants: SimpleNamespace,
) -> SimpleNamespace:
    """初始化相位掩模、前后向场以及传播常量，为优化迭代做好准备。"""
    L = data.gaussian_masks.shape[0]
    Masks = torch.zeros((cfg.Planes, cfg.Ny, cfg.Nx), dtype=torch.float32, device=device)
    Masks_complex = torch.exp(1j * Masks)
    Modes_in = torch.zeros((L, cfg.Planes, cfg.n_of_modes, cfg.Ny, cfg.Nx), dtype=torch.complex64, device=device)
    Phi_bwd = torch.zeros_like(Modes_in)
    eff_distribution = torch.ones((cfg.n_of_modes), dtype=torch.float32, device=device)
    dFdpsi = torch.zeros_like(Modes_in)
    history_len = max(1, cfg.iterations // max(1, cfg.calc_perf_every_it))
    crs_history = torch.zeros((history_len), dtype=torch.double, device=device)

    kz_list: List[torch.Tensor] = []
    modes_in0: List[torch.Tensor] = []
    scls: List[float] = []

    for l_idx, lambda_val in enumerate(data.lambda_list):
        k_val = float((2 * math.pi) / float(lambda_val))
        kz_sq = (k_val ** 2) - grids.k_sq
        kz_c = torch.sqrt(kz_sq.to(torch.complex64))
        kz_list.append(kz_c)

        Modes_in[l_idx, 0] = propagate_HK(data.lp[l_idx], kz_c, cfg.d_in)
        Phi_bwd[l_idx, cfg.Planes - 1] = propagate_HK(data.phi[l_idx], kz_c, -cfg.d_out)
        modes_in0.append(Modes_in[l_idx, 0].clone())
        scls.append(float(data.lambda_c / float(lambda_val)))

    return SimpleNamespace(
        Masks=Masks,
        Masks_complex=Masks_complex,
        Modes_in=Modes_in,
        Phi_bwd=Phi_bwd,
        eff_distribution=eff_distribution,
        dFdpsi=dFdpsi,
        crs_history=crs_history,
        conv_count=0,
        kz_list=kz_list,
        modes_in0=modes_in0,
        scls=scls,
        constants=constants,
    )


def build_mask_cache(Masks: torch.Tensor, scls: List[float]) -> List[List[torch.Tensor]]:
    """按波长缩放系数缓存各平面相位掩模，避免重复指数计算。"""
    Planes = Masks.shape[0]
    cache: List[List[torch.Tensor]] = []
    for scl in scls:
        cache.append([torch.exp(1j * (Masks[pl] * scl)) for pl in range(Planes)])
    return cache


def evaluate_performance(cfg: SimpleNamespace, data: SimpleNamespace, state: SimpleNamespace) -> SimpleNamespace:
    """在多波长下前向传播并计算保真度、串扰与效率等性能指标。"""
    fids = []
    crss = []
    effs = []
    eff_lists = []
    for l_idx, _ in enumerate(data.lambda_list):
        kz_l = state.kz_list[l_idx]
        scl = state.scls[l_idx]
        modes = propagate_HK(data.lp[l_idx], kz_l, cfg.d_in)
        for pl in range(cfg.Planes - 1):
            modes = modes * torch.exp(1j * (state.Masks[pl] * scl))
            modes = propagate_HK(modes, kz_l, cfg.d)
        modes = modes * torch.exp(1j * (state.Masks[cfg.Planes - 1] * scl))
        eout = propagate_HK(modes, kz_l, cfg.d_out)
        intensity = torch.abs(eout) ** 2
        fid, _ = performance_loc_fidelity(eout, data.gaussian_masks[l_idx], data.phi[l_idx])
        crs, _, _ = performance_crosstalk(intensity, data.gaussian_masks[l_idx])
        eff, eff_list = performance_efficiency(intensity, data.gaussian_masks[l_idx])
        fids.append(fid)
        crss.append(crs)
        effs.append(eff)
        eff_lists.append(eff_list)

    fid_avg = torch.stack(fids).mean()
    crs_avg = torch.stack(crss).mean()
    eff_avg = torch.stack(effs).mean()
    return SimpleNamespace(fid=fid_avg, crs=crs_avg, eff=eff_avg, fids=fids, crss=crss, effs=effs, eff_lists=eff_lists)


def optimize_phase_masks(cfg: SimpleNamespace, data: SimpleNamespace, state: SimpleNamespace) -> SimpleNamespace:
    """执行主优化循环，通过前后向传播和梯度近似迭代更新相位掩模。"""
    Planes = cfg.Planes
    L = data.gaussian_masks.shape[0]
    Ny, Nx = cfg.Ny, cfg.Nx
    device = state.Masks.device
    stop_requested = False

    for iteration in range(1, cfg.iterations + 1):
        delta_theta = cfg.delta_theta_0 if iteration < cfg.first_n_iterations else cfg.delta_theta_1
        mask_cache = build_mask_cache(state.Masks, state.scls)

        for mask_idx in range(Planes):
            for l_idx in range(L):
                kz_l = state.kz_list[l_idx]
                mask_layers = mask_cache[l_idx]

                modes = state.modes_in0[l_idx]
                state.Modes_in[l_idx, 0] = modes
                for pl in range(Planes - 1):
                    modes = modes * mask_layers[pl]
                    modes = propagate_HK(modes, kz_l, cfg.d)
                    state.Modes_in[l_idx, pl + 1] = modes
                modes = modes * mask_layers[Planes - 1]
                eout_l = propagate_HK(modes, kz_l, cfg.d_out)

                for mode_idx in range(cfg.n_of_modes):
                    ovlp = torch.sum(eout_l[mode_idx] * torch.conj(data.phi[l_idx, mode_idx]))
                    a = data.phi[l_idx, mode_idx] * ovlp
                    psi_cr = eout_l[mode_idx] * data.phi_cr[l_idx, mode_idx]
                    psi_bk = eout_l[mode_idx] * data.phi_bk[l_idx]
                    state.dFdpsi[l_idx, Planes - 1, mode_idx] = -cfg.alpha * a + (cfg.beta * psi_cr - cfg.gamma * psi_bk) * 0.5

                state.dFdpsi[l_idx, Planes - 1] = propagate_HK(state.dFdpsi[l_idx, Planes - 1], kz_l, -cfg.d_out)
                state.Phi_bwd[l_idx, Planes - 1] = propagate_HK(data.phi[l_idx], kz_l, -cfg.d_out)

                for pl in range(Planes - 1, mask_idx, -1):
                    dFdpsi_prop = state.dFdpsi[l_idx, pl] * torch.conj(mask_layers[pl])
                    dFdpsi_prop = propagate_HK(dFdpsi_prop, kz_l, -cfg.d)
                    state.dFdpsi[l_idx, pl - 1] = dFdpsi_prop

                    phi_prop = state.Phi_bwd[l_idx, pl] * torch.conj(mask_layers[pl])
                    phi_prop = propagate_HK(phi_prop, kz_l, -cfg.d)
                    state.Phi_bwd[l_idx, pl - 1] = phi_prop

            total_term = torch.zeros((Ny, Nx), dtype=torch.complex64, device=device)
            if cfg.equalize_efficiency == 1:
                inv_eff = (1.0 / state.eff_distribution.view(cfg.n_of_modes, 1, 1))
                for l_idx in range(L):
                    mask_cmplx = mask_cache[l_idx][mask_idx]
                    Mi = state.Modes_in[l_idx, mask_idx]
                    Gi = state.dFdpsi[l_idx, mask_idx]
                    weighted = torch.sum(inv_eff * Mi * torch.conj(Gi), dim=0)
                    total_term = total_term + mask_cmplx * weighted
            else:
                for l_idx in range(L):
                    mask_cmplx = mask_cache[l_idx][mask_idx]
                    Mi = state.Modes_in[l_idx, mask_idx]
                    Gi = state.dFdpsi[l_idx, mask_idx]
                    overlaps = torch.sum(Mi * torch.conj(Gi), dim=0)
                    total_term = total_term + mask_cmplx * overlaps

            delta_P = delta_theta * torch.sign(torch.imag(total_term))

            if cfg.smoothing_switch == 1:
                ov_sum = torch.zeros((Ny, Nx), dtype=torch.float32, device=device)
                for l_idx in range(L):
                    overlap = torch.sum(
                        state.Modes_in[l_idx, mask_idx] * torch.conj(state.Phi_bwd[l_idx, mask_idx]), dim=0
                    )
                    ov_sum = ov_sum + torch.abs(overlap)
                ov_max = torch.amax(ov_sum)
                ovrlp_in_out = ov_sum / (ov_max + 1e-6)
                mask_complex = ovrlp_in_out * torch.exp(1j * (state.Masks[mask_idx] + delta_P))
                if state.constants.mask_offset != 0:
                    mask_complex = mask_complex + torch.tensor(state.constants.mask_offset, dtype=torch.float32, device=device)
                state.Masks[mask_idx] = torch.angle(mask_complex)
            else:
                state.Masks[mask_idx] = state.Masks[mask_idx] + delta_P

            state.Masks_complex[mask_idx] = torch.exp(1j * state.Masks[mask_idx])
            for l_idx, scl in enumerate(state.scls):
                mask_cache[l_idx][mask_idx] = torch.exp(1j * (state.Masks[mask_idx] * scl))

        if iteration % max(1, cfg.calc_perf_every_it) == 0:
            metrics = evaluate_performance(cfg, data, state)
            fid_val = round(float(metrics.fid.detach().cpu().numpy().item()), 2)
            crs_val = round(float(metrics.crs.detach().cpu().numpy().item()), 2)
            eff_val = round(float(metrics.eff.detach().cpu().numpy().item()), 2)
            print(f"iteration {iteration}: loc. fidelity = {fid_val}, crosstalk = {crs_val}, efficiency = {eff_val}")

            if state.conv_count < state.crs_history.numel():
                state.crs_history[state.conv_count] = metrics.crs
            if (
                state.conv_count > 0
                and iteration > (cfg.iterations / 3)
                and (state.crs_history[state.conv_count - 1] - state.crs_history[state.conv_count]) < state.constants.crs_delta
            ):
                stop_requested = True
            state.conv_count = min(state.conv_count + 1, state.crs_history.numel() - 1)

            if cfg.equalize_efficiency == 1:
                eff_stack = torch.stack(metrics.eff_lists, dim=0)
                eff_med = torch.median(eff_stack, dim=0).values
                eff_dist = torch.clamp(eff_med / torch.max(eff_med), min=1e-6)
                state.eff_distribution = eff_dist.to(device=device, dtype=torch.float32)
                if cfg.plot_eff_distribution == 1:
                    plt.figure()
                    plt.title("efficiency distribution")
                    plt.ylim((0, 1))
                    plt.plot(state.eff_distribution.detach().cpu().numpy())
                    plt.show()

        if stop_requested:
            break

    return SimpleNamespace(last_iteration=iteration, stop_requested=stop_requested)


def save_phase_masks(state: SimpleNamespace, results_dir: str) -> None:
    """将最终相位掩模保存为 `masks_full.pt` 供后续分析或部署。"""
    path = os.path.join(results_dir, "masks_full.pt")
    torch.save(state.Masks.detach().cpu(), path)
    print(f"[Save] Full-size phase masks saved to {path}")


def report_phase_masks(cfg: SimpleNamespace, state: SimpleNamespace) -> None:
    """若开启绘图选项，则可视化每个平面的相位掩模。"""
    if cfg.plot_results != 1:
        return
    for idx in range(cfg.Planes):
        plt.title(f"Phase mask {idx + 1}")
        plot_in_GS(state.Masks[idx])


def report_per_wavelength(data: SimpleNamespace, metrics: SimpleNamespace) -> None:
    """逐个波长输出性能指标，便于分析多波长表现差异。"""
    for idx, (fid, crs, eff) in enumerate(zip(metrics.fids, metrics.crss, metrics.effs)):
        fid_val = float(fid.detach().cpu().numpy().item())
        crs_val = float(crs.detach().cpu().numpy().item())
        eff_val = float(eff.detach().cpu().numpy().item())
        print(
            f"λ={data.lambda_list[idx] * 1e6:.3f} µm -> fidelity={fid_val:.3f}, "
            f"crosstalk={crs_val:.3f}, efficiency={eff_val:.3f}"
        )


def main() -> None:
    """组织整个 MPLC 优化流程：加载数据、优化掩模、报告与保存结果。"""
    cfg = parse_cfg()
    device = select_device()
    results_dir = ensure_results_dir()
    constants = runtime_constants(cfg)
    write_run_meta(cfg, device, results_dir)
    grids = build_frequency_grids(cfg, device)
    data = load_field_data(cfg, device)
    preview_inputs(cfg, data)

    state = initialize_state(cfg, device, data, grids, constants)
    optimize_phase_masks(cfg, data, state)
    final_metrics = evaluate_performance(cfg, data, state)
    fid_final = round(float(final_metrics.fid.detach().cpu().numpy().item()), 3)
    crs_final = round(float(final_metrics.crs.detach().cpu().numpy().item()), 3)
    eff_final = round(float(final_metrics.eff.detach().cpu().numpy().item()), 3)
    print(
        "Final performance (avg over λ): loc. fidelity =",
        fid_final,
        ", crosstalk =",
        crs_final,
        ", efficiency =",
        eff_final,
    )

    report_phase_masks(cfg, state)
    report_per_wavelength(data, final_metrics)
    save_phase_masks(state, results_dir)

    return SimpleNamespace(
        cfg=cfg,
        device=device,
        data=data,
        state=state,
        metrics=final_metrics,
        results_dir=results_dir,
    )


if __name__ == "__main__":
    main()



