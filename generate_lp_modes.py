#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""多波长 LP 模生成脚本。

该脚本复用 pyMMF 的阶跃型光纤求解器，流程包含：

1. 解析光纤与网格配置（折射率、数值孔径、网格尺寸等）。
2. 对每个波长调用 `pyMMF.propagationModeSolver` 求解 LP 模，并可按预设次序重排。
3. 以参考波长 LP01 的 1/e² 半径为基准计算统一缩放因子，按需将模式重采样到目标像素尺寸。
4. 将所有波长的前 K 个模式堆叠成 4D 数组并保存，同时附带关键元数据（光纤参数、缩放信息等）。
5. 可选地绘制指定波长的模式网格预览便于快速检查。

与 `generate_gaussian.py` 保持一致的结构，使得参数配置与输出格式相互兼容。"""

from __future__ import annotations

import argparse
import os
from typing import Dict, List, Optional, Sequence, Tuple

import matplotlib.pyplot as plt
import numpy as np
from scipy.ndimage import map_coordinates

import pyMMF


# ---------------------------------------------------------------------------
# 参数解析与校验
# ---------------------------------------------------------------------------


# 构建命令行参数解析器
def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Generate LP modes for a step-index fiber (SI solver)")

    # Fiber & wavelength
    parser.add_argument("--wavelengths-um", type=float, nargs="*", default=[1.53, 1.55, 1.57, 1.59, 1.61, 1.625],)
    parser.add_argument("--core-index", type=float, default=1.4587)
    parser.add_argument("--numerical-aperture", type=float, default=0.21)
    parser.add_argument("--core-radius-um", type=float, default=8.25)
    parser.add_argument("--grid-points", type=int, default=512)
    parser.add_argument("--native-pixel-um", type=float, default=0.25)
    parser.add_argument("--degenerate-mode",choices=["sin", "exp"],default="sin",)
    parser.add_argument("--parallel-jobs", type=int, default=-2)
    parser.add_argument("--disable-radius-scaling",dest="enable_radius_scaling",action="store_false",default=True,)
    parser.add_argument("--target-radius-um", type=float, default=140.0)
    parser.add_argument("--target-pixel-um", type=float, default=8.0)
    parser.add_argument("--reorder-mode",choices=["none", "preset1", "custom"],default="preset1",)
    parser.add_argument("--reorder-list", type=str, default="",)
    parser.add_argument("--output-path", type=str, default="lp_out_140.npz")
    parser.add_argument("--modes-to-save", type=int, default=10)
    parser.add_argument("--plot-wavelength-um", type=float, default=1.55)
    parser.add_argument("--figure-path", type=str, default=None)

    return parser.parse_args()
# ---------------------------------------------------------------------------
# 基础工具函数
# ---------------------------------------------------------------------------

# 根据当前参数生成阶跃型折射率分布
def create_index_profile(args: argparse.Namespace) -> pyMMF.IndexProfile:
    area = args.native_pixel_um * (args.grid_points - 1)
    profile = pyMMF.IndexProfile(npoints=args.grid_points, areaSize=area)
    profile.initStepIndex(n1=args.core_index, a=args.core_radius_um, NA=args.numerical_aperture)
    return profile
# 使用 pyMMF 求解指定波长的 LP 模
def solve_modes(profile: pyMMF.IndexProfile, wl: float, args: argparse.Namespace):
    solver = pyMMF.propagationModeSolver()
    solver.setIndexProfile(profile)
    solver.setWL(wl)
    modes = solver.solve(solver="SI",options={"degenerate_mode": args.degenerate_mode,"n_jobs": args.parallel_jobs,},)
    return modes
# 打印求解得到的模式数量与参数概览
def summarize_modes(wl: float, args: argparse.Namespace, modes) -> None:
    n2 = float(np.sqrt(args.core_index**2 - args.numerical_aperture**2))
    est = int(pyMMF.estimateNumModesSI(wl, args.core_radius_um, args.numerical_aperture))
    print("\n=== LP modes (SI) summary ===")
    print(f"wl = {wl} µm, n1 = {args.core_index}, NA = {args.numerical_aperture} (n2 ≈ {n2:.6f}), a = {args.core_radius_um} µm")
    print(f"estimated modes (SI) ≈ {est}")
    print(f"found modes = {modes.number}")
# 基于 1/e^2 强度定义估算模式半径
def estimate_w_e2_um(field: np.ndarray, X: np.ndarray, Y: np.ndarray) -> float:
    intensity = np.abs(field) ** 2
    i_max = float(intensity.max())
    if i_max <= 0:
        return 0.0
    power = float(intensity.sum())
    x_c = float((intensity * X).sum() / power)
    y_c = float((intensity * Y).sum() / power)
    radius = np.sqrt((X - x_c) ** 2 + (Y - y_c) ** 2)
    r_sorted = radius.ravel()
    i_sorted = (intensity / i_max).ravel()
    order = np.argsort(r_sorted)
    r_sorted = r_sorted[order]
    i_sorted = i_sorted[order]
    target = float(np.exp(-2.0))
    crossing = i_sorted <= target
    if not np.any(crossing):
        return float(r_sorted[-1])
    idx = int(np.argmax(crossing))
    if idx == 0:
        return float(r_sorted[0])
    r1, r2 = float(r_sorted[idx - 1]), float(r_sorted[idx])
    i1, i2 = float(i_sorted[idx - 1]), float(i_sorted[idx])
    if abs(i2 - i1) < 1e-12:
        return r2
    alpha = np.clip((target - i1) / (i2 - i1), 0.0, 1.0)
    return float(r1 + alpha * (r2 - r1))
# 对模式进行缩放并重采样到目标网格
def resample_scaled_field(
    field2d: np.ndarray,
    Xb: np.ndarray,
    Yb: np.ndarray,
    npoints: int,
    pixel_target: float,
    scale_s: float,
) -> np.ndarray:
    # 当前重采样策略：先按 target_pixel_um 定义目标物理尺寸，再用 scale_s（由 LP01 半径和 target_radius 推得）
    # 对坐标反向缩放，最后通过 scipy.ndimage.map_coordinates 在原始实值幅度上做三阶插值。
    # 注意：如果未启用缩放，则调用方会绕过本函数，直接使用 normalize_field。
    x_min = float(Xb[0, 0])
    y_min = float(Yb[0, 0])
    dx = float(Xb[0, 1] - Xb[0, 0])

    area_target = pixel_target * (npoints - 1)
    coords = np.linspace(-area_target / 2.0, area_target / 2.0, npoints)
    X_t, Y_t = np.meshgrid(coords, coords)
    X_q = X_t / scale_s
    Y_q = Y_t / scale_s
    col_idx = (X_q - x_min) / dx
    row_idx = (Y_q - y_min) / dx
    coords_stack = np.vstack([row_idx.ravel(), col_idx.ravel()])
    order = 3 if field2d.shape[0] > 32 else 1
    interp = map_coordinates(
        field2d,
        coords_stack,
        order=order,
        mode="constant",
        cval=0.0,
        prefilter=True,
    )
    field = interp.reshape(npoints, npoints)
    power = float(np.sum(np.abs(field) ** 2))
    if power > 1e-15:
        field = field / np.sqrt(power)
    return field.astype(np.float32)
# 将场归一化到单位功率
def normalize_field(field: np.ndarray) -> np.ndarray:
    power = float(np.sum(np.abs(field) ** 2))
    if power > 1e-15:
        field = field / np.sqrt(power)
    return field.astype(np.float32)


# ---------------------------------------------------------------------------
# 模式排序
# ---------------------------------------------------------------------------
# 解析模式重排列表字符串
def parse_label_list(list_str: str) -> List[Tuple[int, int, int]]:
    result: List[Tuple[int, int, int]] = []
    for token in [t.strip() for t in list_str.split(",") if t.strip()]:
        try:
            lp, ori = token.split(":")
            if not lp.startswith("LP"):
                continue
            m_str, l_str = lp[2:].split("_")
            result.append((int(m_str), int(l_str), int(ori)))
        except ValueError:
            continue
    return result
# 默认的模式重排顺序
def preset1_list() -> List[Tuple[int, int, int]]:
    return [
        (2, 1, 0), (1, 2, 0),
        (1, 1, 0), (3, 1, 0),
        (0, 1, 0), (0, 2, 0),
        (1, 1, 1), (3, 1, 1),
        (2, 1, 1), (1, 2, 1),
    ]
# 依据角向分布判断模式取向
def orientation_index(field2d: np.ndarray, TH: np.ndarray, R: np.ndarray, a_core: float, m: int) -> int:
    if m == 0:
        return 0
    base = np.real(field2d)
    mask = R <= a_core
    c0 = float(np.sum(base[mask] * np.cos(m * TH[mask])))
    c1 = float(np.sum(base[mask] * np.sin(m * TH[mask])))
    return 0 if abs(c0) >= abs(c1) else 1
# 基于重排策略选择需要保留的模式索引
def select_mode_indices(
    modes,
    keep: int,
    reorder_mode: str,
    reorder_list: str,
    TH: np.ndarray,
    R: np.ndarray,
    a_core: float,
    deg_mode: str,
) -> List[int]:
    n_total = modes.number
    indices = list(range(n_total))
    if reorder_mode == "none" or keep <= 0 or deg_mode != "sin":
        return indices[:keep]
    triples: List[Tuple[int, int, int, int]] = []
    for i in range(n_total):
        m = int(modes.m[i])
        l = int(modes.l[i])
        field = np.array(modes.profiles[i]).reshape(R.shape)
        ori = orientation_index(field, TH, R, a_core, m)
        triples.append((i, m, l, ori))
    desired = preset1_list() if reorder_mode == "preset1" else parse_label_list(reorder_list)
    used = set()
    ordered: List[int] = []
    for m, l, ori in desired:
        candidates = [i for (i, mm, ll, oo) in triples if mm == m and ll == l and oo == ori and i not in used]
        candidates.sort()
        if candidates:
            ordered.append(candidates[0])
            used.add(candidates[0])
        if len(ordered) >= keep:
            break
    for i in indices:
        if i not in used:
            ordered.append(i)
            if len(ordered) >= keep:
                break
    return ordered[:keep]


# ---------------------------------------------------------------------------
# 缩放与输出
# ---------------------------------------------------------------------------
# 查找 LP01 模式的索引
def find_lp01_index(modes) -> int:
    indices = [i for i in range(modes.number) if int(modes.m[i]) == 0 and int(modes.l[i]) == 1]
    return indices[0] if indices else 0
# 选择用于缩放的参考波长
def pick_reference_wavelength(wls: Sequence[float], prefer: float = 1.55) -> float:
    if prefer in wls:
        return prefer
    return wls[len(wls) // 2]
# 计算所有波长共用的半径缩放因子
def compute_scaling(
    args: argparse.Namespace,
    wls: Sequence[float],
    modes_cache: Dict[float, any],
    Xb: np.ndarray,
    Yb: np.ndarray,
) -> Tuple[Dict[float, Optional[float]], Dict[str, float]]:
    scale_cache: Dict[float, Optional[float]] = {wl: None for wl in wls}
    scaling_info: Dict[str, float] = {}
    if not args.enable_radius_scaling:
        return scale_cache, scaling_info

    ref_wl = pick_reference_wavelength(wls)
    modes_ref = modes_cache[ref_wl]
    idx_lp01 = find_lp01_index(modes_ref)
    field_lp01 = np.array(modes_ref.profiles[idx_lp01]).reshape(Xb.shape)
    radius = max(estimate_w_e2_um(np.real(field_lp01), Xb, Yb), 1e-12)

    scale = args.target_radius_um / radius
    scaling_info = {
        "reference_wavelength": float(ref_wl),
        "lp01_measured_radius": float(radius),
        "target_radius": float(args.target_radius_um),
        "scaling_factor": float(scale),
        "base_pixel_size": float(args.native_pixel_um),
        "target_pixel_size": float(args.target_pixel_um),
        "grid_points": float(args.grid_points),
    }

    print("\n=== Radius Scaling Analysis ===")
    print(f"Reference wavelength: {ref_wl:.3f} µm")
    print(f"LP01 measured radius (1/e²): {radius:.2f} µm")
    print(f"Target radius: {args.target_radius_um:.2f} µm")
    print(f"Scaling factor: {scale:.3f}")
    if scale < 0.1:
        print("Warning: 大幅缩小可能导致混叠")
    if scale > 10:
        print("Warning: 大幅放大可能引入插值误差")

    for wl in wls:
        scale_cache[wl] = scale
    return scale_cache, scaling_info
# 构建按波长堆叠的 4D 模式数组
def build_profiles_stack(
    args: argparse.Namespace,
    wls: Sequence[float],
    modes_cache: Dict[float, any],
    scale_cache: Dict[float, Optional[float]],
    profile: pyMMF.IndexProfile,
) -> Tuple[np.ndarray, int]:
    if not modes_cache:
        raise RuntimeError("No modes available")
    min_modes = min(m.number for m in modes_cache.values())
    keep = min(args.modes_to_save, min_modes)
    if keep == 0:
        raise RuntimeError("没有可保存的模式")

    Xb, Yb = profile.X, profile.Y
    stack_list: List[np.ndarray] = []

    for wl in wls:
        modes = modes_cache[wl]
        indices = select_mode_indices(
            modes,
            keep,
            args.reorder_mode,
            args.reorder_list,
            TH=profile.TH,
            R=profile.R,
            a_core=args.core_radius_um,
            deg_mode=args.degenerate_mode,
        )
        profile_stack = []
        for idx in indices:
            field = np.array(modes.profiles[idx]).reshape(args.grid_points, args.grid_points)
            base = np.real(field) if args.degenerate_mode == "sin" else np.abs(field)
            scale = scale_cache.get(wl)
            if scale is not None:
                out_field = resample_scaled_field(base, Xb, Yb, args.grid_points, args.target_pixel_um, scale)
            else:
                out_field = normalize_field(base)
            profile_stack.append(out_field)
        stack_list.append(np.stack(profile_stack, axis=0))
        print(f"  λ={wl:.3f}µm: {len(profile_stack)} modes processed")

    profiles_4d = np.stack(stack_list, axis=0)
    return profiles_4d, keep
# 保存模式与相关元数据
def save_profiles(
    path: str,
    profiles_4d: np.ndarray,
    args: argparse.Namespace,
    wls: Sequence[float],
    keep: int,
    scaling_info: Dict[str, float],
) -> None:
    if not path:
        return
    directory = os.path.dirname(path)
    if directory:
        os.makedirs(directory, exist_ok=True)
    data = {
        "profiles": profiles_4d,
        "wavelengths": np.array(wls, dtype=np.float64),
    "fiber_n1": np.array(args.core_index),
    "fiber_NA": np.array(args.numerical_aperture),
    "fiber_core_radius_um": np.array(args.core_radius_um),
    "grid_npoints": np.array(args.grid_points),
    "grid_base_pixel_um": np.array(args.native_pixel_um),
    "grid_target_pixel_um": np.array(args.target_pixel_um if args.enable_radius_scaling else args.native_pixel_um),
        "modes_per_wavelength": np.array(keep),
        "degenerate_mode_is_sin": np.array(args.degenerate_mode == "sin"),
    }
    if scaling_info:
        data.update({
            "scaling_reference_wl": np.array(scaling_info["reference_wavelength"]),
            "scaling_measured_radius": np.array(scaling_info["lp01_measured_radius"]),
            "scaling_target_radius": np.array(scaling_info["target_radius"]),
            "scaling_factor": np.array(scaling_info["scaling_factor"]),
        })
    np.savez_compressed(path, **data)
    size_mb = os.path.getsize(path) / 1024 ** 2
    print(f"✓ Saved 4D array: {profiles_4d.shape}, file size ≈ {size_mb:.1f} MB")
    try:
        loaded = np.load(path)
        print(f"✓ Verification: loaded shape = {loaded['profiles'].shape}")
        loaded.close()
    except Exception as exc:
        print(f"⚠ Verification failed: {exc}")


# ---------------------------------------------------------------------------
# 可视化
# ---------------------------------------------------------------------------
# 绘制指定波长的模式预览
def plot_mode_grid(
    args: argparse.Namespace,
    wls: Sequence[float],
    modes_cache: Dict[float, any],
    scale_cache: Dict[float, Optional[float]],
    profile: pyMMF.IndexProfile,
) -> None:
    if not modes_cache:
        return
    plot_wl = args.plot_wavelength_um
    modes = modes_cache.get(plot_wl)
    if modes is None:
        rounded = {round(key, 3): key for key in modes_cache}
        modes = modes_cache.get(rounded.get(round(plot_wl, 3)))
    if modes is None:
        print(f"Warning: No modes to plot at {plot_wl:.3f} µm")
        return

    keep = min(10, modes.number)
    indices = select_mode_indices(
        modes,
        keep,
        args.reorder_mode,
        args.reorder_list,
        TH=profile.TH,
        R=profile.R,
        a_core=args.core_radius_um,
        deg_mode=args.degenerate_mode,
    )

    imgs: List[np.ndarray] = []
    scale = scale_cache.get(plot_wl)
    for idx in indices:
        field = np.array(modes.profiles[idx]).reshape(args.grid_points, args.grid_points)
        base = np.real(field) if args.degenerate_mode == "sin" else np.abs(field)
        if scale is not None:
            img = resample_scaled_field(base, profile.X, profile.Y, args.grid_points, args.target_pixel_um, scale)
        else:
            img = normalize_field(base)
        imgs.append(np.abs(img))

    if not imgs:
        print("Warning: 无模式可绘制")
        return

    vmax = max(np.max(im) for im in imgs)
    fig, axes = plt.subplots(5, 2, figsize=(8, 18))
    for idx, img in enumerate(imgs):
        row, col = divmod(idx, 2)
        ax = axes[row, col]
        imh = ax.imshow(img, origin="lower", cmap="inferno", vmin=0.0, vmax=vmax)
        ax.set_title(f"WL {plot_wl:.3f}µm | idx {indices[idx]}")
        ax.axis("off")
    for idx in range(len(imgs), 10):
        row, col = divmod(idx, 2)
        axes[row, col].axis("off")
    cax = fig.add_axes([0.92, 0.1, 0.02, 0.8])
    fig.colorbar(imh, cax=cax)
    plt.tight_layout(rect=[0, 0, 0.9, 1])
    if args.figure_path:
        plt.savefig(args.figure_path, dpi=200, bbox_inches="tight")
        print(f"✓ Saved visualization: {args.figure_path}")
    else:
        plt.show()


# ---------------------------------------------------------------------------
# 主流程
# ---------------------------------------------------------------------------
# 主流程：解析参数、求解、缩放、保存与绘图
def main() -> int:
    args = parse_args()
    wls = list(args.wavelengths_um or [])
    profile = create_index_profile(args)
    modes_cache: Dict[float, any] = {}

    print("\n=== Solving LP modes ===")
    for wl in wls:
        modes = solve_modes(profile, wl, args)
        modes_cache[wl] = modes
        summarize_modes(wl, args, modes)

    scale_cache, scaling_info = compute_scaling(args, wls, modes_cache, profile.X, profile.Y)

    if args.output_path:
        print("\n=== Saving 4D Mode Data ===")
        profiles_4d, keep = build_profiles_stack(args, wls, modes_cache, scale_cache, profile)
        save_profiles(args.output_path, profiles_4d, args, wls, keep, scaling_info)

    print("\n=== Plotting Preview ===")
    plot_mode_grid(args, wls, modes_cache, scale_cache, profile)

    total_modes = sum(m.number for m in modes_cache.values())
    print("\n=== Generation Complete ===")
    print(f"Total wavelengths processed: {len(wls)}")
    print(f"Total modes generated: {total_modes}")
    if args.output_path:
        print(f"Output saved to: {args.output_path}")
    return 0


if __name__ == "__main__":
    import sys

    sys.exit(main())
