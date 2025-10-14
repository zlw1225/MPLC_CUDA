#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
简化版：生成多波长 Gaussian 与 Triangle 基底与对应的二值 mask，并保存为 .npz
"""

import argparse
import numpy as np
import matplotlib.pyplot as plt


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument('--rows', type=int, default=10)
    p.add_argument('--cols', type=int, default=1)
    p.add_argument('--max_beams', type=int, default=10)
    p.add_argument('--out_gauss', type=str, default='gauss_array_simple.npz')
    p.add_argument('--out_tri', type=str, default='triangle_array_simple.npz')
    p.add_argument('--wls', type=float, nargs='+', default=[1.53, 1.55, 1.57])
    p.add_argument('--mfd_um', type=float, default=70.0)
    p.add_argument('--lambda_ref', type=float, default=1.55)
    p.add_argument('--d_fiber_um', type=float, default=250.0)
    p.add_argument('--npoints', type=int, default=512)
    p.add_argument('--pixel', type=float, default=8.0)
    p.add_argument('--mask_extra_pixels', type=float, default=7.0)
    return p.parse_args()


def make_xy_grid(npoints, pixel):
    L = pixel * (npoints - 1)
    coords = np.linspace(-L / 2.0, L / 2.0, npoints, dtype=np.float32)
    X, Y = np.meshgrid(coords, coords, indexing='xy')
    return X, Y


def gaussian_field(X, Y, x0, y0, w0_um, lambda_um):
    r2 = (X - x0) ** 2 + (Y - y0) ** 2
    E = np.exp(-r2 / (w0_um ** 2)).astype(np.complex64)
    return E / np.sqrt(np.sum(np.abs(E) ** 2))


def mask_from_gauss(X, Y, x0, y0, w0_um, pixel_size, extra_px=7.0):
    r = np.sqrt((X - x0) ** 2 + (Y - y0) ** 2)
    w0_px = w0_um / pixel_size
    mask_radius_um = (w0_px + extra_px) * pixel_size
    return (r <= mask_radius_um).astype(np.uint8)


def generate_grid_centers(rows, cols, d_um, max_beams=None):
    total = rows * cols
    K = total if max_beams is None else min(max_beams, total)
    x_grid = (np.arange(cols) - (cols - 1) / 2) * d_um
    y_grid = (np.arange(rows) - (rows - 1) / 2) * d_um
    centers = [(y, x) for y in y_grid for x in x_grid][:K]
    return centers, K


def generate_triangle_centers(rows, cols, d_um, max_beams=None):
    """
    生成等腰直角三角布局（dy = dx），
    逆时针旋转 45°，并右移使整体在 X 方向居中。
    """
    centers = []
    count = 0
    layer = 1
    while count < (max_beams or rows * cols):
        for i in range(layer):
            if count >= (max_beams or rows * cols):
                break
            x = -i * d_um
            y = (layer - 1 - i) * d_um
            centers.append((x, y))
            count += 1
        layer += 1

    # 逆时针旋转 45°
    theta = np.deg2rad(45)
    rotated = [
        (x * np.cos(theta) - y * np.sin(theta),
         x * np.sin(theta) + y * np.cos(theta))
        for x, y in centers
    ]

    # X方向居中：右移 Δx = (max(x) + abs(min(x)) - 单个圆半径) / 2
    # 但这里没有圆半径参数，取 fiber pitch 的一半近似代替
    xs = [x for x, _ in rotated]
    if len(xs) > 1:
        r_approx = d_um / 2
        dx_shift = (max(xs) + abs(min(xs)) ) / 2
        rotated = [(x + dx_shift, y) for x, y in rotated]

    # 输出格式与原函数一致：(y, x)
    centers_yx = [(y, x) for x, y in rotated]

    if max_beams:
        centers_yx = centers_yx[:max_beams]
    return centers_yx, len(centers_yx)



def generate_basis_and_masks(X, Y, centers_yx, args):
    w0_ref = args.mfd_um / 2.0
    nwls = len(args.wls)
    K = len(centers_yx)
    basis = np.zeros((nwls, K, args.npoints, args.npoints), dtype=np.complex64)
    masks = np.zeros((nwls, K, args.npoints, args.npoints), dtype=np.uint8)

    for iw, wl in enumerate(args.wls):
        w0_um = w0_ref * (wl / args.lambda_ref)
        for k, (y0, x0) in enumerate(centers_yx):
            E = gaussian_field(X, Y, x0, y0, w0_um, wl)
            mask = mask_from_gauss(X, Y, x0, y0, w0_um, args.pixel, args.mask_extra_pixels)
            basis[iw, k] = E
            masks[iw, k] = mask
    return basis, masks


def main():
    args = parse_args()
    X, Y = make_xy_grid(args.npoints, args.pixel)

    # ---------------- Gaussian 阵列 ----------------
    centers_grid, K_g = generate_grid_centers(args.rows, args.cols, args.d_fiber_um, args.max_beams)
    gauss_basis, gauss_masks = generate_basis_and_masks(X, Y, centers_grid, args)
    np.savez_compressed(args.out_gauss,
                        Gaussian_basis=gauss_basis, Gaussian_Masks=gauss_masks)
    print(f"已保存 Gaussian 阵列: {args.out_gauss}")
    print(f"内存占用: basis={gauss_basis.nbytes / 1024 ** 2:.1f} MB, masks={gauss_masks.nbytes / 1024 ** 2:.1f} MB")

    # ---------------- Triangle 阵列 ----------------
    centers_tri, K_t = generate_triangle_centers(args.rows, args.cols, args.d_fiber_um, args.max_beams)
    tri_basis, tri_masks = generate_basis_and_masks(X, Y, centers_tri, args)
    np.savez_compressed(args.out_tri,
                        triangle_basis=tri_basis, triangle_Masks=tri_masks)
    print(f"已保存 Triangle 阵列: {args.out_tri}")
    print(f"内存占用: basis={tri_basis.nbytes / 1024 ** 2:.1f} MB, masks={tri_masks.nbytes / 1024 ** 2:.1f} MB")

    # ---------------- 验证与显示 ----------------
    print("\n验证保存结果...")
# ...existing code...
    d_g = np.load(args.out_gauss)
    d_t = np.load(args.out_tri)

    # 新增：打印形状
    print("Gaussian_basis shape:", d_g["Gaussian_basis"].shape)      # (nwls, K_g, npoints, npoints)
    print("Gaussian_Masks shape:", d_g["Gaussian_Masks"].shape)
    print("triangle_basis shape:", d_t["triangle_basis"].shape)      # (nwls, K_t, npoints, npoints)
    print("triangle_Masks shape:", d_t["triangle_Masks"].shape)

    # 汇总所有 (波长, beam) 维度
    E_g_sum = d_g["Gaussian_basis"].sum(axis=(0, 1))
    M_g_sum = d_g["Gaussian_Masks"].sum(axis=(0, 1))  # 为整数覆盖次数
    E_t_sum = d_t["triangle_basis"].sum(axis=(0, 1))
    M_t_sum = d_t["triangle_Masks"].sum(axis=(0, 1))

    amp_g = np.abs(E_g_sum)
    amp_t = np.abs(E_t_sum)

    # 可选：把 mask 覆盖次数>0 转为 0/1 显示轮廓
    M_g_edge = (M_g_sum > 0).astype(np.uint8)
    M_t_edge = (M_t_sum > 0).astype(np.uint8)

    fig, axs = plt.subplots(1, 2, figsize=(10, 4))
    axs[0].imshow(amp_g + 0.1 * (1 - M_g_edge), cmap="inferno")
    axs[0].set_title("Gaussian Sum (all λ & beams)")
    axs[1].imshow(amp_t + 0.1 * (1 - M_t_edge), cmap="inferno")
    axs[1].set_title("Triangle Sum (all λ & beams)")
    for ax in axs:
        ax.axis("off")
    plt.tight_layout()
    plt.show()
# ...existing code...


if __name__ == '__main__':
    main()
