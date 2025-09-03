# -*- coding: utf-8 -*-
import os
os.makedirs('results', exist_ok=True)

with torch.no_grad():
    for l in range(len(lambda_list)):
        scale_l = lambda_list[l] / lambda_c

        # 从该波长的输入模式重新做一次前向（不改变已有训练结果）
        modes = propagate_HK(Speckle_basis_torch[l], kz_torch_list[l], d_in)  # (n_modes, Ny, Nx)
        for pl in range(Planes - 1):
            mask_cmplx_l = torch.exp(1j * (Masks[pl] * scale_l))
            modes = modes * mask_cmplx_l
            modes = propagate_HK(modes, kz_torch_list[l], d)

        modes = modes * torch.exp(1j * (Masks[Planes - 1] * scale_l))
        psi = fft2(modes)
        psi_int_only = (torch.abs(psi))**2

        # 计算该波长的耦合矩阵
        crs, crs_list, crs_matrix = performance_crosstalk(psi_int_only, Gaussian_Masks_torch[l])

        # 画图并保存
        plt.figure(figsize=(5,4))
        plt.imshow(crs_matrix.detach().cpu().numpy(), cmap='magma', origin='lower', aspect='equal')
        plt.colorbar(label='coupling')
        plt.title(f'Coupling matrix λ={lambda_list[l]*1e6:.3f} μm')
        plt.tight_layout()
        out_path = f"results/coupling_matrix_{l}_{lambda_list[l]*1e6:.3f}um.png"
        plt.savefig(out_path, dpi=150)
        plt.show()
        print("saved:", out_path)