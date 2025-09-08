if CFG.get("do_padded_eval", 0) == 1:
    # 选择中心波长对应索引（更稳健而非固定 2）
    l_c = int(np.argmin(np.abs(lambda_list - lambda_c))) if len(lambda_list) > 0 else 0
    scl_c = float(lambda_c / float(lambda_list[l_c])) if len(lambda_list) > 0 else 1.0  # 通常=1

    # 更大的计算视场：上下左右各扩 200 像素（可调）
    newNx = Nx + 400
    newNy = Ny + 400
    pad_x = (newNx - Nx) // 2
    pad_y = (newNy - Ny) // 2

    # 宽域 k-空间：在设备上用 torch 构建，与主流程一致（indexing='xy'）
    nx_w = torch.linspace((-(newNx-1)/2), ((newNx-1)/2), steps=newNx, device=DEVICE, dtype=torch.float32)
    ny_w = torch.linspace((-(newNy-1)/2), ((newNy-1)/2), steps=newNy, device=DEVICE, dtype=torch.float32)
    kx1d_w = (2*math.pi) * nx_w / (newNx * pixelSize)
    ky1d_w = (2*math.pi) * ny_w / (newNy * pixelSize)
    kx_w, ky_w = torch.meshgrid(kx1d_w, ky1d_w, indexing='xy')  # 形状 (newNy, newNx)
    kperp2_w = kx_w**2 + ky_w**2
    k0_c = (2*math.pi) / float(lambda_list[l_c]) if len(lambda_list) > 0 else (2*math.pi) / lambda_c
    kz_torch_wide = torch.sqrt((k0_c**2 - kperp2_w).to(torch.complex64))  # complex64 on DEVICE

    # 将输入（z=0）嵌入宽域中心，然后在宽域传播 d_in -> 得到 p0 pre-phase
    modes_wide = torch.zeros((n_of_modes, newNy, newNx), dtype=torch.complex64, device=DEVICE)
    modes_wide[:, pad_y:pad_y+Ny, pad_x:pad_x+Nx] = Speckle_basis_torch[l_c]
    modes_wide = propagate_HK(modes_wide, kz_torch_wide, d_in)

    # 宽域相位面：默认外圈相位为 0，把已优化相位贴到中心，并按 λ 比例缩放（λc 时 ==1）
    Masks_wide = torch.zeros((Planes, newNy, newNx), dtype=torch.float32, device=DEVICE)
    Masks_wide[:, pad_y:pad_y+Ny, pad_x:pad_x+Nx] = Masks
    # per-plane complex phase at λ_c
    Masks_complex_wide = torch.exp(1j * (Masks_wide * scl_c))

    # 依次在宽域传播到各平面，再到输出面
    for pl in range(Planes-1):
        modes_wide = modes_wide * Masks_complex_wide[pl]
        modes_wide = propagate_HK(modes_wide, kz_torch_wide, d)
    modes_wide = modes_wide * Masks_complex_wide[Planes-1]
    eout_wide = propagate_HK(modes_wide, kz_torch_wide, d_out)

    # 裁剪到原视场并评估
    eout_cropped = eout_wide[:, pad_y:pad_y+Ny, pad_x:pad_x+Nx]
    eout_cropped_int_only = (torch.abs(eout_cropped))**2
    fid_wide, _ = performance_loc_fidelity(eout_cropped, Gaussian_Masks_torch[l_c], phi[l_c])
    crs_wide, _, _ = performance_crosstalk(eout_cropped_int_only, Gaussian_Masks_torch[l_c])
    eff_wide, _ = performance_efficiency(eout_cropped_int_only, Gaussian_Masks_torch[l_c])
    print('performance padded (λc): loc. fidelity =', round(fid_wide.detach().cpu().numpy().item(),3), ', crosstalk =', round(crs_wide.detach().cpu().numpy().item(),3), ', efficiency =', round(eff_wide.detach().cpu().numpy().item(),3))

    # 宽域全谱评估（逐波长打印所需指标）
    with torch.no_grad():
        Nl = len(lambda_list)
        ILs_w = np.zeros(Nl)
        MDLs_w = np.zeros(Nl)
        XTs_avg_dB_w = np.zeros(Nl)
        fids_w = np.zeros(Nl)
        crss_w = np.zeros(Nl)
        effs_w = np.zeros(Nl)

        for l in range(Nl):
            scl = float(lambda_c / float(lambda_list[l]))
            k0 = (2*math.pi) / float(lambda_list[l])
            kz_torch_wide_l = torch.sqrt((k0**2 - kperp2_w).to(torch.complex64))

            # 从 z=0 的输入场开始在宽域传播
            modes_w = torch.zeros((n_of_modes, newNy, newNx), dtype=torch.complex64, device=DEVICE)
            modes_w[:, pad_y:pad_y+Ny, pad_x:pad_x+Nx] = Speckle_basis_torch[l]
            modes_w = propagate_HK(modes_w, kz_torch_wide_l, d_in)
            for pl in range(Planes-1):
                modes_w = modes_w * torch.exp(1j * (Masks_wide[pl] * scl))
                modes_w = propagate_HK(modes_w, kz_torch_wide_l, d)
            modes_w = modes_w * torch.exp(1j * (Masks_wide[Planes-1] * scl))
            eout_w = propagate_HK(modes_w, kz_torch_wide_l, d_out)

            # 裁剪到原视场
            eout_crop = eout_w[:, pad_y:pad_y+Ny, pad_x:pad_x+Nx]

            # 耦合矩阵与 IL/MDL/XTs_avg_dB（宽域裁剪后）
            E = eout_crop.reshape(n_of_modes, -1)
            P = phi[l].reshape(n_of_modes, -1)
            num = E @ torch.conj(P).T
            normE = torch.sum(torch.abs(E)**2, dim=1)
            normP = torch.sum(torch.abs(P)**2, dim=1)
            denom = torch.sqrt(normE[:, None] * normP[None, :]) + 1e-12
            C_np = (num / denom).detach().cpu().numpy()
            s = np.linalg.svd(C_np, compute_uv=False)
            s2 = s**2
            ILs_w[l] = 10 * np.log10(np.mean(s2))
            MDLs_w[l] = 10 * np.log10(np.max(s2) / (np.min(s2) + 1e-15))
            C2 = np.abs(C_np)**2
            totalPower = np.sum(C2, axis=1)
            signalPower = np.clip(np.diag(C2), 1e-15, None)
            XTs_avg_dB_w[l] = 10 * np.log10(np.mean((totalPower - signalPower) / signalPower))

            # 原有 mask 指标（fidelity/crosstalk/efficiency）
            eout_int = (torch.abs(eout_crop))**2
            fid_l, _ = performance_loc_fidelity(eout_crop, Gaussian_Masks_torch[l], phi[l])
            crs_l, _, _ = performance_crosstalk(eout_int, Gaussian_Masks_torch[l])
            eff_l, _ = performance_efficiency(eout_int, Gaussian_Masks_torch[l])
            fids_w[l] = float(fid_l.detach().cpu().numpy())
            crss_w[l] = float(crs_l.detach().cpu().numpy())
            effs_w[l] = float(eff_l.detach().cpu().numpy())

        print('WIDE-FOV | Wavelengths (μm):', [f'{wl*1e6:.3f}' for wl in lambda_list])
        print('WIDE-FOV | IL (dB):         ', [f'{v:.3f}' for v in ILs_w])
        print('WIDE-FOV | MDL (dB):        ', [f'{v:.3f}' for v in MDLs_w])
        print('WIDE-FOV | XTs_avg (dB):    ', [f'{v:.3f}' for v in XTs_avg_dB_w])
        print('WIDE-FOV | fidelity:        ', [f'{v:.3f}' for v in fids_w])
        print('WIDE-FOV | crosstalk:       ', [f'{v:.3f}' for v in crss_w])
        print('WIDE-FOV | efficiency:      ', [f'{v:.3f}' for v in effs_w])


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

    # 画图：前向（自适应子图数量）
    import matplotlib.pyplot as plt
    nplots_fwd = len(fwd_maps)
    ncols = 4
    nrows = math.ceil(nplots_fwd / ncols)
    fig1, axes1 = plt.subplots(nrows, ncols, figsize=(3*ncols, 3*nrows))
    axes1_flat = np.array(axes1).ravel() if isinstance(axes1, np.ndarray) else np.array([axes1])
    vmax_fwd = max([float(torch.max(m).detach().cpu().numpy()) for m in fwd_maps]) if fwd_maps else 1.0
    for idx in range(nplots_fwd):
        ax = axes1_flat[idx]
        im = ax.imshow(fwd_maps[idx].detach().cpu().numpy(), cmap='inferno', origin='lower')
        ax.set_title(fwd_titles[idx])
        ax.axis('off')
    for k in range(nplots_fwd, nrows*ncols):
        axes1_flat[k].axis('off')
    fig1.suptitle('Forward pre-phase intensity (λ=1.57 μm)')
    fig1.tight_layout()
    fig1.savefig('results/forward_prephase_1p57.png', dpi=150)

    # 画图：后向（自适应子图数量）
    nplots_bwd = len(bwd_maps)
    ncols = 4
    nrows = math.ceil(nplots_bwd / ncols)
    fig2, axes2 = plt.subplots(nrows, ncols, figsize=(3*ncols, 3*nrows))
    axes2_flat = np.array(axes2).ravel() if isinstance(axes2, np.ndarray) else np.array([axes2])
    vmax_bwd = max([float(torch.max(m).detach().cpu().numpy()) for m in bwd_maps]) if bwd_maps else 1.0
    for idx in range(nplots_bwd):
        ax = axes2_flat[idx]
        im = ax.imshow(bwd_maps[idx].detach().cpu().numpy(), cmap='inferno', origin='lower')
        ax.set_title(bwd_titles[idx])
        ax.axis('off')
    for k in range(nplots_bwd, nrows*ncols):
        axes2_flat[k].axis('off')
    fig2.suptitle('Backward pre-phase intensity (λ=1.57 μm)')
    fig2.tight_layout()
    fig2.savefig('results/backward_prephase_1p57.png', dpi=150)

    # 三行 overview：第一行前向，第二行后向按前向顺序反着放（无标题），第三行掩膜居中（少两个，隐藏空位坐标轴）
    ncols_ovr = len(fwd_maps)
    fig_ovr, axes_ovr = plt.subplots(3, ncols_ovr, figsize=(3*ncols_ovr, 9))
    # row 0: forward
    for c in range(ncols_ovr):
        ax = axes_ovr[0, c]
        ax.imshow(fwd_maps[c].detach().cpu().numpy(), cmap='inferno', origin='lower')
        ax.set_title(fwd_titles[c])
        ax.axis('off')
    # row 1: backward, reversed to align positions with forward (no titles)
    bwd_aligned = list(reversed(bwd_maps))
    for c in range(min(ncols_ovr, len(bwd_aligned))):
        ax = axes_ovr[1, c]
        ax.imshow(bwd_aligned[c].detach().cpu().numpy(), cmap='inferno', origin='lower')
        ax.axis('off')
    # row 2: masks centered (Planes = ncols_ovr - 2)
    start = 1 if ncols_ovr >= 2 else 0
    for p in range(Planes):
        c = start + p
        if c < ncols_ovr:
            ax = axes_ovr[2, c]
            ax.imshow(Masks[p].detach().cpu().numpy(), cmap='twilight', origin='lower', vmin=-math.pi, vmax=math.pi)
            ax.axis('off')  # 第三行标题略去
    # 关闭空白子图（第一行、第二行多余列，以及第三行两侧空位）
    for c in range(ncols_ovr):
        if c >= len(fwd_maps):
            axes_ovr[0, c].axis('off')
        if c >= len(bwd_aligned):
            axes_ovr[1, c].axis('off')
        if c < start or c >= start + Planes:
            axes_ovr[2, c].axis('off')
    fig_ovr.suptitle('Overview: forward / backward(aligned) / masks')
    fig_ovr.tight_layout()
    fig_ovr.savefig('results/overview_prephase.png', dpi=150)

    # 相位图：相位面数量自适应（按每行 4 列排布）
    ncols = 4
    nrows = math.ceil(Planes / ncols)
    fig3, axes3 = plt.subplots(nrows, ncols, figsize=(3*ncols, 3*nrows))
    # 将 axes 拉平成 1D，便于按索引逐个填充
    axes3_flat = np.array(axes3).ravel() if isinstance(axes3, np.ndarray) else np.array([axes3])
    for p in range(Planes):
        ax = axes3_flat[p]
        ax.imshow(Masks[p].detach().cpu().numpy(), cmap='twilight', origin='lower', vmin=-math.pi, vmax=math.pi)
        ax.set_title(f'Mask p{p}')
        ax.axis('off')
    # 关掉多余子图
    for k in range(Planes, nrows*ncols):
        axes3_flat[k].axis('off')
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

    # 预创建子图（自适应 Nl 个波长）
    ncols = 3
    nrows = math.ceil(Nl / ncols)
    fig_cm, axes_cm = plt.subplots(nrows, ncols, figsize=(4*ncols, 3.5*nrows))
    axes_cm_flat = np.array(axes_cm).ravel() if isinstance(axes_cm, np.ndarray) else np.array([axes_cm])

    for idx in range(nrows * ncols):
        ax = axes_cm_flat[idx]
        if idx >= Nl:
            ax.axis('off')
            continue
        l = idx
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
        C2_plot = np.flip(C2, axis=1)  # 列翻转显示
        im = ax.imshow(C2_plot, cmap='magma', origin='lower', aspect='equal', vmin=0.0, vmax=1.0)
        ax.set_title(f'λ={lambda_list[l]*1e6:.3f} μm')
        ax.axis('off')
        # 仅为第一行的第一个子图添加色标，避免太多 colorbar
        if idx == 0:
            fig_cm.colorbar(im, ax=ax, fraction=0.046, pad=0.04)

    # 关闭多余子图
    for k in range(Nl, nrows*ncols):
        axes_cm_flat[k].axis('off')

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

    # 自适应每模式可视化（显示所有可用模式，不新增参数）
    M = min(modes_b.shape[0], n_of_modes)
    ncols = min(5, M) if M > 0 else 1
    nrows = math.ceil(M / ncols) if M > 0 else 1
    fig, axes = plt.subplots(nrows, ncols, figsize=(3*ncols, 3*nrows))
    axes_flat = np.array(axes).ravel() if isinstance(axes, np.ndarray) else np.array([axes])
    # 固定可视化范围
    vmax_z0 = max([float(torch.max(torch.abs(modes_b[j])**2).detach().cpu().numpy()) for j in range(M)]) if M > 0 else 1.0
    for j in range(M):
        inten = torch.abs(modes_b[j]) ** 2
        axes_flat[j].imshow(inten.detach().cpu().numpy(), cmap='inferno', origin='lower', vmin=0.0, vmax=vmax_z0)
        axes_flat[j].set_title(f'mode {j} @ z=0')
        axes_flat[j].axis('off')
    for k in range(M, nrows*ncols):
        axes_flat[k].axis('off')
    fig.suptitle('Backward to z=0 per-mode intensity (λ=1.57 μm)')
    fig.tight_layout()
    fig.savefig('results/backward_z0_modes_1p57.png', dpi=150)
    plt.show()

    # 可视化每个模式在 z=0 处的相位分布
    fig_phase, axes_phase = plt.subplots(nrows, ncols, figsize=(3*ncols, 3*nrows))
    axes_phase_flat = np.array(axes_phase).ravel() if isinstance(axes_phase, np.ndarray) else np.array([axes_phase])
    for j in range(M):
        phase = torch.angle(modes_b[j])  # 相位范围 [-π, π]
        axes_phase_flat[j].imshow(phase.detach().cpu().numpy(), cmap='twilight', origin='lower', vmin=-math.pi, vmax=math.pi)
        axes_phase_flat[j].set_title(f'mode {j} phase @ z=0')
        axes_phase_flat[j].axis('off')
    for k in range(M, nrows*ncols):
        axes_phase_flat[k].axis('off')
    fig_phase.suptitle('Backward to z=0 per-mode phase (λ=1.57 μm)')
    fig_phase.tight_layout()
    fig_phase.savefig('results/backward_z0_modes_phase_1p57.png', dpi=150)
    plt.show()


