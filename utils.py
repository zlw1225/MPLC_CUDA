from typing import Union
import numpy as np
import torch
from functools import singledispatch
from math import pi
from matplotlib import pyplot as plt



__all__ = [
    "fft2",
    "ifft2",
    "normalize",
    "propagate_HK",
    "fidelity",
    "loc_fidelity",
    "performance_loc_fidelity",
    "performance_efficiency",
    "performance_crosstalk",
    "complim",
    "complim_subplot2",
    "plot_in_GS",
]


@singledispatch
def fft2(x: Union[np.ndarray, torch.Tensor]) -> Union[np.ndarray, torch.Tensor]:
    """
    2D discrete forward fourier transform

    For any 2D array `x`
    assert(np.allclose(x, ifft2(fft2(x))))
    assert(np.allclose(x, fft2(ifft2(x))))
    """
    raise NotImplementedError(
        f"Cannot fourier transform `x` for type: {type(x)}")

@fft2.register
def _(x: np.ndarray) -> np.ndarray:
    """
    2D discrete forward fourier transform
    """
    return np.fft.fftshift(np.fft.fft2(np.fft.ifftshift(x, axes=(-1, -2)), norm="ortho"), axes=(-1, -2))

@fft2.register
def fft2_torch(x: torch.Tensor) -> torch.Tensor:
    """
    2D discrete forward fourier transform
    (x: torch.Tensor) -> torch.Tensor
    """
    return torch.fft.fftshift(torch.fft.fft2(torch.fft.ifftshift(x, dim=(-1, -2)), norm="ortho"), dim=(-1, -2))



@singledispatch
def ifft2(x: Union[np.ndarray, torch.Tensor]) -> Union[np.ndarray, torch.Tensor]:
    """
    2D discrete inverse fourier transform

    For any 2D array `x`
    assert(np.allclose(x, ifft2(fft2(x))))
    assert(np.allclose(x, fft2(ifft2(x))))
    """
    raise NotImplementedError(f"Cannot Inverse fourier transform `x` for {type(x)}")

@ifft2.register
def _(x: np.ndarray) -> np.ndarray:
    """
    2D discrete inverse fourier transform
    """
    return np.fft.fftshift(np.fft.ifft2(np.fft.ifftshift(x, axes=(-1, -2)), norm="ortho"), axes=(-1, -2))

@ifft2.register
def ifft2_torch(x: torch.Tensor) -> torch.Tensor:
    """
    2D discrete inverse fourier transform
    """
    return torch.fft.fftshift(torch.fft.ifft2(torch.fft.ifftshift(x, dim=(-1, -2)), norm="ortho"), dim=(-1, -2))



@singledispatch
def normalize(x: Union[np.ndarray, torch.Tensor]) -> Union[np.ndarray, torch.Tensor]:
    raise NotImplementedError(f"Cannot normalize for {type(x)}")

@normalize.register
def _(x: torch.Tensor) -> torch.Tensor:
    return x / torch.linalg.norm(x)

@normalize.register
def _(x: np.ndarray) -> np.ndarray:
    return x / np.linalg.norm(x)



@singledispatch
def propagate_HK(FieldIn: Union[np.ndarray, torch.Tensor],
                 distance: float = 0.0,
                 wavelength: float = 532e-9,
                 dx: float = 8e-6,
                 dy: float = 8e-6,
                 pad_ratio: float = 2.0) -> Union[np.ndarray, torch.Tensor]:
    """
    Free-space propagation using Band-Limited Angular Spectrum Method (BLAS)
    --------------------------------------------------------
    Automatically removes evanescent waves, applies zero-padding, 
    and performs band-limiting to avoid aliasing.

    Args:
        FieldIn: 2D complex field (np.ndarray or torch.Tensor)
        distance: propagation distance [m]
        wavelength: wavelength [m]
        dx, dy: sampling pitch [m]
        pad_ratio: zero-padding ratio (1.0 = no pad, 2.0 = double size)

    Returns:
        Propagated complex field, same shape as input.
    """
    raise NotImplementedError(f"Cannot process `FieldIn` type: {type(FieldIn)}")


@propagate_HK.register
def _(FieldIn: np.ndarray,
      distance: float = 0.0,
      wavelength: float = 532e-9,
      dx: float = 8e-6,
      dy: float = 8e-6,
      pad_ratio: float = 2.0) -> np.ndarray:
    """
    ✅ NumPy 版本带限角谱法传播
    """
    Ny, Nx = FieldIn.shape
    Δfx = 1 / (Nx * dx)
    Δfy = 1 / (Ny * dy)

    # 1️⃣ 混叠判断
    z_crit_x = (1 / (2 * Δfx)) * np.sqrt((2 * dx / wavelength)**2 - 1)
    z_crit_y = (1 / (2 * Δfy)) * np.sqrt((2 * dy / wavelength)**2 - 1)

    if distance >= z_crit_x:
        print(f"[BLAS] ⚠ z = {distance:.3e} >= z_crit_x = {z_crit_x:.3e} → x方向可能混叠")
    if distance >= z_crit_y:
        print(f"[BLAS] ⚠ z = {distance:.3e} >= z_crit_y = {z_crit_y:.3e} → y方向可能混叠")

    # 2️⃣ 零填充
    if pad_ratio > 1.0:
        pad_x = int((pad_ratio - 1) * Nx // 2)
        pad_y = int((pad_ratio - 1) * Ny // 2)
        FieldIn = np.pad(FieldIn, ((pad_y, pad_y), (pad_x, pad_x)), mode='constant')
    else:
        pad_x = pad_y = 0

    Ny_p, Nx_p = FieldIn.shape

    # 3️⃣ 构造频率网格
    nx_t = np.arange(Nx_p) - Nx_p // 2
    ny_t = np.arange(Ny_p) - Ny_p // 2

    # 手动构造空间角频率的取值范围 fx, fy (cycles/m)
    Δfx = 1 / (Nx_p * dx)
    Δfy = 1 / (Ny_p * dy)
    fx = nx_t * Δfx
    fy = ny_t * Δfy
    FX, FY = np.meshgrid(fx, fy, indexing="xy")

    # 4️⃣ kz与带限掩模
    k = 1 / wavelength
    inside = (FX**2 + FY**2) <= k**2
    kz = 2 * np.pi * np.sqrt(np.clip(k**2 - FX**2 - FY**2, 0, None))

    u_max_x = 1 / (np.sqrt((2 * Δfx * distance)**2 + 1) * wavelength)
    u_max_y = 1 / (np.sqrt((2 * Δfy * distance)**2 + 1) * wavelength)
    BandMask = ((np.abs(FX) < u_max_x) & (np.abs(FY) < u_max_y)).astype(float)

    # 5️⃣ 傅里叶传播
    FieldIn_FT = np.fft.fftshift(np.fft.fft2(np.fft.ifftshift(FieldIn)))
    H = np.exp(1j * kz * distance) * inside
    FieldOut_FT = FieldIn_FT * H * BandMask
    FieldOut = np.fft.fftshift(np.fft.ifft2(np.fft.ifftshift(FieldOut_FT)))

    # 6️⃣ 截取
    if pad_ratio > 1.0:
        FieldOut = FieldOut[pad_y:-pad_y, pad_x:-pad_x]

    return FieldOut


@propagate_HK.register
def _(FieldIn: torch.Tensor,
      distance: float = 0.0,
      wavelength: float = 532e-9,
      dx: float = 8e-6,
      dy: float = 8e-6,
      pad_ratio: float = 2.0) -> torch.Tensor:
    """
    ✅ 带限角谱法 (Band-Limited Angular Spectrum Propagation)
    自动零填充 + 带限滤波 + 隐失波去除 + 截取中心区域
    """

    device = FieldIn.device
    Ny, Nx = FieldIn.shape

    Δfx = 1 / (Nx * dx)
    Δfy = 1 / (Ny * dy)

    # === 1️⃣ 混叠检查 ===
    z_crit_x = (1 / (2 * Δfx)) * np.sqrt((2 * dx / wavelength)**2 - 1)
    z_crit_y = (1 / (2 * Δfy)) * np.sqrt((2 * dy / wavelength)**2 - 1)

    if distance >= z_crit_x:
        print(f"[BLAS] ⚠ z = {distance:.3e} >= z_crit_x = {z_crit_x:.3e} → x方向可能混叠")
    if distance >= z_crit_y:
        print(f"[BLAS] ⚠ z = {distance:.3e} >= z_crit_y = {z_crit_y:.3e} → y方向可能混叠")

    # === 2️⃣ 零填充 ===
    if pad_ratio > 1.0:
        pad_x = int((pad_ratio - 1) * Nx // 2)
        pad_y = int((pad_ratio - 1) * Ny // 2)
        FieldIn = torch.nn.functional.pad(FieldIn, (pad_x, pad_x, pad_y, pad_y))
    else:
        pad_x = pad_y = 0

    Ny_p, Nx_p = FieldIn.shape

    # === 3️⃣ 构造频率坐标 ===
    nx_t = torch.arange(Nx_p, device=device, dtype=torch.float32) - Nx_p // 2
    ny_t = torch.arange(Ny_p, device=device, dtype=torch.float32) - Ny_p // 2

    # 手动构造空间角频率的取值范围 fx, fy (cycles/m)
    Δfx = 1 / (Nx_p * dx)
    Δfy = 1 / (Ny_p * dy)
    fx = nx_t * Δfx
    fy = ny_t * Δfy
    FX, FY = torch.meshgrid(fx, fy, indexing="xy")

    # === 4️⃣ kz与带限掩模 === 2\pi\sqrt{\lambda^{-2} - v_x^2 - v_y^2}
    k = 1 / wavelength
    inside = (FX**2 + FY**2) <= k**2
    kz = 2 * np.pi * torch.sqrt(torch.clamp(k**2 - FX**2 - FY**2, min=0))

    u_max_x = 1 / (torch.sqrt((2 * Δfx * distance)**2 + 1) * wavelength)
    u_max_y = 1 / (torch.sqrt((2 * Δfy * distance)**2 + 1) * wavelength)
    BandMask = ((torch.abs(FX) < u_max_x) & (torch.abs(FY) < u_max_y)).float()

    # === 5️⃣ 传播 ===
    FieldIn_FT = torch.fft.fftshift(torch.fft.fft2(torch.fft.ifftshift(FieldIn)))
    H = torch.exp(1j * kz * distance) * inside
    FieldOut_FT = FieldIn_FT * H * BandMask
    FieldOut = torch.fft.fftshift(torch.fft.ifft2(torch.fft.ifftshift(FieldOut_FT)))

    # === 6️⃣ 截取中心区域 ===从 2D 数组 U 中，去除上下各 pad_y 行、左右各 pad_x 列的边缘区域，保留中间有效部分。
    if pad_ratio > 1.0:
        FieldOut = FieldOut[pad_y:-pad_y, pad_x:-pad_x]

    return FieldOut

# @propagate_HK.register
# def _(FieldIn: np.ndarray, kz: np.ndarray, distance: float = 0.0) -> np.ndarray:
#     FieldIn_FT = fft2(FieldIn)
#     FieldOut_FT = FieldIn_FT*np.exp(1j*kz*distance)*(np.imag(kz)==0)
#     FieldOut = ifft2(FieldOut_FT)
#     return FieldOut

# @propagate_HK.register
# def _(FieldIn: torch.Tensor, kz: torch.Tensor, distance: float = 0.0) -> torch.Tensor:
#     FieldIn_FT = fft2(FieldIn)
#     FieldOut_FT = FieldIn_FT*torch.exp(1j*kz*distance)*(torch.imag(kz)==0)
#     FieldOut = ifft2(FieldOut_FT)
#     return FieldOut

@singledispatch
def fidelity(a: Union[np.ndarray, torch.Tensor], b: Union[np.ndarray, torch.Tensor]) -> Union[np.ndarray, torch.Tensor]:
    # returns a float number between 0 and 1
    raise NotImplementedError(f"Cannot check fidelity of `a`, `b` for {type(a)}, {type(b)}")

@fidelity.register
def _(a: np.ndarray, b: np.ndarray) -> float:
    return np.square(np.abs(np.sum(normalize(a).conj() * normalize(b))))

@fidelity.register
def _(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    return torch.square(torch.abs(torch.sum(normalize(a).conj() * normalize(b))))




@singledispatch
def loc_fidelity(a: Union[np.ndarray, torch.Tensor], channel: Union[np.ndarray, torch.Tensor], b: Union[np.ndarray, torch.Tensor]) -> Union[np.ndarray, torch.Tensor]:
    # returns a float number between 0 and 1
    raise NotImplementedError(f"Cannot check fidelity of `a`, `b` for {type(a)}, {type(b)}")

@loc_fidelity.register
def _(a: np.ndarray, channel: np.ndarray, b: np.ndarray) -> float:
    a = a*channel
    return np.square(np.abs(np.sum(normalize(a).conj() * normalize(b))))

@loc_fidelity.register
def _(a: torch.Tensor, channel: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    a = a*channel
    return torch.square(torch.abs(torch.sum(normalize(a).conj() * normalize(b))))




@singledispatch
def performance_loc_fidelity(A: Union[np.ndarray, torch.Tensor], channels: Union[np.ndarray, torch.Tensor], B: Union[np.ndarray, torch.Tensor]) -> Union[np.ndarray, torch.Tensor]:
    raise NotImplementedError(f"Cannot check fidelity of `A`, `B` for {type(A)}, {type(B)}")

@performance_loc_fidelity.register
def _(A: np.ndarray, channels: np.ndarray, B: np.ndarray) -> Union[np.ndarray, float]:
    A = np.squeeze(A)
    B = np.squeeze(B)
    CH = np.squeeze(channels)
    fid_list = np.zeros((A.shape[0]))
    for i in range(0, A.shape[0]):
        fid_list[i] = loc_fidelity(A[i,:,:], CH[i,:,:], B[i,:,:])
    av_loc_fid = 100*np.sum(fid_list)/A.shape[0]
    return av_loc_fid, fid_list

@performance_loc_fidelity.register
def _(A: torch.Tensor, channels: torch.Tensor, B: torch.Tensor) -> Union[torch.Tensor, float]:
    A = torch.squeeze(A)
    B = torch.squeeze(B)
    CH = torch.squeeze(channels)
    # keep outputs on the same device as inputs to avoid device mismatch
    fid_list = torch.zeros((A.shape[0]), device=A.device, dtype=torch.float32)
    for i in range(0, A.shape[0]):
        fid_list[i] = loc_fidelity(A[i,:,:], CH[i,:,:], B[i,:,:])
    av_loc_fid = 100*torch.sum(fid_list)/A.shape[0]
    return av_loc_fid, fid_list




@singledispatch
def performance_efficiency(A: Union[np.ndarray, torch.Tensor], channels: Union[np.ndarray, torch.Tensor]) -> Union[np.ndarray, torch.Tensor]:
    raise NotImplementedError(f"Cannot check efficiency of `A` for {type(A)}")

@performance_efficiency.register
def _(A: np.ndarray, channels: np.ndarray) -> Union[np.ndarray, float]:
    A = np.squeeze(A)
    CH = np.squeeze(channels)
    eff_list = np.zeros((A.shape[0]))
    for i in range(0, A.shape[0]):
        eff_list[i] = np.sum(A[i,:,:]*CH[i,:,:])
    av_eff = 100*np.sum(eff_list)/A.shape[0]
    return av_eff, eff_list

@performance_efficiency.register
def _(A: torch.Tensor, channels: torch.Tensor) -> Union[torch.Tensor, float]:
    A = torch.squeeze(A)
    CH = torch.squeeze(channels)
    # allocate on same device/dtype as inputs
    eff_list = torch.zeros((A.shape[0]), device=A.device, dtype=A.dtype)
    for i in range(0, A.shape[0]):
        eff_list[i] = torch.sum(A[i,:,:]*CH[i,:,:])
    av_eff = 100*torch.sum(eff_list)/A.shape[0]
    return av_eff, eff_list



@singledispatch
def performance_crosstalk(A: Union[np.ndarray, torch.Tensor], channels: Union[np.ndarray, torch.Tensor]) -> Union[np.ndarray, torch.Tensor]:
    raise NotImplementedError(f"Cannot check cross-talk for {type(A)}")

@performance_crosstalk.register
def _(A: np.ndarray, channels: np.ndarray) -> Union[np.ndarray, np.ndarray, float]:
    A = np.squeeze(A)
    CH = np.squeeze(channels)
    crs_list = np.zeros((A.shape[0]))
    crs_matrix = np.zeros((A.shape[0],A.shape[0]))
    for i in range(0, A.shape[0]):
        for j in range(0, A.shape[0]):
            crs_matrix[i,j] = np.sum(A[j,:,:]*CH[i,:,:])
    for i in range(0, A.shape[0]): 
        crs_list[i] = 1 - (crs_matrix[i,i]/np.sum(crs_matrix[:,i]))
    av_crs = 100*np.sum(crs_list)/A.shape[0]
    return av_crs, crs_list, crs_matrix

@performance_crosstalk.register
def _(A: torch.Tensor, channels: torch.Tensor) -> Union[torch.Tensor, torch.Tensor, float]:
    A = torch.squeeze(A)
    CH = torch.squeeze(channels)
    # allocate on same device/dtype as inputs
    crs_list = torch.zeros((A.shape[0]), device=A.device, dtype=A.dtype)
    crs_matrix = torch.zeros((A.shape[0],A.shape[0]), device=A.device, dtype=A.dtype)
    for i in range(0, A.shape[0]):
        for j in range(0, A.shape[0]):
            crs_matrix[i,j] = torch.sum(A[j,:,:]*CH[i,:,:])
    for i in range(0, A.shape[0]): 
        crs_list[i] = 1 - (crs_matrix[i,i]/torch.sum(crs_matrix[:,i]))
    av_crs = 100*torch.sum(crs_list)/A.shape[0]
    return av_crs, crs_list, crs_matrix







@singledispatch
def complim(x: Union[np.ndarray, torch.Tensor]):
    # visualize a complex field. brightness = amplitude, colour = phase
    raise NotImplementedError(
        f"Cannot visualize `x` for type: {type(x)}")

@complim.register
def _(x: np.ndarray) -> np.ndarray:
    mAx = np.amax(np.abs(x))
    M = x/mAx
    A = np.abs(M)
    P = np.angle(M)
    A[A > 1.] = 1.

    R = A*(np.cos(P - 2*pi/3)/2+0.5)
    G = A*(np.cos(P)/2+0.5)
    B = A*(np.cos(P + 2*pi/3)/2+0.5)
    
    C = np.dstack((R, G, B))
    plt.imshow(C)
    plt.show() 

@complim.register
def _(x: torch.Tensor) -> torch.Tensor:
    mAx = torch.amax(torch.abs(x))
    M = x/mAx
    A = torch.abs(M)
    P = torch.angle(M)
    A[A > 1.] = 1.

    R = A*(torch.cos(P - 2*pi/3)/2+0.5)
    G = A*(torch.cos(P)/2+0.5)
    B = A*(torch.cos(P + 2*pi/3)/2+0.5)
    
    C = torch.dstack((R, G, B))
    plt.imshow(C.detach().cpu().numpy())
    plt.show()    




@singledispatch
def plot_in_GS(x: Union[np.ndarray, torch.Tensor]):
    # visualize a 2D phase distribution in gray scale (8 bit)
    raise NotImplementedError(
        f"Cannot visualize `x` for type: {type(x)}")

@plot_in_GS.register
def _(x: np.ndarray) -> np.ndarray:
    x = np.angle(np.exp(1j*x))
    plt.imshow(x, cmap="gray")
    plt.show()    

@plot_in_GS.register
def _(x: torch.Tensor) -> torch.Tensor:
    x = torch.angle(torch.exp(1j*x))
    plt.imshow(x.detach().cpu().numpy(), cmap="gray")
    plt.show()    




@singledispatch
def complim_subplot2(x: Union[np.ndarray, torch.Tensor]):
    # visualize two complex fields side by side. brightness = amplitude, colour = phase
    raise NotImplementedError(
        f"Cannot visualize `x` for type: {type(x)}")

@complim_subplot2.register
def _(x: np.ndarray, y: np.ndarray, titles: list) -> np.ndarray:
    mAx = np.amax(np.abs(x))
    M = x/mAx
    A = np.abs(M)
    P = np.angle(M)
    A[A > 1.] = 1.

    R = A*(np.cos(P - 2*pi/3)/2+0.5)
    G = A*(np.cos(P)/2+0.5)
    B = A*(np.cos(P + 2*pi/3)/2+0.5)
    
    C1 = np.dstack((R, G, B))

    mAx = np.amax(np.abs(y))
    M = y/mAx
    A = np.abs(M)
    P = np.angle(M)
    A[A > 1.] = 1.

    R = A*(np.cos(P - 2*pi/3)/2+0.5)
    G = A*(np.cos(P)/2+0.5)
    B = A*(np.cos(P + 2*pi/3)/2+0.5)
    
    C2 = np.dstack((R, G, B))

    C = [C1, C2]
    fig, axs = plt.subplots(1, 2)
    i = 0
    for ax, interp in zip(axs, titles):
        ax.imshow(C[i])
        ax.set_title(interp, fontsize=10)
        i = i+1
    plt.show()

@complim_subplot2.register
def _(x: torch.Tensor, y: torch.Tensor, titles: list) -> torch.Tensor:
    mAx = torch.amax(torch.abs(x))
    M = x/mAx
    A = torch.abs(M)
    P = torch.angle(M)
    A[A > 1.] = 1.

    R = A*(torch.cos(P - 2*pi/3)/2+0.5)
    G = A*(torch.cos(P)/2+0.5)
    B = A*(torch.cos(P + 2*pi/3)/2+0.5)
    
    C1 = torch.dstack((R, G, B))

    mAx = torch.amax(torch.abs(y))
    M = y/mAx
    A = torch.abs(M)
    P = torch.angle(M)
    A[A > 1.] = 1.

    R = A*(torch.cos(P - 2*pi/3)/2+0.5)
    G = A*(torch.cos(P)/2+0.5)
    B = A*(torch.cos(P + 2*pi/3)/2+0.5)
    
    C2 = torch.dstack((R, G, B))

    C = [C1, C2]
    fig, axs = plt.subplots(1, 2)
    i = 0
    for ax, interp in zip(axs, titles):
        ax.imshow(C[i].detach().cpu().numpy())
        ax.set_title(interp, fontsize=10)
        i = i+1
    plt.show()