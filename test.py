import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Ellipse

def supergaussian_window(shape, rx, ry, n=4, center=None):
    H, W = shape
    if center is None:
        cx, cy = W/2, H/2
    else:
        cx, cy = center
    x = (np.arange(W) - cx)[None, :]
    y = (np.arange(H) - cy)[:, None]
    X = (x / rx)**(2*n)
    Y = (y / ry)**(2*n)
    Wmat = np.exp(-(X + Y))
    return Wmat

# 例：计算窗口尺寸 512x256，设横纵比 2:1，选 normalized_width = 0.6
H, W = 512, 256
norm = 150/256
norm = 1.5*norm
# 这里我们把 rx 归一化到半宽 (W/2)，ry 归一化到半高 (H/2)
rx = norm * (W/2)
ry = norm * (H/2)
win = supergaussian_window((H, W), rx=rx, ry=ry, n=4)

plt.figure(figsize=(6,4))
plt.imshow(win, origin='lower')
# 画出 1/e 等值线（即椭圆半轴 rx,ry）
ax = plt.gca()
ell = Ellipse((W/2, H/2), width=2*rx, height=2*ry, edgecolor='white',
              facecolor='none', linestyle='--', linewidth=1.2)
ax.add_patch(ell)
plt.title(f"Super-Gaussian window (n=4), norm={norm}")
plt.colorbar(label='W')
plt.show()
