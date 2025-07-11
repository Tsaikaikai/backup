import numpy as np

AOI = np.array([1.15, 1.143, 1.152, 1.145, 1.151])
edge = np.array([1.1448, 1.144, 1.1445, 1.1439, 1.1439])

# 擬合 edge = alpha * AOI + beta
A = np.vstack([AOI, np.ones_like(AOI)]).T
alpha, beta = np.linalg.lstsq(A, edge, rcond=None)[0]

print(f"alpha = {alpha:.6f}")
print(f"beta = {beta:.6f}")

有一組AOI數據1.15 1.143 1.152 1.145 1.151 有另一組edge數據 1.1448 1.144 1.1445 1.1439 1.1439 想利用edge = AOI*alpha + beta，進行擬合求出最佳的alpha以及beta為何?