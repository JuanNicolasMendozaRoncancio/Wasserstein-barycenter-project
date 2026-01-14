import numpy as np
from MAMR_python.MAMR_Original import MAMR
from MAMR_python.distGrid import distGrid
from MAMR_python.salvaPNG import salvaPNG
import matplotlib.pyplot as plt
import os

def main():
    np.random.seed(0)

    # ===============================
    # Parameters
    # ===============================
    M = 25          # number of images
    K = 60          # image size K x K
    MaxCPU = 60     # seconds
    PrintEvery = 5  # seconds
    tol = -np.inf

    UseGPU = False  # CPU only
    rho = 350

    # ===============================
    # Distance matrix
    # ===============================
    print("Computing distance...")
    Kn = K
    R = K * K
    D = (distGrid(K, 1) ** 2) / (K * K)

    # ===============================
    # Read data
    # ===============================
    print("Reading data...")
    Q = np.zeros((K * K, M))

    for i in range(M):
        data = np.loadtxt(
            os.path.join("dataPeyre", f"{i+1}.csv"),
            delimiter=","
        )

        # Normalize mass
        data[:, 2] /= data[:, 2].sum()

        im = np.zeros((K, K))
        for r, c, v in data:
            r = int(r) - 1
            c = max(int(c) - 1, 0)
            im[r, c] = v

        Q[:, i] = im.reshape(-1)

        if i < 25:
            plt.subplot(5, 5, i + 1)
            plt.imshow(1 - im, cmap="hot")
            plt.axis("off")

    plt.show()

    # ===============================
    # Arrange data for MAM-R
    # ===============================
    print("Arranging data...")
    S = []
    q = []
    d = []

    p = np.ones(R) / R   # p0 ∈ H

    for m in range(M):
        I = Q[:, m] > 1e-15
        S.append(I.sum())
        d.append(D[:, I])
        qm = Q[I, m]
        q.append(qm / qm.sum())

    # ===============================
    # Run MAM-R
    # ===============================
    print("Running MAM-R...")
    p, _, _, _ = MAMR(
        d=d,
        q=q,
        M=M,
        R=R,
        S=S,
        p=p,
        rho=rho,
        UseGPU=UseGPU,
        tol=tol,
        MaxCPU=MaxCPU,
        PrintEvery=PrintEvery
    )

    # ===============================
    # Plot result
    # ===============================
    p = p.reshape(Kn, Kn)
    plt.imshow(1 - p, cmap="hot")
    plt.colorbar()
    plt.title("MAM-R Barycenter")
    plt.show()

    salvaPNG(plt.gcf(), "Final-MAMR.png")
