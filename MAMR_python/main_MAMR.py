import numpy as np
from MAMR_Original import MAMR
from MAM_Restriction_on_capa import MAMR_RC
from MAM_FROB_RESTRICTION import MAMR_RC as MAMR_FROB_RC
from distGrid import distGrid
from salvaPNG import salvaPNG
import matplotlib.pyplot as plt
import os
os.chdir(os.path.dirname(__file__))

def main(capacity=False, frobenius=False, u = 1, tau = 1):
    np.random.seed(0)

    # ===============================
    # Parameters
    # ===============================
    M = 30          # number of images
    K =60         # image size K x K    
    MaxCPU = 600   # seconds
    PrintEvery = 10  # seconds
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
    if capacity:
        print("Running MAM-R with restriction on capacity...")
        p, _, _, _ = MAMR_RC(
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
            PrintEvery=PrintEvery,
            u=u
        )
    elif frobenius:
        print("Running MAM-R with Frobenius norm regularization...")
        p, _, _, _ = MAMR_FROB_RC(
            d=d,
            tau=tau,
            q=q,
            M=M,
            R=R,
            S=S,
            p=p,
            rho=rho,
            UseGPU=UseGPU,
            tol=tol,
            MaxCPU=MaxCPU,
            PrintEvery=PrintEvery,
            u=u
        )
    else:
        print("Running MAM-R original...")
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

    print("p min:", p.min())
    print("p max:", p.max())
    print("p sum:", p.sum())
    
    p = p.reshape(Kn, Kn)
    img = 1 - p
    img = (img - img.min()) / (img.max() - img.min())

    if capacity:
        plt.title(f"MAM-R with capacity u={u}")
        plt.figure()
        plt.imshow(img, cmap="hot")
        plt.colorbar()
        plt.title(f"MAM-R Barycenter capacity u={u}")
        salvaPNG(plt.gcf(), f"Final-MAMR_Cap_u={u}.png", outputFolder=r"Fig\capacity")
        plt.show()
    elif frobenius:
        plt.title(f"MAM-R with Frobenius norm regularization tau={tau}")
        plt.figure()
        plt.imshow(img, cmap="hot")
        plt.colorbar()
        plt.title(f"MAM-R Barycenter Frobenius tau={tau}")
        salvaPNG(plt.gcf(), f"Final-MAMR_Frob_tau={tau}.png",outputFolder=r"Fig\frobenius")
        plt.show()
    else:
        plt.title("MAM-R Original")
        plt.figure()
        plt.imshow(img, cmap="hot")
        plt.colorbar()
        plt.title("MAM-R Barycenter")
        salvaPNG(plt.gcf(), "Final-MAMR_Original.png",outputFolder=r"Fig\originals")
        plt.show()


if __name__ == "__main__":
    cap = input("Use capacity constraint? (y/n): ").lower() == 'y'
    capf = input("Use Frobenius norm regularization? (y/n): ").lower() == 'y'
    if cap:
        list = input("Run for multiple u values? (y/n): ").lower() == 'y'
        if not list:
            u_value = float(input("Enter the value of u (between 0 and 1): "))
            main(capacity=True, u=u_value)
        else:
            u_values = [1, 0.75, 0.5, 0.25, 0.1,0.01,0.001,0.0001]
            print("Running MAM-R with capacity constraints for different u values:", u_values)

            for u_value in u_values:
                print(f"\n{'='*50}")
                print(f"Running with u = {u_value}")
                print(f"{'='*50}\n")
                main(capacity=True, u=u_value)
    elif capf:
        list = input("Run for multiple tau values? (y/n): ").lower() == 'y'
        if list:
            tau_values = [0.5, 0.25, 0.1, 0.01, 0.001, 0.0001, 0.00001, 5.8556e-06]
            print("Running MAM-R with Frobenius norm regularization for different tau values:", tau_values)

            for tau_value in tau_values:
                print(f"\n{'='*50}")
                print(f"Running with tau = {tau_value}")
                print(f"{'='*50}\n")
                main(frobenius=True, tau=tau_value)
        else:
            tau_value = float(input("Enter the value of tau (positive number): "))
            main(frobenius=True, tau=tau_value)
    else:
        main(capacity=False)