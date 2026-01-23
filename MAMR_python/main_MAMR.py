import numpy as np

from MAMR_Original import MAMR
from MAM_Restriction_on_capa import MAMR_RC
from MAM_FROB_RESTRICTION import MAMR_RC_F 
from MAM_Restriction_on_bray import MAMR_RC_B
from MAM_Restriction_on_components import MAMR_RC_C

from distGrid import distGrid
from salvaPNG import salvaPNG
import matplotlib.pyplot as plt
import os
os.chdir(os.path.dirname(__file__))

def main(res='o', u_vect=None, u_sca = 1, tau = 1, T = None):
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
    if res == 'c':
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
            u=u_sca
        )
    elif res == 'f':
        print("Running MAM-R with Frobenius norm regularization...")
        p, _, _, _ = MAMR_RC_F(
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
        )
    elif res == 'b':
        print("Running MAM-R with restriction on capacity on barycenter...")
        p, _, _, _ = MAMR_RC_B(
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
            u=u_vect
        )
    elif res == 'm':
        print("Running MAM-R with restriction on components...")
        p, _, _, _ = MAMR_RC_C(
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
            T=T
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

    if res == 'c':
        plt.title(f"MAM-R with capacity u={u}")
        plt.figure()
        plt.imshow(img, cmap="hot")
        plt.colorbar()
        plt.title(f"MAM-R Barycenter capacity u={u}")
        salvaPNG(plt.gcf(), f"Final-MAMR_Cap_u={u}.png", outputFolder=r"Fig\capacity")
        plt.show()
    elif res == 'f':
        plt.title(f"MAM-R with Frobenius norm regularization tau={tau}")
        plt.figure()
        plt.imshow(img, cmap="hot")
        plt.colorbar()
        plt.title(f"MAM-R Barycenter Frobenius tau={tau}")
        salvaPNG(plt.gcf(), f"Final-MAMR_Frob_tau={tau}.png",outputFolder=r"Fig\frobenius")
        plt.show()
    elif res == 'b':
        plt.title("MAM-R with capacity on barycenter")
        plt.figure()
        plt.imshow(img, cmap="hot")
        plt.colorbar()
        plt.title("MAM-R Barycenter with capacity")
        salvaPNG(plt.gcf(), "Final-MAMR_Cap_on_barycenter.png",outputFolder=r"Fig\capacity_on_barycenter")
        plt.show()
    elif res == 'm':
        plt.title("MAM-R with restriction on components")
        plt.figure()
        plt.imshow(img, cmap="hot")
        plt.colorbar()
        plt.title("MAM-R Barycenter with restriction on components")
        salvaPNG(plt.gcf(), "Final-MAMR_Restriction_on_components.png",outputFolder=r"Fig\restriction_on_components")
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
    res = input("Enter c fot capacity on plans, " \
    "f for Frobenius on plans, " \
    "b for capacity on barycenter, m for restriction on components, " \
    "n for none (c/f/b/m/n): ").lower()
    
    if res == 'c':
        list = input("Run for multiple u values? (y/n): ").lower() == 'y'
        if not list:
            u_value = float(input("Enter the value of u (between 0 and 1): "))
            main(res='c', u_sca=u_value)
        else:
            u_values = [1, 0.75, 0.5, 0.25, 0.1,0.01,0.001,0.0001]
            print("Running MAM-R with capacity constraints for different u values:", u_values)

            for u_value in u_values:
                print(f"\n{'='*50}")
                print(f"Running with u = {u_value}")
                print(f"{'='*50}\n")
                main(res='c', u_sca=u_value)
    elif res == 'f':
        list = input("Run for multiple tau values? (y/n): ").lower() == 'y'
        if list:
            tau_values = [0.5, 0.25, 0.1, 0.01, 0.001, 0.0001, 0.00001, 5.8556e-06]
            print("Running MAM-R with Frobenius norm regularization for different tau values:", tau_values)

            for tau_value in tau_values:
                print(f"\n{'='*50}")
                print(f"Running with tau = {tau_value}")
                print(f"{'='*50}\n")
                main(res='f', tau=tau_value)
        else:
            tau_value = float(input("Enter the value of tau (positive number): "))
            main(res='f', tau=tau_value)
    elif res == 'b':
        u = input("Want to set a specific value for vector u (non recomended for large R)? (y/n): ").lower()
        if u == 'y':
            size_u = int(input("Enter the size of vector u (should be equal to R): "))
            u_vector = []
            print("Enter the elements of vector u (non-negative, sum at least 1):")
            for i in range(size_u):
                val = float(input(f"u[{i}]: "))
                u_vector.append(val)
            main(res='b', u_vect=u_vector)
        else:
            K = int(input("Enter the image size K (to compute R=K*K): "))
            R = K * K
            equal_u = 1.0 / R + 0.01  # Slightly above uniform to ensure sum(u) > 1
            u_vector = [equal_u] * R
            print(f"Using uniform capacity vector u with each element = {equal_u}")
            main(res='b', u_vect=u_vector)  
    elif res == 'm':
        M = int(input("Enter the number of marginals M: "))
        K = int(input("Enter the image size K (to compute R=K*K): "))
        R = K * K
        comp = input("Want to set restricted components manually? (y/n): ").lower()
        if comp == 'y':
            T = []
            for m in range(M):
                T_m = []
                components = input(f"Enter the restricted components for marginal {m+1} as pairs (i,j) separated by spaces (e.g., '0,1 2,3')")
                if components.strip():
                    pairs = components.split()
                    for pair in pairs:
                        i, j = map(int, pair.split(','))
                        T_m.append((i, j))
                T.append(T_m)
            main(res='m', T=T)
        else:
            #Select random restricted components for of the M marginals
            #For simplicity, we restrict 5% of the components in each marginal
            #Each marginal has size S[m], we randomly select 5% of R*S[m] components to be restricted
            np.random.seed(0)
            T = [[] for _ in range(M)]

            #Geting the sizes S[m] from the data
            Q = np.zeros((K * K, M))
            for i in range(M):
                data = np.loadtxt(
                    os.path.join("dataPeyre", f"{i+1}.csv"),
                    delimiter=","
                )
                data[:, 2] /= data[:, 2].sum()
                im = np.zeros((K, K))
                for r, c, v in data:
                    r = int(r) - 1
                    c = max(int(c) - 1, 0)
                    im[r, c] = v
                Q[:, i] = im.reshape(-1)

            S = []
            for m in range(M):
                I = Q[:, m] > 1e-15
                S.append(I.sum())


            for m in range(M):
                num_restricted = int(0.05 * R * S[m])  # 5% restriction
                restricted_indices = np.random.choice(R * S[m], num_restricted, replace=False)
                for idx in restricted_indices:
                    i = idx // S[m]
                    j = idx % S[m]
                    T[m].append((i, j))

            main(res='m', T=T)
    else:
        main(res='o')