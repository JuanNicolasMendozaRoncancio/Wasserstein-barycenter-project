import numpy as np
import time
import GetPlan as gp
from GetPlan import GetPlan
import matplotlib.pyplot as plt
from salvaPNG import salvaPNG

def MAMR_RC_F(d, tau,  q, M, R, S, p, rho, tol, MaxCPU,filename, folder, PrintEvery,UseGPU = False):
    """
    Multi-Marginal Algorithm with Restriction on Capacity (MAM-RC)
    Solves the multi-marginal problem with capacity constraints using
    a variant of the MAM-R algorithm.
    Inputs:
    - d: list of M cost matrices (each of size R x S[m])
    - tau: Frobenius norm regularization parameter
    - q: list of M marginal distributions (each of size S[m])
    - M: number of marginals
    - R: size of the primary space
    - S: list of sizes of the secondary spaces
    - p: initial distribution (size R)
    - rho: regularization parameter
    - tol: tolerance for convergence
    - MaxCPU: maximum CPU time
    - PrintEvery: time interval for printing progress
    - UseGPU: boolean flag to use GPU acceleration
    Outputs:
    - p: computed barycenter distribution (size R)
    - val: final objective value
    - cpu: total CPU time taken
    - theta: list of transport plans (each of size R x S[m])

    """

    #Verift if tau is valid, tau should be greater than max 1/(R*S[m])
    if tau >= 1:
        raise ValueError("tau should be less than 1")

    mx_rs = -np.inf
    for m in range(M):
        mx_rs = max(mx_rs, 1/(R*S[m]))
    
    print(f"max 1/(R*S[m]) = {mx_rs}")
    if tau < mx_rs:
        raise ValueError(f"tau should be greater than max 1/(R*S[m]) = {mx_rs}")


    # Optional GPU support
    UseGPU = False
    xp = np

    t0 = time.time()
    Kn = int(round(np.sqrt(R)))

    # Initialization
    theta = [None] * M # transport plans
    pk = [None] * M # marginals

    p = xp.asarray(p)

    avg = xp.zeros(R) # average marginal

    # weights a_m = (1/S_m) / sum_j (1/S_j)
    a = 1.0 / xp.asarray(S)
    a = a / a.sum()
    u = float(u)


    #Romero
    for m in range(M):
        theta[m] = GetPlan(p, q[m], R, S[m])
        pk[m] = p.copy()
        d[m] = d[m] / rho   # scale cost

    avg = p.copy()
    val = 0.0

    # Main loop
    NextPrint = 0.0
    cpu = 0.0
    k = 0

    # plt.figure()
    # img = xp.reshape(1 - p, (Kn, Kn))
    # img_np = xp.asnumpy(img) if UseGPU else img

    # im_handle = plt.imshow(img_np, cmap='hot')
    # cbar = plt.colorbar(im_handle)
    # plt.title("MAM-R")
    # plt.pause(0.01)

    while cpu <= MaxCPU:
        k += 1
        cpu = time.time() - t0

        if cpu >= NextPrint:

            # Projection onto H (X = R^R)
            proj = avg - (avg.sum() - 1.0) / R
            nx = xp.linalg.norm(p - proj)

            print(f"k = {k:5d}, |pk-pkk| = {nx:5.2e}, cpu = {cpu:5.0f}")

            # img = xp.reshape(1 - p, (Kn, Kn))

            # img_np = xp.asnumpy(img) if UseGPU else img
            # plt.imshow(img_np, cmap='hot')

            # plt.title(f"k={k}, t={NextPrint}")
            # plt.colorbar()
            # plt.show(block=False)
            # plt.pause(0.01)

            # img = xp.reshape(1 - p, (Kn, Kn))
            # img_np = xp.asnumpy(img) if UseGPU else img

            # im_handle.set_data(img_np)
            # im_handle.set_clim(img_np.min(), img_np.max())
            # plt.title(f"k={k}, t={round(cpu)}")
            # plt.pause(0.01)
            
            if nx <= tol:
                break

            NextPrint += PrintEvery

        # Projection onto H
        p = avg - (avg.sum() - 1.0) / R

        val = 0.0
        avg = xp.zeros(R)

        for m in range(M):
            # #Theta update
            qm = theta[m].sum(axis=0)


            #Romeros
            beta = (
                (p - pk[m])[:, None] / S[m]
                + (q[m] - qm)[None, :] / R
                + (qm.sum() - 1.0) / (R * S[m])
            )

            coef_m = tau/np.maximum(tau, np.linalg.norm(np.maximum(0,theta[m] + 2*beta - d[m]), 'fro'))

            #Update without simplex   
            theta[m] = coef_m*np.maximum(0,theta[m]+ 2*beta - d[m]) - beta 

            ## Bien NO TOCAR

            # # First proximal opp
            # pi = theta[m] + beta

            # #Second proximal opp
            # coef_m = tau/np.maximum(tau, np.linalg.norm(np.maximum(0,2*pi - theta[m] - d[m]), 'fro'))
            # pi_hat = coef_m * np.maximum(0,2*pi - theta[m] - d[m])

            # #Update
            # theta[m] += pi_hat - pi

            #Marginal update
            pk[m] = theta[m].sum(axis=1)
            avg += a[m] * pk[m]

            cpu_values = [1.0 * MaxCPU]
            for cv in cpu_values:
                if cpu >= cv and cpu < cv + PrintEvery:
                    p_img = p.reshape(Kn, Kn)
                    img = 1 - p_img
                    img = (img - img.min()) / (img.max() - img.min())

                    fig, ax = plt.subplots(figsize=(4, 4))

                    ax.imshow(img, cmap="gray")
                    ax.axis("off")              # ← quita ejes
                    ax.set_position([0, 0, 1, 1])  # ← ocupa toda la figura

                    filename = f"{filename}_barycenter_cap_{int(cv/MaxCPU*100)}.png"
                    salvaPNG(
                        fig=fig,
                        filename=filename,
                        outputFolder=folder
                    )
                    plt.close(fig)

    cpu = time.time() - t0
    print(f"k = {k:5d}, |pk-pkk| = {nx:5.2e}, cpu = {cpu:5.0f}")

    # img = xp.reshape(1 - p, (Kn, Kn))
    # img_np = xp.asnumpy(img) if UseGPU else img

    # plt.figure()
    # plt.imshow(img_np, cmap='hot')
    # plt.colorbar()
    # plt.title(f"Final k={k}, t={round(cpu)}")
    # plt.show()

    return p, val, cpu, theta