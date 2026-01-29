import numpy as np
import time
import GetPlan as gp
from GetPlan import GetPlan
import matplotlib.pyplot as plt
from salvaPNG import salvaPNG

def MAMR_RC(d, q, M, R, S, p, rho, tol, MaxCPU, PrintEvery,u,folder,filename,UseGPU = False):
    """
    Multi-Marginal Algorithm with Restriction on Capacity (MAM-RC)
    Solves the multi-marginal problem with capacity constraints using
    a variant of the MAM-R algorithm.
    Inputs:
    - d: list of M cost matrices (each of size R x S[m])
    - q: list of M marginal distributions (each of size S[m])
    - M: number of marginals
    - R: size of the primary space
    - S: list of sizes of the secondary spaces
    - p: initial distribution (size R)
    - rho: regularization parameter
    - tol: tolerance for convergence
    - MaxCPU: maximum CPU time
    - PrintEvery: time interval for printing progress
    - u: capacity constraint
    - UseGPU: boolean flag to use GPU acceleration
    Outputs:
    - p: computed barycenter distribution (size R)
    - val: final objective value
    - cpu: total CPU time taken
    - theta: list of transport plans (each of size R x S[m])

    """

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

    # Lo comentamos
    # ==========================================
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
            #Theta update
            qm = theta[m].sum(axis=0)

            #Romeros
            beta = (
                (p - pk[m])[:, None] / S[m]
                + (q[m] - qm)[None, :] / R
                + (qm.sum() - 1.0) / (R * S[m])
            )

            #Update without simplex
            theta[m] = xp.minimum(xp.maximum(theta[m] + beta - d[m], -beta),u - beta)

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

                    ax.imshow(img, cmap="viridis")
                    ax.axis("off")              # ← quita ejes
                    ax.set_position([0, 0, 1, 1])  # ← ocupa toda la figura

                    out_filename = f"{filename}_barycenter_cap_{int(cv/MaxCPU*100)}.png"
                    salvaPNG(
                        fig=fig,
                        filename=out_filename,
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
