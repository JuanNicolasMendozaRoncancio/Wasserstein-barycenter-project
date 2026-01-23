import numpy as np

def proj_restric_p(A,u):
    """
    Projection onto C = {Y: Y >= 0, Y 1 <= u}

    Parameters:
        A: matrix to be projected
        u: upper bound vector

    Returns:
        Proj_A: projected matrix
    """
    proj_A = np.zeros_like(A)

    for r in range(A.shape[0]):
        row = np.maximum(A[r,:], 0)

        if row.sum() <= u[r]:
            proj_A[r,:] = row
        else:
            sorted_ = np.sort(row)[::-1]
            cum_sum = np.cumsum(sorted_)
            idx = np.where(sorted_ > (cum_sum - u[r]) / (np.arange(len(sorted_)) + 1))[0]
            k = idx[-1]
            lamb = (cum_sum[k] - u[r]) / (k + 1)
            proj_A[r,:] = np.maximum(A[r,:] - lamb, 0)

    return proj_A

        