import numpy as np
from scipy.spatial.distance import cdist

def distGrid(K, M):
    """
    Computes the Euclidean distance between two grids:
    - Fine grid with step 1/M
    - Coarse grid with step 1
    """

    # Fine grid (barycenter support)
    x = np.arange(1, K + 1 / M, 1 / M)
    X, Y = np.meshgrid(x, x)
    ptsK = np.column_stack((X.ravel(), Y.ravel()))

    # Coarse grid (input images)
    x = np.arange(1, K + 1, 1)
    X, Y = np.meshgrid(x, x)
    ptsk = np.column_stack((X.ravel(), Y.ravel()))

    # Pairwise Euclidean distances
    d = cdist(ptsK, ptsk, metric='euclidean')

    return d
