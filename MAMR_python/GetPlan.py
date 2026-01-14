import numpy as np

def GetPlan(p, q, R, S):
    p = np.asarray(p)
    q = np.asarray(q)

    pi = (1 / S) * np.tile(p, (1, S)) \
       + (1 / R) * np.tile(q.reshape(-1, 1), (R, 1)) \
       - 1 / (S * R)

    return pi
