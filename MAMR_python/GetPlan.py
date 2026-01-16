import numpy as np

def GetPlan(p, q, R, S):
    """
    Python translation of:
    pi = (1/S)*repmat(p,1,S) + (1/R)*repmat(q',R,1) - 1/(S*R)
    """

    p = p.reshape(R, 1)      # (R,1)
    q = q.reshape(1, S)      # (1,S)

    pi = (1 / S) * p + (1 / R) * q - 1 / (S * R)
    return pi