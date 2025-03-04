from enum import Enum

BLEND_FPE = 0.85
TOL_FPE = 1e-5
MIN_ITER_FPE = 10
MAX_ITER_FPE = 10000
INT_LIMS = 7.5
INITIAL_CONDITION = 1e-1

OVERLAPS = ["m", "q", "sigma", "A", "P", "F"]
HAT_OVERLAPS = ["m_hat", "q_hat", "sigma_hat", "A_hat", "F_hat", "P_hat"]

OVERLAPS_FGM = ["m", "q", "sigma", "A", "P", "F", "N"]
HAT_OVERLAPS_FGM = ["m_hat", "q_hat", "sigma_hat", "A_hat", "F_hat", "P_hat", "N_hat"]


class SEProblemType(Enum):
    Logistic = 0
    LogisticFGM = 1
