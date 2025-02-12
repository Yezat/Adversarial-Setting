from state_evolution.constants import INITIAL_CONDITION


def damped_update(new, old, damping):
    return damping * new + (1 - damping) * old


class Overlaps:  # TODO, shall this be a dataclass? Then we can improve also the log_overlaps method
    def __init__(self) -> None:
        self.m = INITIAL_CONDITION[0]
        self.q = INITIAL_CONDITION[1]
        self.sigma = INITIAL_CONDITION[2]
        self.A = INITIAL_CONDITION[3]
        self.P = INITIAL_CONDITION[4]
        self.F = INITIAL_CONDITION[5]

        self.m_hat = 0
        self.q_hat = 0
        self.sigma_hat = 0
        self.A_hat = 0
        self.F_hat = 0
        self.P_hat = 0

    def update_overlaps(self, m, q, sigma, A, P, F):
        self.n_m = damped_update(m, self.m, self.BLEND_FPE)
        self.n_q = damped_update(q, self.q, self.BLEND_FPE)
        self.n_sigma = damped_update(sigma, self.sigma, self.BLEND_FPE)
        self.n_A = damped_update(A, self.A, self.BLEND_FPE)
        self.n_P = damped_update(P, self.P, self.BLEND_FPE)
        self.n_F = damped_update(F, self.F, self.BLEND_FPE)

        # Compute the error
        err = max(
            [
                abs(self.n_m - self.m),
                abs(self.n_q - self.q),
                abs(self.n_sigma - self.sigma),
                abs(self.n_A - self.A),
                abs(self.n_P - self.P),
                abs(self.n_F - self.F),
            ]
        )

        # Update the overlaps
        self.m = self.n_m
        self.q = self.n_q
        self.sigma = self.n_sigma
        self.A = self.n_A
        self.P = self.n_P
        self.F = self.n_F

        return err
