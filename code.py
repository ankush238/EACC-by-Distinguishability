import itertools
import time

import cvxpy as cp
import numpy as np

# ---------------------------------------------------------------- parameters
THETA = np.pi / 3        # fixes the task, Eq. [bell]
D0 = 0.4                 # distinguishability bound D <= D0
TARGET = 2.2290          # bound to certify for phi+
I2 = np.eye(2)
Z2 = np.zeros((2, 2))
SX = np.array([[0, 1], [1, 0]], dtype=complex)
SY = np.array([[0, -1j], [1j, 0]], dtype=complex)
SZ = np.array([[1, 0], [0, -1]], dtype=complex)
INPUTS = ["00", "01", "10", "11"]   # index k = 2a + x


# ===================================================== 1. weights, Eq. [bell]
def task_weights(theta):
    """W[k, y, b] = coefficient of p(b|ax,y) in Eq. [bell]:
       W(ax,y,b) = (-1)^(a+b+xy) beta_ax + delta_x0 delta_y0 alpha beta_a0 (-1)^b."""
    mu = np.arccos(1 / np.sqrt(1 + np.sin(2 * theta) ** 2))
    alpha = 2 / np.sqrt(1 + 2 * np.tan(2 * theta) ** 2)
    b00 = np.cos(mu) * np.cos(theta) ** 2 - np.cos(mu) / 2 + 0.5
    b01 = -np.sin(mu) * np.cos(theta) ** 2 + np.sin(mu) / 2 + 0.5
    beta = {(0, 0): b00, (1, 0): 1 - b00, (0, 1): b01, (1, 1): 1 - b01}
    W = np.zeros((4, 2, 2))
    for a, x, y, b in itertools.product(range(2), repeat=4):
        w = (-1) ** (a + b + x * y) * beta[(a, x)]
        if x == 0 and y == 0:
            w += alpha * beta[(a, 0)] * (-1) ** b
        W[2 * a + x, y, b] = w
    return W, beta


W, BETA = task_weights(THETA)


def cost_operators(N):
    """C_{ax,m} = sum_y [W(ax,y,0) N^{my}_0 + W(ax,y,1) N^{my}_1], Eq. [S_sigma].
       N[m][y] is Bob's effect N^{my}_0 for output b = 0."""
    return [[sum(W[k, y, 0] * N[m][y] + W[k, y, 1] * (I2 - N[m][y]) for y in range(2))
             for m in range(2)] for k in range(4)]


# ============================== 2. Bob's cases, Eqs. [bobcases], [eta_basis]
# For each y, N^{0y}_0 in {0, |e+><e+|, I} and N^{1y}_0 in {0, |e-><e-|, I}.
OPTIONS = [(o0, o1) for o0 in ("0", "P", "I") for o1 in ("0", "P", "I")]   # 9 per y


def proj(v):
    return np.outer(v, v.conj())


def basis(y, eta):
    """{|e^y_+>, |e^y_->}: computational basis for y=0, Eq. [eta_basis] for y=1."""
    if y == 0:
        return proj(np.array([1.0, 0.0])), proj(np.array([0.0, 1.0]))
    e_plus = np.array([np.cos(eta / 2), np.sin(eta / 2)])
    e_minus = np.array([np.sin(eta / 2), -np.cos(eta / 2)])
    return proj(e_plus), proj(e_minus)


def bob_strategy(case, eta):
    """N[m][y] for one of the 81 cases (opt_y0, opt_y1) and angle eta."""
    N = [[None, None], [None, None]]
    for y in range(2):
        Pp, Pm = basis(y, eta)
        o0, o1 = case[y]
        N[0][y] = {"0": Z2, "P": Pp, "I": I2}[o0]   # message m = 0
        N[1][y] = {"0": Z2, "P": Pm, "I": I2}[o1]   # message m = 1
    return N


def depends_on_eta(case):
    """F depends on eta only if both measurement pairs contain a rank-one effect."""
    return "P" in case[0] and "P" in case[1]


# =============== 3. Alice's exact optimum for fixed Bob, Eq. [alice_sdp]
class AliceSDP:
    """F = max sum_ax [Tr(tau C_{ax,0}) + Tr((I/2 - tau) C_{ax,1})]
       s.t. 0 <= tau_ax <= I/2,  Y_0 >= tau_ax,  Y_1 >= I/2 - tau_ax,
            Tr(Y_0 + Y_1) <= 4 D0.                        Eqs. [Dconstraint], [alice_sdp]
       The problem is built once; only the operators C change between solves."""

    def __init__(self, D0):
        self.tau = [cp.Variable((2, 2), hermitian=True) for _ in range(4)]
        self.Y = [cp.Variable((2, 2), hermitian=True) for _ in range(2)]
        self.C = [[cp.Parameter((2, 2), symmetric=True) for _ in range(2)] for _ in range(4)]
        cons = []
        for t in self.tau:
            cons += [t >> 0, I2 / 2 - t >> 0,                 # 0 <= tau <= I/2
                     self.Y[0] - t >> 0,                      # Y_0 >= sigma_{0|ax}
                     self.Y[1] - (I2 / 2 - t) >> 0]           # Y_1 >= sigma_{1|ax}
        self.trace_con = cp.real(cp.trace(self.Y[0] + self.Y[1])) <= 4 * D0
        cons.append(self.trace_con)
        obj = sum(cp.real(cp.trace(self.C[k][0] @ self.tau[k]))
                  + cp.real(cp.trace(self.C[k][1] @ (I2 / 2 - self.tau[k])))
                  for k in range(4))
        self.problem = cp.Problem(cp.Maximize(obj), cons)

    def solve(self, N):
        C = cost_operators(N)
        for k in range(4):
            for m in range(2):
                self.C[k][m].value = np.real(C[k][m])
        self.problem.solve(solver=cp.CLARABEL)
        return self.problem.value

    def nu(self):
        """Multiplier of the trace constraint: nu in Eq. [alice_dual]."""
        return float(self.trace_con.dual_value)


# ================== 4. cases excluded without the constraint, Eq. [free_bound]
def trace_positive_part(H):
    ev = np.linalg.eigvalsh((H + H.conj().T) / 2)
    return ev[ev > 0].sum()


def free_bound(N):
    """nu = 0 in Eq. [alice_dual]:
       F <= sum_ax [ Tr(C_{ax,1})/2 + Tr(C_{ax,0} - C_{ax,1})_+ / 2 ]."""
    C = cost_operators(N)
    return sum(np.real(np.trace(C[k][1])) / 2 + trace_positive_part(C[k][0] - C[k][1]) / 2
               for k in range(4))


# ======================== 5. Lipschitz constant, Eq. [Lconst]
L = 0.5 * sum(abs(W[k, 1, 0] - W[k, 1, 1]) for k in range(4))   # = sum beta_ax = 2


def certify_phiplus():
    alice = AliceSDP(D0)
    upper, best, best_where = -np.inf, -np.inf, None
    easy, hard = [], []

    # --- step 4: split the 81 cases using the unconstrained bound
    grid = np.linspace(0, 2 * np.pi, 4001)
    h = grid[1] - grid[0]
    for case in itertools.product(OPTIONS, OPTIONS):
        if depends_on_eta(case):
            fb = max(free_bound(bob_strategy(case, e)) for e in grid) + L * h / 2
        else:
            fb = free_bound(bob_strategy(case, 0.0))
        (easy if fb < TARGET else hard).append((case, fb))
    easy_max = max(fb for _, fb in easy)
    upper = max(upper, easy_max)
    print(f"Cases excluded without the constraint: {len(easy)} "
          f"(bound <= {easy_max:.4f}); cases needing it: {len(hard)}")

    # --- step 5: adaptive Lipschitz scan, Eq. [lipbound], for the remaining cases
    n_solves = 0
    for case, _ in hard:
        if not depends_on_eta(case):
            v = alice.solve(bob_strategy(case, 0.0)); n_solves += 1
            upper, best = max(upper, v), max(best, v)
            continue

        def F(eta):
            nonlocal n_solves
            n_solves += 1
            return alice.solve(bob_strategy(case, eta))

        edges = np.linspace(0, 2 * np.pi, 65)
        values = [F(e) for e in edges]
        stack = [(edges[i], edges[i + 1], values[i], values[i + 1]) for i in range(64)]
        while stack:
            a, b, fa, fb = stack.pop()
            for e, f in ((a, fa), (b, fb)):
                if f > best:
                    best, best_where = f, (case, e)
            bound = max(fa, fb) + L * (b - a) / 2          # Eq. [lipbound]
            if bound < TARGET or b - a < 1e-4:
                upper = max(upper, bound)
                continue
            mid = (a + b) / 2
            fm = F(mid)
            stack += [(a, mid, fa, fm), (mid, b, fm, fb)]

    # --- step 6: multiplier nu* at the optimum, Eq. [alice_dual]
    case_star, eta_star = best_where
    alice.solve(bob_strategy(case_star, eta_star))
    return upper, best, case_star, eta_star, alice.nu(), n_solves


# ====== 7. explicit Psi strategy, Eqs. [alice_meas_psi]-[bob_meas_psi]
GAMMA = {"00": 0.5898, "01": -0.0460, "10": -0.3818, "11": 2.9746}
LAMBDA = 0.4416
XI = {(0, 0): 0.3873, (0, 1): 2.6276, (1, 0): -2.9468, (1, 1): -0.2821}


def phi(g):        # Eq. [phi_gamma]
    return np.array([np.cos(g / 2), np.sin(g / 2)])


def psi_state(theta):
    v = np.array([np.cos(theta), 0, 0, np.sin(theta)])
    return np.outer(v, v)


def bob_conditional(rho, A):
    """Tr_A[(A x I) rho]: Bob's unnormalized state."""
    R = rho.reshape(2, 2, 2, 2)                       # R[i,k,j,l] = <ik|rho|jl>
    return np.einsum("ip,pkil->kl", A, R)


def psi_strategy():
    A = []
    for k in INPUTS:
        P = proj(phi(GAMMA[k]))
        M0 = LAMBDA * P if k == "11" else P
        A.append([M0, I2 - M0])                       # {M^0_ax, M^1_ax}
    N = [[proj(phi(XI[(m, y)])) for y in range(2)] for m in range(2)]
    return A, N


def merit(rho, A, N):
    """S = sum_{ax,m,y,b} W(ax,y,b) Tr[(M^m_ax x N^{my}_b) rho], Eq. [bell]."""
    return sum(np.real(np.trace(np.kron(A[k][m],
                                        W[k, y, 0] * N[m][y] + W[k, y, 1] * (I2 - N[m][y])) @ rho))
               for k in range(4) for m in range(2) for y in range(2))


def distinguishability(rho, A):
    """D = (DS_0 + DS_1)/4, each DS_m from the primal SDP, Eq. [primal]."""
    total = 0.0
    for m in range(2):
        sig = [bob_conditional(rho, A[k][m]) for k in range(4)]
        E = [cp.Variable((2, 2), hermitian=True) for _ in range(4)]
        prob = cp.Problem(cp.Maximize(sum(cp.real(cp.trace(E[k] @ sig[k])) for k in range(4))),
                          [e >> 0 for e in E] + [sum(E) == I2])
        prob.solve(solver=cp.CLARABEL)
        total += prob.value
    return total / 4


# ============================================================= 8. main
if __name__ == "__main__":
    t0 = time.time()
    print(f"Task theta = pi/3, constraint D <= {D0}, Lipschitz constant L = {L:.4f}\n")

    U, best, case_star, eta_star, nu_star, n = certify_phiplus()
    print(f"\nphi+ : certified upper bound   S <= {U:.4f}   (target {TARGET})")
    print(f"phi+ : best value attained         {best:.4f}   at eta* = {eta_star:.4f}")
    print(f"phi+ : Bob's case at optimum       {case_star}")
    print(f"phi+ : multiplier nu*              {nu_star:.3f}   (4 nu* = {4 * nu_star:.2f})")
    print(f"       ({n} convex programs, {time.time() - t0:.0f} s)")

    rho = psi_state(THETA)
    A, N = psi_strategy()
    S_psi = merit(rho, A, N)
    D_psi = distinguishability(rho, A)
    print(f"\nPsi  : success merit               {S_psi:.4f}")
    print(f"Psi  : distinguishability          {D_psi:.5f}")

    print()
    if U < S_psi and D_psi <= D0:                      # Eq. [chain]
        print(f"S(phi+) <= {U:.4f} < {S_psi:.4f} = S(Psi) at D <= {D0}:"
              " the maximally entangled state is not optimal.")
    else:
        print("Not certified with these settings.")
