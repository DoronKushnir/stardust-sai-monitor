
import sys, json, numpy as np
from pathlib import Path
ROOT = Path(sys.argv[1]); step = sys.argv[2]
sys.path.insert(0, str(ROOT)); sys.path.insert(0, str(ROOT / "reproduce"))
import element_observable_closure as C
Kt, St = C.triplet_rows()
S_eps = C._blockdiag(C.photon_cov(C.y0, C.SNR_REF) + C.line_covariance(C.SIGMA_ETA_BASE), St)
K = np.vstack([C.K0, Kt])
Sinv = np.linalg.inv(S_eps)
def post(Sa_inv):
    S_hat = np.linalg.inv(K.T @ Sinv @ K + Sa_inv); return S_hat, S_hat @ K.T @ Sinv
S_hat, G = post(C.S_a_inv)
Sa_flat = C.S_a.copy()
for k in range(C.I_AER.start, C.I_AER.stop): Sa_flat[k, k] = 1e12
S_hat_f, G_f = post(np.linalg.inv(Sa_flat))
b = C.removal_budget(C.SNR_REF, C.SIGMA_ETA_BASE)
np.savez_compressed(ROOT / "outputs" / f"grid_convergence_s{step}.npz",
    y0=C.y0, y_gas0=C.y_gas0, y_mean=C.y_mean, K0=C.K0, G_sil=G[C.I_SIL], G_sil_flat=G_f[C.I_SIL],
    sigma_sil_slant=np.sqrt(S_hat[C.I_SIL, C.I_SIL]), sigma_sil_slant_flat=np.sqrt(S_hat_f[C.I_SIL, C.I_SIL]),
    rho=C.RHO_SLANT_TO_SHELL, removal=np.array([b["sigma_total"], b["photon"], b["line"], b["prior_smoothing"]]),
    n_nu=C.n_nu, step=float(step), tau_slant_880=C.tau_slant_880, S_hat_diag=np.sqrt(np.diag(S_hat)))
print("dumped", step, C.n_nu, C.y_gas0[C.i880], flush=True)
