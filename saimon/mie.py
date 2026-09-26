"""
mie.py — Bohren-Huffman Mie scattering engine.

Computes extinction, scattering, and absorption efficiencies Q_ext, Q_sca, Q_abs
for a homogeneous sphere using the standard Bohren & Huffman (1983) algorithm.

Sign convention (matching sage3.interfaces.MieModelProtocol and P0 contract):
    m = n + i k,   k >= 0 for an absorbing particle.

Two backend classes implement MieModelProtocol:
    BohrenHuffmanMie   — built-in, no external dependencies (default).
    MiePythonBackend   — wraps miepython for cross-validation (optional).
"""

import numpy as np


# ---------------------------------------------------------------------------
# Internal scalar kernel
# ---------------------------------------------------------------------------

def _mie_single_with_g(x: float, m: complex) -> tuple[float, float, float, float]:
    """Bohren-Huffman Mie efficiencies + asymmetry parameter g for one size parameter."""
    if x < 1e-10:
        return 0.0, 0.0, 0.0, 0.0

    z = m * x
    nstop = max(int(x + 4.0 * x ** (1.0 / 3.0) + 2) + 1, 3)
    nmx   = max(nstop, int(abs(z))) + 15

    D = np.zeros(nmx + 2, dtype=complex)
    for n in range(nmx, 0, -1):
        D[n - 1] = n / z - 1.0 / (D[n] + n / z)

    psi_prev = np.cos(x)
    psi_curr = np.sin(x)
    chi_prev = -np.sin(x)
    chi_curr = np.cos(x)

    q_ext_sum = 0.0
    q_sca_sum = 0.0
    # Store all Mie coefficients (1-indexed; index 0 unused)
    an_arr = np.zeros(nstop + 2, dtype=complex)
    bn_arr = np.zeros(nstop + 2, dtype=complex)

    for n in range(1, nstop + 1):
        fn = (2 * n - 1) / x
        psi_next = fn * psi_curr - psi_prev
        chi_next = fn * chi_curr - chi_prev
        xi_n   = psi_next - 1j * chi_next
        xi_nm1 = psi_curr - 1j * chi_curr
        Dn = D[n]

        tmp_a = Dn / m + n / x
        an = (tmp_a * psi_next - psi_curr) / (tmp_a * xi_n - xi_nm1)
        tmp_b = m * Dn + n / x
        bn = (tmp_b * psi_next - psi_curr) / (tmp_b * xi_n - xi_nm1)

        an_arr[n] = an
        bn_arr[n] = bn

        wt = 2 * n + 1
        q_ext_sum += wt * (an + bn).real
        q_sca_sum += wt * (abs(an) ** 2 + abs(bn) ** 2)

        psi_prev, psi_curr = psi_curr, psi_next
        chi_prev, chi_curr = chi_curr, chi_next

    norm = 2.0 / x ** 2
    q_ext = norm * q_ext_sum
    q_sca = norm * q_sca_sum
    q_abs = q_ext - q_sca

    # Asymmetry parameter g (Bohren & Huffman eq. 4.70)
    g_sum = 0.0
    for n in range(1, nstop):   # a_{n+1} exists up to n = nstop-1
        an  = an_arr[n]
        an1 = an_arr[n + 1]
        bn  = bn_arr[n]
        bn1 = bn_arr[n + 1]
        g_sum += (n * (n + 2) / (n + 1)) * (an * an1.conjugate() + bn * bn1.conjugate()).real
        g_sum += ((2 * n + 1) / (n * (n + 1))) * (an * bn.conjugate()).real

    g_asym = (4.0 / (q_sca * x ** 2)) * g_sum if q_sca > 1e-20 else 0.0

    return float(q_ext), float(q_sca), float(max(0.0, q_abs)), float(np.clip(g_asym, -1.0, 1.0))


def _mie_single(x: float, m: complex) -> tuple[float, float, float]:
    """Bohren-Huffman Mie efficiencies for one size parameter x."""
    if x < 1e-10:
        return 0.0, 0.0, 0.0

    # Wiscombe (1980) two-parameter truncation — Bohren & Huffman §4.4:
    #   nstop : series truncation — bounds the Riccati-Bessel upward recurrence
    #           (psi/chi overflow for n >> x, so keep n <= nstop ~ x).
    #   nmx   : D_n downward-recurrence start — must exceed |m*x| (not just x)
    #           so the recurrence has converged by the time it reaches n=1.
    #   Reference: Wiscombe (1980), Appl. Opt. 19, 1505; EODG mie_single.pro.
    z = m * x
    nstop = max(int(x + 4.0 * x ** (1.0 / 3.0) + 2) + 1, 3)
    nmx   = max(nstop, int(abs(z))) + 15

    # Downward recurrence for D_n(mx), starting from D[nmx] = 0+0j.
    # Only D[1..nstop] are used in the series; nmx >> nstop ensures convergence.
    D = np.zeros(nmx + 2, dtype=complex)
    for n in range(nmx, 0, -1):
        D[n - 1] = n / z - 1.0 / (D[n] + n / z)

    # Initial values for the Riccati-Bessel upward recurrence:
    #   psi_n(x)  = x * j_n(x)    =>  psi_{-1} = cos(x),  psi_0 = sin(x)
    #   chi_n(x)  = -x * y_n(x)   =>  chi_{-1} = -sin(x), chi_0 = cos(x)
    #   xi_n(x)   = psi_n - i*chi_n
    # Recurrence (both psi and chi): f_n = (2n-1)/x * f_{n-1} - f_{n-2}
    # Series runs only to nstop (not nmx) to avoid Riccati-Bessel overflow.
    psi_prev = np.cos(x)         # psi_{-1}
    psi_curr = np.sin(x)         # psi_0
    chi_prev = -np.sin(x)        # chi_{-1}
    chi_curr = np.cos(x)         # chi_0

    q_ext_sum = 0.0
    q_sca_sum = 0.0

    for n in range(1, nstop + 1):
        fn = (2 * n - 1) / x

        psi_next = fn * psi_curr - psi_prev    # psi_n
        chi_next = fn * chi_curr - chi_prev    # chi_n
        xi_n     = psi_next - 1j * chi_next   # xi_n
        xi_nm1   = psi_curr - 1j * chi_curr   # xi_{n-1}

        Dn = D[n]

        # Mie coefficients a_n and b_n  (B&H eq. 4.88)
        tmp_a = Dn / m + n / x
        an = (tmp_a * psi_next - psi_curr) / (tmp_a * xi_n - xi_nm1)

        tmp_b = m * Dn + n / x
        bn = (tmp_b * psi_next - psi_curr) / (tmp_b * xi_n - xi_nm1)

        wt = 2 * n + 1
        q_ext_sum += wt * (an + bn).real
        q_sca_sum += wt * (abs(an) ** 2 + abs(bn) ** 2)

        # Advance recurrence
        psi_prev, psi_curr = psi_curr, psi_next
        chi_prev, chi_curr = chi_curr, chi_next

    norm = 2.0 / x ** 2
    q_ext = norm * q_ext_sum
    q_sca = norm * q_sca_sum
    q_abs = q_ext - q_sca

    return float(q_ext), float(q_sca), float(max(0.0, q_abs))


# ---------------------------------------------------------------------------
# Public vectorised API
# ---------------------------------------------------------------------------

def mie_efficiencies(
    x: np.ndarray,
    m: complex,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Bohren & Huffman (1983) Mie efficiencies for a sphere.

    Parameters
    ----------
    x : size parameter = 2*pi*r/lambda (scalar or array)
    m : complex refractive index n + i k  (k >= 0 absorbing)

    Returns
    -------
    q_ext, q_sca, q_abs : arrays shaped like x
    """
    x_arr = np.asarray(x, dtype=float)
    orig_shape = x_arr.shape
    x_flat = x_arr.ravel()

    q_ext = np.empty(x_flat.size)
    q_sca = np.empty(x_flat.size)
    q_abs = np.empty(x_flat.size)

    for i, xi in enumerate(x_flat):
        q_ext[i], q_sca[i], q_abs[i] = _mie_single(xi, m)

    return (q_ext.reshape(orig_shape),
            q_sca.reshape(orig_shape),
            q_abs.reshape(orig_shape))


def mie_efficiencies_g(
    x: np.ndarray,
    m: complex,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Bohren & Huffman Mie efficiencies plus asymmetry parameter g.

    Parameters
    ----------
    x : size parameter = 2*pi*r/lambda (scalar or array)
    m : complex refractive index n + i k  (k >= 0 absorbing)

    Returns
    -------
    q_ext, q_sca, q_abs, g_asym : arrays shaped like x
        g_asym is the cosine-averaged asymmetry parameter in [-1, 1].
    """
    x_arr = np.asarray(x, dtype=float)
    orig_shape = x_arr.shape
    x_flat = x_arr.ravel()

    q_ext  = np.empty(x_flat.size)
    q_sca  = np.empty(x_flat.size)
    q_abs  = np.empty(x_flat.size)
    g_asym = np.empty(x_flat.size)

    for i, xi in enumerate(x_flat):
        q_ext[i], q_sca[i], q_abs[i], g_asym[i] = _mie_single_with_g(xi, m)

    return (q_ext.reshape(orig_shape), q_sca.reshape(orig_shape),
            q_abs.reshape(orig_shape), g_asym.reshape(orig_shape))


# ---------------------------------------------------------------------------
# Backend classes (MieModelProtocol)
# ---------------------------------------------------------------------------

class BohrenHuffmanMie:
    """Built-in Mie backend — wraps mie_efficiencies. No external dependencies."""

    def efficiencies(
        self,
        size_parameter: np.ndarray,
        m: complex,
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        return mie_efficiencies(size_parameter, m)

    def efficiencies_g(
        self,
        size_parameter: np.ndarray,
        m: complex,
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """Like efficiencies() but also returns asymmetry parameter g."""
        return mie_efficiencies_g(size_parameter, m)


class MiePythonBackend:
    """Optional cross-check backend wrapping miepython >= 3.x.

    miepython uses the convention m = n - i*k for absorption (k > 0 absorbing),
    opposite to this project's convention m = n + i*k (k >= 0).
    The imaginary part is negated on entry so both conventions agree physically.

    Install with:  pip install miepython
    API note: miepython 3.x uses efficiencies_mx(m, x) returning (qext, qsca,
    qback, g); the old mie(m, x) from 2.x is no longer available.
    """

    def efficiencies(
        self,
        size_parameter: np.ndarray,
        m: complex,
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        try:
            import miepython
        except ImportError:
            raise ImportError(
                "miepython is not installed. "
                "Install it with: pip install miepython"
            )
        # Convert project convention (k >= 0) to miepython convention (k <= 0).
        m_mpy = complex(m.real, -abs(m.imag))
        x = np.atleast_1d(np.asarray(size_parameter, dtype=float))
        q_ext, q_sca, _q_back, _g = miepython.efficiencies_mx(m_mpy, x)
        q_abs = np.maximum(q_ext - q_sca, 0.0)
        return q_ext, q_sca, q_abs
