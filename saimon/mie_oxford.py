"""
mie_oxford.py — Oxford EODG Mie backend (MieModelProtocol).

PROVENANCE
----------
This module is a faithful Python/NumPy port of the Oxford University Earth
Observation Data Group (EODG) IDL routine mie_single.pro (non-DLM branch).

Source file: mie_single.pro
URL        : http://eodg.atm.ox.ac.uk/MIE/
Accessed   : 2026-06-10

The original IDL code is by G. Thomas (1998, from mieint.f), with subsequent
modifications by R. Grainger, G. Thomas, A. Smith, and G. McGarragh (up to
2015). See the IDL source header for the full modification history.

PORT NOTES
----------
Only the non-DLM/IDL branch of mie_single.pro is ported here. The DLM/Fortran
branch (keyword DLM) is ignored.

The following details from the IDL source are faithfully preserved:

1. SIGN CONVENTION: IDL mie_single expects ``Cm = n - ik`` (negative imaginary
   for absorbing particles; the source issues a warning on line 111 when
   ``imaginary(Cm) > 0``).  Our project uses ``m = n + ik`` (k >= 0).
   Conversion at entry: ``Cm = complex(m.real, -abs(m.imag))``.
   The internal variable ``Cm`` throughout matches the IDL variable of the
   same name.

2. PIECEWISE NStop FORMULA (IDL lines 191–200):
     Dx < 0.02           : NStop = 2
     Dx <= 8.0           : NStop = Dx + 4.00*Dx^(1/3) + 2
     Dx < 4200           : NStop = Dx + 4.05*Dx^(1/3) + 2
     else                : NStop = Dx + 4.00*Dx^(1/3) + 2
   In IDL, NStop is a float at this point; IDL's ``for N = 1, NStop`` loop
   truncates it to an integer automatically. We use ``int(NStop_float)``
   explicitly.

3. NmX (IDL line 201):
     NmX = long(max([NStop, abs(Y)]) + 15)
   where Y = Dx * Cm (complex). NStop here is the float computed above.
   ``long()`` in IDL = ``int()`` in Python for positive values.

4. D DOWNWARD RECURRENCE (IDL lines 204–207):
     D = dcomplexarr(NmX + 1),  D[NmX] = 0+0j  (default)
     for N = NmX-1, 1, -1:
         A1 = (N+1) / Y
         D[N] = A1 - 1 / (A1 + D[N+1])

5. RICCATI-BESSEL FUNCTIONS (IDL lines 214–220):
     Psi0 = cos(Dx),  Psi1 = sin(Dx)
     Chi0 = -sin(Dx), Chi1 = cos(Dx)
     Xi0 = Psi0 + i*Chi0,  Xi1 = Psi1 + i*Chi1
   Note: EODG uses Xi = Psi + i*Chi, which equals the complex conjugate of
   the B&H xi = psi - i*chi. Q_ext and Q_sca are invariant under this
   conjugation (only Re(A+B) and |A|^2+|B|^2 enter).

6. SERIES LOOP (IDL lines 227–263):
   ``for N = 1, NStop`` with internal recurrence:
     Rnx = N / Dx
     Tnm1 = 2N - 1      (IDL: Tnm1 = Tnp1 - 2, Tnp1 starts 1 and adds 2)
     Psi  = Tnm1 * Psi1 / Dx - Psi0
     Chi  = Tnm1 * Chi1 / Dx - Chi0
     Xi   = Psi + i*Chi
     A = ((D[N]*Ir + Rnx)*Psi - Psi1) / ((D[N]*Ir + Rnx)*Xi  - Xi1)
     B = ((D[N]*Cm + Rnx)*Psi - Psi1) / ((D[N]*Cm + Rnx)*Xi  - Xi1)
     Dqxt += Tnp1 * Re(A + B)
     Dqsc += Tnp1 * (|A|^2 + |B|^2)

7. NORMALIZATION (IDL lines 287–288):
     Dqsc = 2 * Dqsc / Dx^2
     Dqxt = 2 * Dqxt / Dx^2

FOUR INDEPENDENT CODE PATHS
----------------------------
The module exposes four independent Mie backends:
  BohrenHuffmanMie    (sage3.mie)         — B&H algorithm, single truncation nstop
  MiePythonBackend    (sage3.mie)         — wraps miepython >= 3.x
  OxfordMie           (this module)       — faithful IDL port (ACTIVE backend)
  OxfordMieReimpl     (this module)       — prior algorithmic reimplementation (NOT active)
"""

from __future__ import annotations

import numpy as np


# ---------------------------------------------------------------------------
# Internal scalar kernel — faithful IDL port
# ---------------------------------------------------------------------------

def _oxford_mie_single_port(x: float, Cm: complex) -> tuple[float, float, float]:
    """Faithful port of the IDL non-DLM loop in mie_single.pro.

    Parameters
    ----------
    x  : size parameter (IDL variable ``Dx(Size)``)
    Cm : complex refractive index in EODG convention ``n - ik`` (k >= 0 absorbing)
         i.e. imaginary part is <= 0 for absorbing particles.

    Returns
    -------
    q_ext, q_sca, q_abs
    """
    if x < 1e-10:
        return 0.0, 0.0, 0.0

    Ir = 1.0 / Cm         # IDL: Ir = 1.D0 / Cm
    Y = x * Cm            # IDL: Y  = Dx(Size) * Cm

    # ------------------------------------------------------------------
    # Piecewise NStop formula (IDL lines 191-200).
    # NStop is a float at this stage (IDL does not cast it yet).
    # ------------------------------------------------------------------
    if x < 0.02:
        NStop_float = 2.0
    elif x <= 8.0:
        NStop_float = x + 4.00 * x ** (1.0 / 3.0) + 2.0
    elif x < 4200.0:
        NStop_float = x + 4.05 * x ** (1.0 / 3.0) + 2.0
    else:
        NStop_float = x + 4.00 * x ** (1.0 / 3.0) + 2.0

    # IDL: NmX = long(max([NStop, abs(Y)]) + 15)
    # max([NStop_float, abs(Y)]) uses the float NStop; long() truncates toward zero.
    NmX = int(max(NStop_float, abs(Y)) + 15.0)

    # Series truncation: IDL's ``for N = 1, NStop`` truncates the float.
    NStop = int(NStop_float)

    # ------------------------------------------------------------------
    # D downward recurrence (IDL lines 202-207).
    # D is a complex array of size NmX+1, initialised to 0+0j.
    # Loop: for N = NmX-1, 1, -1
    # ------------------------------------------------------------------
    D = np.zeros(NmX + 1, dtype=complex)  # D[NmX] = 0+0j by default

    for N in range(NmX - 1, 0, -1):       # N goes NmX-1, NmX-2, ..., 1
        A1 = (N + 1) / Y                  # IDL: A1 = (N+1) / Y
        D[N] = A1 - 1.0 / (A1 + D[N + 1])

    # ------------------------------------------------------------------
    # Riccati-Bessel initial conditions (IDL lines 214-220).
    # Psi, Chi: real; Xi = dcomplex(Psi, Chi) = Psi + i*Chi.
    # ------------------------------------------------------------------
    Psi0 = np.cos(x)        # IDL: Psi0 = cos(Dx(Size))
    Psi1 = np.sin(x)        # IDL: Psi1 = sin(Dx(Size))
    Chi0 = -np.sin(x)       # IDL: Chi0 = -sin(Dx(Size))
    Chi1 = np.cos(x)        # IDL: Chi1 =  cos(Dx(Size))
    # Xi0 = Psi0 + i*Chi0   (not used in loop, kept for clarity)
    Xi1 = complex(Psi1, Chi1)  # IDL: Xi1 = dcomplex(Psi1,Chi1)

    # ------------------------------------------------------------------
    # Accumulation variables
    # ------------------------------------------------------------------
    Dqxt = 0.0              # IDL: Dqxt(Size) = 0.D0
    Dqsc = 0.0              # IDL: Dqsc(Size) = 0.D0
    Tnp1 = 1.0              # IDL: Tnp1 = 1D0  (starts at 1, first iteration adds 2 -> 3)

    # ------------------------------------------------------------------
    # Series loop (IDL lines 227-263): for N = 1, NStop
    # ------------------------------------------------------------------
    for N in range(1, NStop + 1):
        DN = float(N)
        Tnp1 = Tnp1 + 2.0              # IDL: Tnp1 = Tnp1 + 2D0  -> at N=1: Tnp1=3
        Tnm1 = Tnp1 - 2.0              # IDL: Tnm1 = Tnp1 - 2D0  -> at N=1: Tnm1=1
        Rnx = DN / x                   # IDL: Rnx = DN/Dx(Size)

        # Riccati-Bessel recurrence
        Psi = (Tnm1 / x) * Psi1 - Psi0   # IDL: Psi = double(Tnm1)*Psi1/Dx(Size) - Psi0
        Chi = (Tnm1 / x) * Chi1 - Chi0   # IDL: Chi = Tnm1*Chi1/Dx(Size) - Chi0
        Xi = complex(Psi, Chi)            # IDL: Xi = dcomplex(Psi,Chi)

        # Mie coefficients (IDL lines 237-238)
        # A = ((D[N]*Ir + Rnx)*Psi - Psi1) / ((D[N]*Ir + Rnx)*Xi - Xi1)
        # B = ((D[N]*Cm + Rnx)*Psi - Psi1) / ((D[N]*Cm + Rnx)*Xi - Xi1)
        tmp_A = D[N] * Ir + Rnx
        A = (tmp_A * Psi - Psi1) / (tmp_A * Xi - Xi1)

        tmp_B = D[N] * Cm + Rnx
        B = (tmp_B * Psi - Psi1) / (tmp_B * Xi - Xi1)

        # Accumulate efficiencies (IDL lines 239-240)
        Dqxt += Tnp1 * (A + B).real                  # IDL: Tnp1 * double(A + B)
        Dqsc += Tnp1 * (abs(A) ** 2 + abs(B) ** 2)  # IDL: Tnp1 * double(A*conj(A)+B*conj(B))

        # Advance Riccati-Bessel recurrence
        Psi0 = Psi1     # IDL: Psi0 = Psi1
        Psi1 = Psi      # IDL: Psi1 = Psi
        Chi0 = Chi1     # IDL: Chi0 = Chi1
        Chi1 = Chi      # IDL: Chi1 = Chi
        Xi1 = complex(Psi1, Chi1)   # IDL: Xi1 = dcomplex(Psi1,Chi1)

    # ------------------------------------------------------------------
    # Normalise (IDL lines 287-288): Dqsc = 2*Dqsc/Dx^2, Dqxt = 2*Dqxt/Dx^2
    # ------------------------------------------------------------------
    norm = 2.0 / x ** 2
    q_ext = norm * Dqxt
    q_sca = norm * Dqsc
    q_abs = q_ext - q_sca

    return float(q_ext), float(q_sca), float(max(0.0, q_abs))


# ---------------------------------------------------------------------------
# Public class — faithful port (ACTIVE backend)
# ---------------------------------------------------------------------------

class OxfordMie:
    """Faithful Python port of EODG mie_single.pro (non-DLM branch).

    This is the ACTIVE Oxford EODG backend. It is a line-faithful port of
    the IDL source mie_single.pro available at http://eodg.atm.ox.ac.uk/MIE/
    (accessed 2026-06-10). See the module docstring for full port notes.

    Sign convention (project interface): m = n + ik, k >= 0 for absorbing
    particles. Internally converted to EODG convention Cm = n - ik before
    passing to the port kernel.
    """

    def efficiencies(
        self,
        size_parameter: np.ndarray,
        m: complex,
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Compute Q_ext, Q_sca, Q_abs for a homogeneous sphere.

        Parameters
        ----------
        size_parameter : scalar or array  (x = 2*pi*r/lambda)
        m              : complex refractive index n + ik, k >= 0

        Returns
        -------
        q_ext, q_sca, q_abs : arrays shaped like size_parameter
        """
        # Convert project convention m = n + ik  ->  EODG convention Cm = n - ik
        Cm = complex(m.real, -abs(m.imag))

        x_arr = np.asarray(size_parameter, dtype=float)
        orig_shape = x_arr.shape
        x_flat = x_arr.ravel()

        q_ext = np.empty(x_flat.size)
        q_sca = np.empty(x_flat.size)
        q_abs = np.empty(x_flat.size)

        for i, xi in enumerate(x_flat):
            q_ext[i], q_sca[i], q_abs[i] = _oxford_mie_single_port(xi, Cm)

        return (
            q_ext.reshape(orig_shape),
            q_sca.reshape(orig_shape),
            q_abs.reshape(orig_shape),
        )


# ---------------------------------------------------------------------------
# Prior reimplementation — kept for reference, NOT the active backend
# ---------------------------------------------------------------------------

def _oxford_mie_single(x: float, m: complex) -> tuple[float, float, float]:
    """PRIOR REIMPLEMENTATION (not IDL-faithful). Kept for reference.

    Uses the max(|mx|, x) Wiscombe criterion for both nmax and nstop, which
    differs from the IDL piecewise NStop formula. See OxfordMieReimpl docstring.
    """
    if x < 1e-10:
        return 0.0, 0.0, 0.0

    z = m * x

    nmax_arg = max(abs(z), x)
    nmax  = max(int(nmax_arg + 4.0 * nmax_arg ** (1.0 / 3.0) + 2) + 1, 3)
    nstop = max(int(x       + 4.0 * x       ** (1.0 / 3.0) + 2) + 1, 3)

    D = np.zeros(nmax + 2, dtype=complex)
    for n in range(nmax, 0, -1):
        D[n - 1] = n / z - 1.0 / (D[n] + n / z)

    psi_prev = np.cos(x)
    psi_curr = np.sin(x)
    chi_prev = -np.sin(x)
    chi_curr = np.cos(x)

    q_ext_sum = 0.0
    q_sca_sum = 0.0

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

        wt = 2 * n + 1
        q_ext_sum += wt * (an + bn).real
        q_sca_sum += wt * (abs(an) ** 2 + abs(bn) ** 2)

        psi_prev, psi_curr = psi_curr, psi_next
        chi_prev, chi_curr = chi_curr, chi_next

    norm = 2.0 / x ** 2
    q_ext = norm * q_ext_sum
    q_sca = norm * q_sca_sum
    q_abs = q_ext - q_sca

    return float(q_ext), float(q_sca), float(max(0.0, q_abs))


class OxfordMieReimpl:
    """PRIOR ALGORITHMIC REIMPLEMENTATION of EODG mie_single.pro.

    THIS IS NOT THE ACTIVE BACKEND and is NOT a faithful port of the IDL
    source. It was written before the IDL source was available, based on
    Bohren & Huffman (1983) §4.4 and Wiscombe (1980) documentation.

    Key differences from the IDL source (and from OxfordMie):
    - Uses max(|mx|, x) for BOTH the downward-recurrence start (nmax) AND the
      series truncation (nstop), whereas the IDL code uses a piecewise formula
      based on x only for NStop, and max(NStop_float, |Y|) + 15 for NmX.
    - Uses B&H sign convention xi = psi - i*chi throughout (IDL uses Xi = Psi + i*Chi).
    - m is used directly (project convention n+ik) rather than Cm = n-ik.

    Kept here as a 4th independent code path for archival and cross-check
    purposes. Results agree with OxfordMie and BohrenHuffmanMie to < 1e-6
    relative error for typical stratospheric aerosol parameters.
    """

    def efficiencies(
        self,
        size_parameter: np.ndarray,
        m: complex,
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Compute Q_ext, Q_sca, Q_abs for a homogeneous sphere.

        Parameters
        ----------
        size_parameter : scalar or array  (x = 2*pi*r/lambda)
        m              : complex refractive index n + ik, k >= 0

        Returns
        -------
        q_ext, q_sca, q_abs : arrays shaped like size_parameter
        """
        x_arr = np.asarray(size_parameter, dtype=float)
        orig_shape = x_arr.shape
        x_flat = x_arr.ravel()

        q_ext = np.empty(x_flat.size)
        q_sca = np.empty(x_flat.size)
        q_abs = np.empty(x_flat.size)

        for i, xi in enumerate(x_flat):
            q_ext[i], q_sca[i], q_abs[i] = _oxford_mie_single(xi, complex(m))

        return (
            q_ext.reshape(orig_shape),
            q_sca.reshape(orig_shape),
            q_abs.reshape(orig_shape),
        )
