import numpy as np
import matplotlib.pyplot as plt
from matplotlib.pyplot import cm
from scipy.integrate import quad, fixed_quad
from scipy.interpolate import interp1d
from scipy.constants import c, e, m_p

from integrators import hlp2, hlp2_parallel



# NOTE: In PyHT we don't use exactly the wake field definition made by Chao.
# Hence, also Chao's impedance formula should not be applied here, because of
# the different conventions. Basically, it's a factor c/omegar difference.
# Chao is consistent in himself. But, for us, to compare the analytic formula
# with PyHT simulations (with the same resonator parameters, i.e. especially
# same Rs and same omegar) we need to use the impedance shown below
# reson_transv_equivPyHT.
def Chao_resonator_transverse(Rs, omegar, Q):
    def Z(omega):
        return c / omega * Rs / (1. + 1j * Q * (omegar / omega - omega / omegar))
    return Z


def reson_transv_equivPyHT(Rs, omegar, Q):
    def Z(omega):
        return omegar / omega * Rs / (1. + 1j * Q * (omegar / omega - omega / omegar))

    return Z


def avg_Q(r, R, eta, Qs, Qpp):
    return Qpp * Qs ** 2 * r ** 2 / (4. * eta ** 2 * R ** 2)


def imported_imp(filename):
    fdata = np.loadtxt(filename)
    freq = fdata[:, 0]
    omg = freq * 2. * np.pi
    reZ = fdata[:, 1]
    imZ = fdata[:, 2]

    omg_min = omg[0]

    reZ_ip = interp1d(omg, reZ)
    imZ_ip = interp1d(omg, imZ)

    def Z(omega):
        zz = np.zeros(len(omega), dtype=np.complex)
        msk_pos = (omega > 0) & (omega > omg_min)
        msk_neg = (omega < 0) & (omega < -omg_min)
        msk_0 = np.abs(omega) <= omg_min
        zz[msk_pos] = reZ_ip(omega[msk_pos]) + 1j * imZ_ip(omega[msk_pos])
        zz[msk_neg] = -reZ_ip(-omega[msk_neg]) + 1j * imZ_ip(-omega[msk_neg])
        zz[msk_0] = 0.
        return zz

    return Z


def get_deltaQ(Z=None, Jz=5e-4, Qp=0., Qpp=0., l=0, p_max=100000):
    # Parameters (use something like SPS)
    N = 1e9   # Number of protons in bunch

    gamma = 27.7  # 103. #27.7 # Relativistic gamma, at injection 26 GeV. No units.
    E0 = m_p * c**2 * gamma
    beta = np.sqrt(1. - 1. / gamma**2)
    C = 6911.  # [m], accelerator circumference
    T0 = C / (beta * c)
    R = C / (2. * np.pi)
    omega0 = 1. / T0 * 2. * np.pi

    alpha_0 = 0.00308
    eta = alpha_0 - gamma ** -2  # Slippage factor
    # eta = 0.001777 # Slippage factor, no units.
    Q_beta = 20.18
    omega_beta = Q_beta * omega0  # Betatron tune, here in y.
    Qs = 0.017
    omega_s = Qs * omega0  # Synchrotron tune, linear motion, Q20 optics.

    beta_z = eta * C / (2. * np.pi) / (omega_s / omega0)
    z_hat = np.sqrt(2. * beta_z * Jz)  # [m]

    # Chromaticity
    xi = Qp / (omega_beta / omega0)
    chi = xi * omega_beta * z_hat / (c * eta)  # Head-tail phase

    # CALCULATE TUNE SHIFT (Antoine Maillard, Eq. 50)
    avQ = avg_Q(z_hat, R, eta, Qs, Qpp)
    p_vect = np.arange(-p_max, p_max + 1)
    omega_p = p_vect * omega0 + omega_beta + l * omega_s
    Zeval = Z(omega_p)

    # Serial versions of hlp2 very slow - hence use openMP parallel one
    hlp2 = hlp2_parallel(Q0=Q_beta, Qs=Qs, rb=z_hat, R=R, eta=eta,
                         Qp=Qp, Qpp=Qpp, l=l, p_max=p_max)

    sum_term = np.sum(hlp2 * Zeval)

    delta_omega_l = -1j * N * e ** 2 * c / (2. * E0 * T0 ** 2 * omega_beta) * sum_term + avQ * omega0
    return delta_omega_l / omega0


