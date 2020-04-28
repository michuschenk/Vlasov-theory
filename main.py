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
