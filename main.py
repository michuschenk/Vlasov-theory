import numpy as np
import matplotlib.pyplot as plt
from matplotlib.pyplot import cm
from scipy.integrate import quad, fixed_quad
from scipy.interpolate import interp1d
from scipy.constants import c, e, m_p

from integrators import hlp2, hlp2_parallel


class Resonator:
    """ Implements a resonator impedance with the PyHeadtail and the
    Chao convention, respectively """

    def __init__(self, r_shunt, omega_res, Q, convention="PyHeadtail"):
        """
        :param r_shunt: shunt impedance [Ohm/m]
        :param omega_res: resonator frequency [rad/s]
        :param Q: quality factor
        :param convention: can be either 'Chao' or 'PyHeadtail' to work
        with two different conventions of resonators.
        :returns: Resonator object
        """
        self.r_shunt = r_shunt
        self.omega_res = omega_res
        self.Q = Q
        self.convention = convention

    def evaluate(self, omega):
        z = (c / omega * self.r_shunt /
             (1. + 1j * self.Q * (self.omega_res / omega - omega / self.omega_res)))
        if self.convention == "PyHeadtail":
            z *= self.omega_res / c
        return z


class ImpedanceFile:
    """ Reads impedance data from file (columns are frequency [Hz], real part
    [Ohm/m], imaginary part [Ohm/m]) and creates interpolation functions based
    on the data. This is suited for impedance models that cannot be described
    by simple resonator, for example. """

    def __init__(self, filename):
        """
        :param: omega: sorted (assumed) array of frequencies [rad/s],
        :param: real: real component [Ohm/m],
        :param: imag: imaginary component [Ohm/m] """
        raw_data = np.loadtxt(filename)
        omega = raw_data[:, 0] * 2 * np.pi  # convert from [Hz] to [rad/s]
        self.real = interp1d(omega, raw_data[:, 1])
        self.imag = interp1d(omega, raw_data[:, 2])
        self.omega_min = omega[0]

    def evaluate(self, omega):
        """ Evaluates impedance for frequency vector :param omega. """
        z = np.zeros(len(omega), dtype=np.complex)
        msk_pos = (omega > 0) & (omega > self.omega_min)
        msk_neg = (omega < 0) & (omega < -self.omega_min)
        msk_0 = np.abs(omega) <= self.omega_min
        z[msk_pos] = self.real(omega[msk_pos]) + 1j * self.imag(omega[msk_pos])
        z[msk_neg] = -self.real(-omega[msk_neg]) + 1j * self.imag(-omega[msk_neg])
        z[msk_0] = 0.
        return z


def avg_Q(r, R, eta, Qs, Qpp):
    return Qpp * Qs ** 2 * r ** 2 / (4. * eta ** 2 * R ** 2)



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
    Zeval = Z.evaluate(omega_p)

    # Serial versions of hlp2 very slow - hence use openMP parallel one
    hlp2 = hlp2_parallel(Q0=Q_beta, Qs=Qs, rb=z_hat, R=R, eta=eta,
                         Qp=Qp, Qpp=Qpp, l=l, p_max=p_max)

    sum_term = np.sum(hlp2 * Zeval)

    delta_omega_l = -1j * N * e ** 2 * c / (2. * E0 * T0 ** 2 * omega_beta) * sum_term + avQ * omega0
    return delta_omega_l / omega0


# In PyHeadtail we do not use exactly the wake field definition
# made by Chao.Hence, also Chao 's impedance formula should not be applied
# here( in general), because of the different conventions. Basically,
# it is a factor c / omega_res.Chao is consistent in itself.But, for us, to
# compare the analytical formula with PyHeadtail simulations with the same
# resonator parameters, i.e.especially same r_shunt and omega_res) we must
# typically use 'PyHeadtail' convention.

# only scan in Q', to compare Antoine's Hlp function with Bessel from Chao.
# in case of Q'' = 0, the two should be equivalent. And they are!
calc = True
plot = True

if calc:
    # Define impedance to be used.
    # Resonator parameters - peaked (i.e. narrow-band)
    omegar = 5.022163e8
    Rs = 1e6 * 5e6  # [Ohm/m**2].
    Q = 1e5
    Z = Resonator(Rs, omegar, Q, convention="PyHeadtail")

    # Dependence on Q'
    Qp_vect = np.linspace(-2, 2, 5)
    l_vect = np.arange(-1, 2, 1)
    deltaQ_re = np.zeros((len(l_vect), len(Qp_vect)))
    deltaQ_im = np.zeros((len(l_vect), len(Qp_vect)))

    ctr = 0
    for i, l in enumerate(l_vect):
        deltaQ_res = []
        for Qp in Qp_vect:
            if ctr % 5 == 0:
                print('{:d} / {:d}'.format(ctr, len(Qp_vect) * len(l_vect)))
            deltaQ_res.append(get_deltaQ(Z=Z, Jz=3e-4, Qp=Qp, Qpp=0., l=l,
                                         p_max=120000))
            ctr += 1
        deltaQ_re[i, :] = np.array(np.real(deltaQ_res))
        deltaQ_im[i, :] = np.array(np.imag(deltaQ_res))

if plot:
    C = 6911.  # [m], accelerator circumference
    gamma = 27.7  # 103. #27.7 # Relativistic gamma, at injection 26 GeV. No units.
    beta = np.sqrt(1. - 1. / gamma ** 2)
    T0 = C / (beta * c)
    omega0 = 1. / T0 * 2. * np.pi
    omega_s = 0.017 * omega0  # Synchrotron tune, linear motion, Q20 optics.

    # Plot result
    fig = plt.figure(figsize=(12, 10))
    ax1 = fig.add_subplot(211)
    ax2 = fig.add_subplot(212, sharex=ax1)
    cols = cm.rainbow(np.linspace(0, 1, len(l_vect)))

    for i, l in enumerate(l_vect):
        # ax1.plot(Qp_vect, (deltaQ_re[i,:]/(omega_s/omega0) + l), ls='solid',
        #         c=cols[i], label='l={:d}'.format(l))
        # ax1.plot(Qp_vect, deltaQ_re[i,:]/(omega_s/omega0) + l, ls='solid',
        #         c=cols[i], label='l={:d}'.format(l))
        ax1.plot(Qp_vect, deltaQ_re[i, :], ls='solid',
                 c=cols[i], label='l={:d}'.format(l))
        ax2.plot(Qp_vect, deltaQ_im[i, :], ls='solid', c=cols[i])

    ax1.set_ylabel(r"$Re\left(\Delta Q_c\right)$")
    ax2.set_xlabel("Q'")
    ax2.set_ylabel(r"$-Im\left(\Delta Q_c\right)$")

    ax1.set_xlim((np.min(Qp_vect), np.max(Qp_vect)))

    ax1.get_yaxis().get_major_formatter().set_useOffset(False)
    ax2.get_yaxis().get_major_formatter().set_useOffset(False)

    ax1.legend(loc='best', fontsize=16, ncol=1, bbox_to_anchor=(1.02, 1))
    plt.subplots_adjust(right=0.85, left=0.12)
    # plt.savefig('Antoine_analytic_RW_Jz3e-4_QpOnly.png', dpi=150)
    plt.show()

