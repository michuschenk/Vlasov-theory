import numpy as np
import matplotlib.pyplot as plt
from matplotlib.pyplot import cm
from scipy.interpolate import interp1d
from scipy.constants import c, e, m_p

from integrators import hlp2_parallel


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


def update_machine_parameters(machine):
    """ Calculates additional machine parameters and adds them to the dict.
    :param beta: relativistic beta,
    :param T0: revolution period [s]
    :param eta: slippage factor
    :param omega_0: revolution frequency [rad/s]
    :param beta_z: longitudinal beta function [m]
    :param radius: machine physical radius [m]
    :param total_energy: total beam energy [J]
    :param z_hat: airbag beam radius [m] """
    beta = np.sqrt(1. - 1. / machine['gamma'] ** 2)
    T0 = machine['circumference'] / (beta * c)
    eta = machine['alpha_0'] - machine['gamma'] ** -2
    omega_0 = 2. * np.pi / T0
    radius = machine['circumference'] / (2. * np.pi)
    beta_z = eta * radius / machine['Qs']
    total_energy = machine['gamma'] * machine['particle_mass'] * c ** 2
    z_hat = np.sqrt(2. * beta_z * machine['Jz'])

    machine.update(
        {'beta': beta, 'T0': T0, 'eta': eta, 'omega_0': omega_0,
         'beta_z': beta_z, 'radius': radius, 'total_energy': total_energy,
         'z_hat': z_hat}
    )

def get_average_tuneshift(machine, Qpp):
    """ Computes average tune shift for given airbag bunch and machine
    parameters (see A. Maillard, Eq. 50).
    :param machine: dictionary with main machine and beam parameters
    :param Qpp: second-order chromaticity """
    return ((Qpp * machine['Qs']**2 * machine['z_hat']**2) /
            (4. * machine['eta']**2 * machine['radius']**2))


def get_complex_tuneshift(machine, impedance, Qp=0., Qpp=0., l=0, p_max=120000):
    """ Computes complex tune shift :param delta_Q for given machine and beam parameters
    stored in :param machine_dict and for a specific impedance model :param impedance.
    :param l denotes the azimuthal mode number, and :param p_max represents the
    number of summands to consider (check convergence!). """
    # TODO: automatic convergence check for p_max

    avg_Q = get_average_tuneshift(machine, Qpp)
    p_vect = np.arange(-p_max, p_max + 1)
    omega_p = (p_vect + machine['Q_beta'] + l * machine['Qs']) * machine['omega_0']
    imp_eval = impedance.evaluate(omega_p)

    # Serial versions of hlp2 very slow - hence use openMP parallel one
    hlp2 = hlp2_parallel(
        Q0=machine['Q_beta'], Qs=machine['Qs'], rb=machine['z_hat'],
        R=machine['radius'], eta=machine['eta'],
        Qp=Qp, Qpp=Qpp, l=l, p_max=p_max)

    sum_term = np.sum(hlp2 * imp_eval)

    pre_factor = (
        -1j * machine['number_particles'] * e ** 2 * c /
        (2. * machine['total_energy'] * machine['T0'] ** 2 * machine['Q_beta']))
    delta_Q = pre_factor * sum_term / machine['omega_0'] ** 2 + avg_Q

    return delta_Q


# Machine dictionary with main parameters
SPS = {
    'number_particles': 1e9,  # number of protons in bunch
    'circumference': 6911.,  # [m]
    'gamma': 27.7,  # relativistic gamma
    'Qs': 0.017,  # synchrotron tune
    'alpha_0': 0.00308,  # momentum compaction factor
    'Q_beta': 20.18,  # betatron tune
    'particle_mass': m_p,  # proton mass [kg]
    'Jz': 3e-4  # longitudinal action [m]
}
update_machine_parameters(SPS)

# Define impedance to be used.
# Resonator parameters - peaked (i.e. narrow-band)
# In PyHeadtail we do not use exactly the wake field definition
# made by Chao.Hence, also Chao 's impedance formula should not be applied
# here( in general), because of the different conventions. Basically,
# it is a factor c / omega_res.Chao is consistent in itself.But, for us, to
# compare the analytical formula with PyHeadtail simulations with the same
# resonator parameters, i.e.especially same r_shunt and omega_res) we must
# typically use 'PyHeadtail' convention.
omega_res = 5.022163e8
r_shunt = 1e6 * 5e6  # [Ohm/m**2].
Q = 1e5
resonator = Resonator(r_shunt, omega_res, Q, convention="PyHeadtail")

# Compute tune shift for specific azimuthal mode
Qp = 2
azimuthal_mode = 1
delta_Q = get_complex_tuneshift(SPS, resonator, Qp=Qp, Qpp=0., l=azimuthal_mode)
print('delta_Q', delta_Q)


"""

# only scan in Q', to compare Antoine's Hlp function with Bessel from Chao.
# in case of Q'' = 0, the two should be equivalent. And they are!
calc = True
plot = True

if calc:


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
            deltaQ_res.append(get_complex_tuneshift(Z=Z, Jz=3e-4, Qp=Qp, Qpp=0., l=l,
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
"""
