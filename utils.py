import numpy as np
from scipy.constants import c, e
from scipy.interpolate import interp1d

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
    """ Calculates additional machine parameters and adds them to the existing dictionary.
    :param machine: dictionary containing the main parameters of the accelerator and the
    of the beam.
    :returns: """
    beta = np.sqrt(1. - 1. / machine['gamma'] ** 2)  # relativistic beta
    t0 = machine['circumference'] / (beta * c)  # revolution period [s]
    eta = machine['alpha_0'] - machine['gamma'] ** -2  # slippage factor
    omega_0 = 2. * np.pi / t0  # revolution frequency [rad/s]
    radius = machine['circumference'] / (2. * np.pi)  # machine physical radius [m]
    beta_z = eta * radius / machine['Qs']  # longitudinal beta function [m]
    total_energy = machine['gamma'] * machine['particle_mass'] * c ** 2  # total beam energy [J]
    z_hat = np.sqrt(2. * beta_z * machine['Jz'])  # airbag beam radius [m]

    machine.update(
        {'beta': beta, 't0': t0, 'eta': eta, 'omega_0': omega_0,
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
