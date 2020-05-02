import numpy as np
from scipy.constants import m_p
from progressbar import ProgressBar
import utils as ut


# 1) Define machine dictionary with main parameters
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
ut.update_machine_parameters(SPS)

# 2) Define impedance model to use
# Here we use resonator parameters for narrow band resonator
omega_res = 5.022163e8  # [rad/s]
r_shunt = 1e6 * 5e6  # [Ohm/m**2]
Q = 1e5
resonator = ut.Resonator(r_shunt, omega_res, Q, convention="PyHeadtail")

# 3) First test: compute tune shifts vs. Q'
# Scan variables are Qp and azimuthal_modes
Qp = np.linspace(-2, 6, 5)
azimuthal_modes = np.arange(-1, 3, 1)
delta_q = np.zeros((len(azimuthal_modes), len(Qp)), dtype=np.complex)

pbar = ProgressBar()
for i, l in pbar(enumerate(azimuthal_modes)):
    for j, qp in enumerate(Qp):
        delta_q[i, j] = ut.get_complex_tuneshift(
            machine=SPS, impedance=resonator, Qp=qp, l=l)

# 4) Plot result
ut.display_result(delta_q, Qp, azimuthal_modes, "Q'", "l")
