import numpy as np
import matplotlib.pyplot as plt
from matplotlib.pyplot import cm
from scipy.constants import m_p
from progressbar import ProgressBar

import utils as ut

plt.style.use('default')

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

# First test: compute tune shifts vs. Q'
# Scan variables
Qp = np.linspace(-2, 6, 5)
azimuthal_modes = np.arange(-1, 3, 1)
delta_Q = np.zeros((len(azimuthal_modes), len(Qp)), dtype=np.complex)

pbar = ProgressBar()
for i, l in pbar(enumerate(azimuthal_modes)):
    for j, qp in enumerate(Qp):
        delta_Q[i, j] = ut.get_complex_tuneshift(
            machine=SPS, impedance=resonator, Qp=qp, l=l)

# Plot result
fig = plt.figure(figsize=(6.5, 7))
ax1 = fig.add_subplot(211)
ax2 = fig.add_subplot(212, sharex=ax1)
axis_multiplier = 3
cols = cm.plasma(np.linspace(0, 1, len(azimuthal_modes)))

for i, l in enumerate(azimuthal_modes):
    ax1.plot(Qp, delta_Q[i, :].real * 10**axis_multiplier,
             'x-', c=cols[i], label='l={:d}'.format(l))
    ax2.plot(Qp, delta_Q[i, :].imag * 10**axis_multiplier,
             'x-', c=cols[i])

ax1.set_ylabel(r"$10^{:d}$ Re $\Delta Q_c$".format(axis_multiplier))
ax2.set_xlabel(r"$Q'$")
ax2.set_ylabel(r"$-10^{:d}$ Im $\Delta Q_c$".format(axis_multiplier))

ax1.set_xlim((np.min(Qp), np.max(Qp)))

ax1.get_yaxis().get_major_formatter().set_useOffset(False)
ax2.get_yaxis().get_major_formatter().set_useOffset(False)

ax1.legend(loc='best', fontsize=12)
plt.show()
