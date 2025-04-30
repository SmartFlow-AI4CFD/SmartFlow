from cales_post import CaNS, Moser
import numpy as np
import matplotlib.pyplot as plt
import scienceplots
from scipy.interpolate import interp1d
import os

folders = [
  '/leonardo/home/userexternal/mxiao000/run/CHA_CPG_RETAU01000_H0.1_SMAG_AR1_NX128_NY48_NZ32_MINIMAL/',
  '/leonardo/home/userexternal/mxiao000/run/CHA_CPG_RETAU05200_H0.1_SMAG_AR1_NX128_NY48_NZ32_MINIMAL/',
  '/leonardo/home/userexternal/mxiao000/run/CHA_CPG_RETAU10000_H0.1_SMAG_AR1_NX128_NY48_NZ32_MINIMAL/',
  '/leonardo/home/userexternal/mxiao000/run/CHA_CPG_RETAU01E05_H0.1_SMAG_AR1_NX128_NY48_NZ32_MINIMAL/',
  '/leonardo/home/userexternal/mxiao000/run/CHA_CPG_RETAU01E06_H0.1_SMAG_AR1_NX128_NY48_NZ32_MINIMAL/',
  '/leonardo/home/userexternal/mxiao000/run/CHA_CPG_RETAU01E07_H0.1_SMAG_AR1_NX128_NY48_NZ32_MINIMAL/',
  '/leonardo/home/userexternal/mxiao000/run/CHA_CPG_RETAU01E08_H0.1_SMAG_AR1_NX128_NY48_NZ32_MINIMAL/',
  '/leonardo/home/userexternal/mxiao000/run/CHA_CPG_RETAU01E09_H0.1_SMAG_AR1_NX128_NY48_NZ32_MINIMAL/',
  '/leonardo/home/userexternal/mxiao000/run/CHA_CPG_RETAU01E10_H0.1_SMAG_AR1_NX128_NY48_NZ32_MINIMAL/',
  'eval_retau_01000/envs/env_00000/',
  'eval_retau_05200/envs/env_00000/',
  'eval_retau_10000/envs/env_00000/',
  'eval_retau_01E05/envs/env_00000/',
  'eval_retau_01E06/envs/env_00000/',
  'eval_retau_01E07/envs/env_00000/',
  'eval_retau_01E08/envs/env_00000/',
  'eval_retau_01E09/envs/env_00000/',
  'eval_retau_01E10/envs/env_00000/',
]

labels = [
  'EQWM $Re_\\tau = 1000$', 
  'EQWM $Re_\\tau = 5200$', 
  'EQWM $Re_\\tau = 10000$', 
  'EQWM $Re_\\tau = 10^{5}$',
  'EQWM $Re_\\tau = 10^{6}$',
  'EQWM $Re_\\tau = 10^{7}$',
  'EQWM $Re_\\tau = 10^{8}$',
  'EQWM $Re_\\tau = 10^{9}$',
  'EQWM $Re_\\tau = 10^{10}$',
  'RLWM $Re_\\tau = 1000$', 
  'RLWM $Re_\\tau = 5200$', 
  'RLWM $Re_\\tau = 10000$',
  'RLWM $Re_\\tau = 10^{5}$',
  'RLWM $Re_\\tau = 10^{6}$',
  'RLWM $Re_\\tau = 10^{7}$',
  'RLWM $Re_\\tau = 10^{8}$',
  'RLWM $Re_\\tau = 10^{9}$',
  'RLWM $Re_\\tau = 10^{10}$',
]

# Extract Re_tau values from stats.txt files
eqwm_retau = []
rlwm_retau = []
eqwm_cf = []
rlwm_cf = []

for i in range(9):  # 9 different Re_tau values
    eqwm_folder = folders[i]
    rlwm_folder = folders[i+9]
    
    # Read EQWM Re_tau
    eqwm_file = os.path.join(eqwm_folder, 'results/stats.txt')
    with open(eqwm_file, 'r') as f:
        tmp = f.readline().split()
        eqwm_retau.append(float(tmp[0]))
        eqwm_cf.append(float(tmp[4]))
    
    
    # Read RLWM Re_tau
    rlwm_file = os.path.join(rlwm_folder, 'results/stats.txt')
    with open(rlwm_file, 'r') as f:
        tmp = f.readline().split()
        rlwm_retau.append(float(tmp[0]))
        rlwm_cf.append(float(tmp[4]))

# Calculate relative errors
rel_errors = [(rlwm - eqwm) / eqwm * 100 for rlwm, eqwm in zip(rlwm_retau, eqwm_retau)]

# Calculate relative errors for C_f
rel_errors_cf = [(rlwm - eqwm) / eqwm * 100 for rlwm, eqwm in zip(rlwm_cf, eqwm_cf)]

# Print values for reference
print("EQWM Re_tau values:", eqwm_retau)
print("RLWM Re_tau values:", rlwm_retau)
print("Relative errors in Re_tau (%):", rel_errors)
print("Relative errors in Cf (%):", rel_errors_cf)

# Create plots
plt.figure(figsize=(10, 6))
plt.semilogx(eqwm_retau, rel_errors, 'o-', linewidth=2, markersize=8, label=r'$Re_\tau$')
plt.semilogx(eqwm_retau, rel_errors_cf, 's--', linewidth=2, markersize=8, label=r'$C_f$')
plt.xlabel(r'$Re_\tau$ (EQWM)')
plt.ylabel('Relative Error (%)')
plt.grid(True, which='both', linestyle='--', alpha=0.7)
plt.title('Relative Error between RLWM and EQWM')
plt.axhline(y=0, color='k', linestyle='-', alpha=0.3)  # Add a reference line at y=0
plt.legend()
plt.tight_layout()
plt.savefig('retau_relative_error.png', dpi=300)
