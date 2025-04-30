from cales_post import CaNS, Moser
import numpy as np
import matplotlib.pyplot as plt
import scienceplots
from scipy.interpolate import interp1d

# plt.style.use('science')  # lines

# Set font sizes for better readability in papers
SMALL_SIZE = 12
MEDIUM_SIZE = 14
BIGGER_SIZE = 16

plt.rc('font', size=SMALL_SIZE)          # controls default text sizes
plt.rc('axes', titlesize=MEDIUM_SIZE)     # fontsize of the axes title
plt.rc('axes', labelsize=MEDIUM_SIZE)     # fontsize of the x and y labels
plt.rc('xtick', labelsize=SMALL_SIZE)     # fontsize of the tick labels
plt.rc('ytick', labelsize=SMALL_SIZE)     # fontsize of the tick labels
plt.rc('legend', fontsize=SMALL_SIZE)     # legend fontsize
plt.rc('figure', titlesize=BIGGER_SIZE)   # fontsize of the figure title

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

# Create a figure with sufficient size to accommodate the external legend
plt.figure(figsize=(10, 6))

x_log = np.logspace(1, np.log10(1e10), 100)
y_log = 1/0.41 * np.log(x_log) + 5.2
plt.plot(x_log, y_log, 'k-', label='$U^+ = \\frac{1}{0.41} \\ln(y^+) + 5.2$')

# Define colors for different retau values
retau_values = [1000, 5200, 10000, 1e5, 1e6, 1e7, 1e8, 1e9, 1e10]
colors = ['blue', 'red', 'green', 'purple', 'orange', 'brown', 'pink', 'gray', 'cyan']
retau_to_color = dict(zip(retau_values, colors))

# Define line styles for EQWM and RLWM
eqwm_linestyle = '-'   # solid for EQWM
rlwm_linestyle = '--'  # dashed for RLWM

# Keep track of which retau values we've already added to the legend
retau_in_legend = set()

for i in range(len(folders)):
  les = CaNS(folders[i])
  les.read_stats()
  retau = les.retau
  utau = 2.0*les.retau/les.reb
  
  # Determine if this is EQWM or RLWM
  is_eqwm = 'run/' in folders[i]
  linestyle = eqwm_linestyle if is_eqwm else rlwm_linestyle
  
  # Find closest retau value to use for color
  closest_retau = min(retau_values, key=lambda x: abs(x - retau))
  color = retau_to_color[closest_retau]
  
  # Use the pre-defined labels list for all entries
  label = labels[i]
  
  plt.plot(les.zc*retau, les.u/utau, linestyle=linestyle, color=color, label=label)

  if i == 12:
    print(les.reb)
    print(les.retau)
    print(les.u)
    print(les.u/utau)

plt.xscale('log')
# Place legend outside of the plot area
plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left', borderaxespad=0.)
# plt.legend(fontsize=MEDIUM_SIZE)
plt.xlabel('$y^+$', fontsize=BIGGER_SIZE)
plt.ylabel('$U^+$', fontsize=BIGGER_SIZE)
plt.tick_params(axis='both', which='major', labelsize=MEDIUM_SIZE)
# plt.xlim([10, 1E10])
# plt.ylim([10, 28])
# plt.yticks(np.arange(10, 30, 4, dtype=int))
# plt.show()

# Adjust layout to make room for the legend
plt.tight_layout()
plt.savefig(f"loglaw.png", bbox_inches='tight', dpi=300)  # Increased DPI for better quality
plt.close()