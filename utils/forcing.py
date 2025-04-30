from cales_post import CaNS, Moser
import numpy as np
import matplotlib.pyplot as plt
import scienceplots
from scipy.interpolate import interp1d

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

# Create figure and axis
fig, ax = plt.subplots(figsize=(8, 6))

# Define colors for different Reynolds numbers
colors = plt.cm.tab10(np.linspace(0, 1, 9))  # 9 distinct colors

# Loop through each folder and plot the forcing.out data
for i, folder in enumerate(folders):
    try:
        # Read the forcing.out file
        forcing_file = folder + 'forcing.out'
        data = np.loadtxt(forcing_file)
        
        # Extract first and second columns
        x = data[:, 0]  # First column
        y = data[:, 4]  # Second column
        
        # Set line style based on whether it's EQWM or RLWM
        # Use same color for matching Reynolds numbers
        if i < 9:  # EQWM
            linestyle = '-'  # solid
            color_idx = i
        else:      # RLWM
            linestyle = '-'  # solid
            color_idx = i - 9  # Match with corresponding EQWM
            
        # Plot the data with appropriate style
        ax.plot(x, y, label=labels[i], linestyle=linestyle, color=colors[color_idx], linewidth=0.8)
    except Exception as e:
        print(f"Error processing {folder}: {e}")


ax.axvline(x=3200, color='black', linestyle='--', linewidth=0.8)

# Customize the plot
ax.set_xlabel('Time')
# ax.set_ylabel('Body force')
ax.set_ylabel('Bulk velocity')
ax.set_ylim(0.9, 1.1)  # Set y-axis range between 0.9 and 1.1
# ax.set_title('Forcing Data Comparison')

# Add legend (with smaller font and outside the plot)
ax.legend(fontsize='small', loc='upper left', bbox_to_anchor=(1.05, 1))
# ax.legend()

# Adjust layout to make room for the legend
plt.tight_layout()

# Save the figure
plt.savefig('forcing.png', dpi=300, bbox_inches='tight')

