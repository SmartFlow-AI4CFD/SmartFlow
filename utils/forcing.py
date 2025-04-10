#!/usr/bin/env python3

import os
import numpy as np
import matplotlib.pyplot as plt
from glob import glob
import matplotlib.cm as cm

def main():
    # Get the current working directory
    base_dir = os.getcwd()
    env_dirs = sorted(glob(os.path.join(base_dir, 'envs/env_*')))
    print(base_dir)
    
    # Select only every nth environment
    env_dirs = env_dirs[::20]
    
    # Initialize figure
    plt.figure(figsize=(12, 8))

    # Add horizontal reference line
    plt.axhline(y=-(4.867100498314740759e-02)**2, color='r', linestyle='-', linewidth=2, label='Reference')
    
    # Create colormap for different files
    cmap = cm.get_cmap('viridis', len(env_dirs))
    
    # Process each env directory
    for i, env_dir in enumerate(env_dirs):
        # Extract env number for labeling
        env_name = os.path.basename(env_dir)
        
        # Path to forcing.out file
        forcing_file = os.path.join(env_dir, 'forcing.out')
        
        if os.path.exists(forcing_file):
            try:
                # Load data (first and third columns)
                data = np.loadtxt(forcing_file)
                x = data[:, 0]  # First column
                y = data[:, 1]  # Second column
                
                # Plot the data with a color from the colormap
                plt.plot(x, y, label=env_name, color=cmap(i/len(env_dirs)), alpha=0.7)
                
            except Exception as e:
                print(f"Error reading {forcing_file}: {e}")
    
    # Add plot details
    plt.xlabel('First Column')
    plt.ylabel('Second Column')
    plt.title('Forcing Data (Every nth Environment)')
    plt.grid(True, linestyle='--', alpha=0.7)
    
    # Set the y-axis range
    plt.ylim(-0.0026, -0.0012)
    
    # Add legend with smaller font and place it outside the plot
    plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left', fontsize='small')
    plt.tight_layout()
    
    # Save plot to file
    plt.savefig(os.path.join(base_dir, 'forcing.png'), dpi=300)

if __name__ == "__main__":
    main()
