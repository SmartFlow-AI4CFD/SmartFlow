import numpy as np

def read_file_to_2d_array(file_path, delimiter=None, dtype=float):
    """
    Read a file with n columns into a 2D array.
    
    Args:
        file_path (str): Path to the file.
        delimiter (str, optional): Delimiter used in the file. If None, whitespace is used.
        dtype (type, optional): Data type for the array elements. Default is float.
        
    Returns:
        numpy.ndarray or list: 2D array containing the file data.
    """
    data = []
    
    with open(file_path, 'r') as file:
        for line in file:
            # Skip empty lines and comments
            line = line.strip()
            if not line or line.startswith('#'):
                continue
            
            # Split the line and convert to the specified data type
            values = line.split(delimiter)
            try:
                row = [dtype(val) for val in values]
                data.append(row)
            except ValueError as e:
                print(f"Skipping invalid line: {line}. Error: {e}")
    
    # Convert to numpy array if possible
    try:
        return np.array(data, dtype=dtype)
    except:
        return data


state = read_file_to_2d_array('trajectories/state/env_00000.dat')
action = read_file_to_2d_array('trajectories/scaled_action/env_00000.dat')
reward = read_file_to_2d_array('trajectories/reward/env_00000.dat')

# Reshape state into a 3D array with first dimension as number of rows and third dimension as 2
rows, cols = state.shape

# Check if columns can be evenly divided by 2
if cols % 2 != 0:
    raise ValueError("Number of columns must be even to reshape into third dimension of 2")

# Calculate the middle dimension
middle_dim = cols // 2

# Reshape the state array into 3D (rows, middle_dim, 2)
state_3d = state.reshape(rows, middle_dim, 2)

print(f"Original state shape: {state.shape}")
print(f"Reshaped state shape: {state_3d.shape}")
print(f"Original action shape: {action.shape}")
print(f"Original reward shape: {reward.shape}")

# Visualization of the state-action map
import matplotlib.pyplot as plt
from scipy import stats

# Handle the case where state has one more row than action (common in RL trajectories)
# This typically happens because the final state doesn't have a corresponding action
if state_3d.shape[0] == action.shape[0] + 1:
    print(f"Note: State has one more row than action. Truncating state to match action dimensions.")
    state_3d = state_3d[:action.shape[0], :, :]

# Extract x and y coordinates from state_3d
x_coords = state_3d[:, :, 0].flatten()
y_coords = state_3d[:, :, 1].flatten()

print(f"X coordinates shape: {x_coords.shape}")
print(f"Y coordinates shape: {y_coords.shape}")

# Prepare action data for visualization
if len(action.shape) == 1:
    action_flat = action.flatten()
else:
    action_flat = action.reshape(-1)

# Filter state points based on action > 1.05
mask = action_flat > 1.05
filtered_x = x_coords[mask]
filtered_y = y_coords[mask]

# Filter state points based on action < 0.95
mask_095 = action_flat < 0.95
filtered_x_095 = x_coords[mask_095]
filtered_y_095 = y_coords[mask_095]

print(f"Number of points with action > 1.05: {len(filtered_x)}")
print(f"Number of points with action < 0.95: {len(filtered_x_095)}")

# Create a joint PDF using binned histogram approach instead of KDE
if len(filtered_x) > 0 and len(filtered_x_095) > 0:
    # Define grid boundaries for histogram
    xmin, xmax = -120, 120
    ymin, ymax = -120, 120
    
    # Number of bins for the histogram
    nbins = 500
    
    # Create the figure
    plt.figure(figsize=(10, 8))
    
    # Set global font size
    plt.rcParams.update({'font.size': 14})
    
    # Create 2D histogram for action > 1.05 (red contours)
    H_red, x_edges, y_edges = np.histogram2d(filtered_x, filtered_y, 
                                          bins=nbins,
                                          range=[[xmin, xmax], [ymin, ymax]])
    
    # No smoothing - use raw histogram data
    
    # Normalize to create a PDF
    H_red_pdf = H_red / H_red.sum()
    
    # Use a red colormap with many colors instead of just a few discrete colors
    red_cmap = plt.cm.Reds
    
    # Create coordinate arrays for contour plotting
    x_centers = (x_edges[1:] + x_edges[:-1]) / 2
    y_centers = (y_edges[1:] + y_edges[:-1]) / 2
    X, Y = np.meshgrid(x_centers, y_centers)
    
    # Plot red contours for action > 1.05 with more levels for more colors
    contour_red = plt.contour(X, Y, H_red_pdf.T, levels=15, cmap=red_cmap)
    
    # Create 2D histogram for action < 0.95 (blue contours)
    H_blue, _, _ = np.histogram2d(filtered_x_095, filtered_y_095, 
                               bins=nbins, 
                               range=[[xmin, xmax], [ymin, ymax]])
    
    # No smoothing - use raw histogram data
    
    # Normalize to create a PDF
    H_blue_pdf = H_blue / H_blue.sum()
    
    # Use a blue colormap with many colors instead of just a few discrete colors
    blue_cmap = plt.cm.Blues
    
    # Plot blue contours for action < 0.95 with more levels for more colors
    contour_blue = plt.contour(X, Y, H_blue_pdf.T, levels=15, cmap=blue_cmap)
    
    # Add the neutral line
    x_range = np.linspace(-120, 120, 100)  # Updated to match the fixed range
    # y_line = 5.2 - x_range
    hwm_plus = 01E10 * 0.075
    y_line = -x_range * np.log(hwm_plus) + 1/0.41 * np.log(hwm_plus) + 5.2
    # plt.plot(x_range, y_line, 'k--', linewidth=2, label=r'$\text{Neutral line: } s_2 = 5.2 - s_1$')
    plt.plot(x_range, y_line, 'k--', linewidth=2, label=r'$\text{Neutral line: }$')
    hwm_plus = 01E10 * 0.15
    y_line = -x_range * np.log(hwm_plus) + 1/0.41 * np.log(hwm_plus) + 5.2
    plt.plot(x_range, y_line, 'r--', linewidth=2, label=r'$\text{Neutral line: }$')
    
    # Create custom legend elements for the contour plots
    red_line = plt.Line2D([0], [0], color=red_cmap(0.8), linewidth=2, label='Action > 1.05')
    blue_line = plt.Line2D([0], [0], color=blue_cmap(0.8), linewidth=2, label='Action < 0.95')
    
    # Get the existing handles and labels
    handles, labels = plt.gca().get_legend_handles_labels()
    # Add our new custom handles
    handles.extend([red_line, blue_line])
    
    # Set the plot limits explicitly
    plt.xlim(-5, 10)
    plt.ylim(-100, 100)
    
    # Increase font size for specific elements
    # plt.xlabel(r'$s_1 = \left(1/\kappa_{wm} - 1/\kappa\right) \ln(h^+_{wm})$', fontsize=16)
    plt.xlabel(r'$s_1 = 1/\kappa_{wm}$', fontsize=16)
    plt.ylabel(r'$s_2 = B_{wm}$', fontsize=16)
    # plt.title(r'Joint PDF of States: Red for Action > 1.05, Blue for Action < 0.95$', fontsize=18)
    plt.xticks(fontsize=14)
    plt.yticks(fontsize=14)
    plt.grid(True)
    plt.legend(handles=handles, fontsize=14)
    
    plt.savefig('state_action_jpdf.png', dpi=300, bbox_inches='tight')
else:
    print("No points found with action > 1.05. Cannot create joint PDF.")