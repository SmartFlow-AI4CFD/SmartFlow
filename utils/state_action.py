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


state = read_file_to_2d_array('trajectories/state/env_000.dat')
action = read_file_to_2d_array('trajectories/scaled_action/env_000.dat')
reward = read_file_to_2d_array('trajectories/reward/env_000.dat')

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
    action_flat = action.reshape(-1, 1)
else:
    action_flat = action.reshape(-1, action.shape[-1])

# Scalar action, use color map
plt.figure(figsize=(10, 8))
# Set font sizes
plt.rcParams.update({'font.size': 14})  # Increase default font size

scatter = plt.scatter(x_coords, y_coords, c=action_flat.flatten(), cmap='viridis', alpha=0.7)
cbar = plt.colorbar(scatter)
cbar.set_label('Action Value', fontsize=16)
cbar.ax.tick_params(labelsize=14)

plt.xlabel(r'$s_1 = \left(1/\kappa_{wm} - 1/\kappa\right) \ln(h^+_{wm})$', fontsize=16)
plt.ylabel(r'$s_2 = B_{wm}$', fontsize=16)
plt.title(r'Action Values Visualized in State Space, $Re_{\tau} = 05200$', fontsize=18)
plt.xticks(fontsize=14)
plt.yticks(fontsize=14)
plt.grid(True)

# Update x_range to use the full specified range
x_range = np.linspace(-600, 600, 100)
y_line = 5.2 - x_range
plt.plot(x_range, y_line, 'k--', linewidth=2, label=r'$\text{Neutral line: } s_2 = 5.2 - s_1$')
plt.legend(fontsize=14)

# Set fixed axis limits
plt.xlim(-600, 600)
plt.ylim(-600, 600)

plt.savefig('state_action.png', dpi=300, bbox_inches='tight')
