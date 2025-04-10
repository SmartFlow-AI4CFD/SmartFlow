import numpy as np
import matplotlib.pyplot as plt
from scipy import stats

# Create a small dataset with 2 variables
def generate_sample_data(n_samples=100, seed=42):
    np.random.seed(seed)
    
    # Creating correlated data
    mean = [0, 0]
    cov = [[1, 0.7], [0.7, 1]]  # Correlation of 0.7 between variables
    
    data = np.random.multivariate_normal(mean, cov, n_samples)
    return data

# Generate data
data = generate_sample_data(n_samples=4)

# Compute joint PDF using kernel density estimation
kde = stats.gaussian_kde(data.T)

# Create a grid to evaluate the PDF
x_min, x_max = data[:, 0].min() - 1, data[:, 0].max() + 1
y_min, y_max = data[:, 1].min() - 1, data[:, 1].max() + 1
x_grid, y_grid = np.meshgrid(np.linspace(x_min, x_max, 4), np.linspace(y_min, y_max, 4))
positions = np.vstack([x_grid.ravel(), y_grid.ravel()])

print(data.T)
print(positions)

# Evaluate PDF on the grid
pdf_values = kde(positions)
pdf_values = pdf_values.reshape(x_grid.shape)

# Visualize the data and its joint PDF
plt.figure(figsize=(12, 5))

# Plot original data
plt.subplot(1, 2, 1)
plt.scatter(data[:, 0], data[:, 1], alpha=0.6)
plt.title('Original Data')
plt.xlabel('Variable 1')
plt.ylabel('Variable 2')

# Plot the joint PDF
plt.subplot(1, 2, 2)
plt.contourf(x_grid, y_grid, pdf_values, cmap='viridis')
plt.colorbar(label='Probability Density')
plt.scatter(data[:, 0], data[:, 1], alpha=0.3, color='white')
plt.title('Joint PDF')
plt.xlabel('Variable 1')
plt.ylabel('Variable 2')

plt.tight_layout()
plt.savefig('joint_pdf.png', dpi=300, bbox_inches='tight')

# Print some information about the data and PDF
print(f"Sample size: {len(data)}")
print(f"Correlation between variables: {np.corrcoef(data[:, 0], data[:, 1])[0, 1]:.3f}")
print(f"PDF values range from {pdf_values.min():.6f} to {pdf_values.max():.6f}")
