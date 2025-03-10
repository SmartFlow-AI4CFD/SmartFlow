import numpy as np

# Read file "data.txt" into an array (columns separated by whitespace)
data = np.loadtxt("train/action/action_env0_eps17616.txt")
import matplotlib.pyplot as plt

# Flatten the data to 1D
flattened_data = data.flatten()

# Create a histogram plot (normalized to form a probability density function)
plt.hist(flattened_data, bins=30, density=True, alpha=0.6, color='blue')

plt.xlabel('Value')
plt.ylabel('Probability Density')
plt.title('Probability Density Function of Data')
plt.xlim(0.000, 0.009)  # Set x-axis limits based on data
plt.savefig("hist.png")


print("Mean:", np.mean(flattened_data))
std_value = np.std(flattened_data)
print("Standard Deviation:", std_value)
mean_value = np.mean(flattened_data)
ratio = std_value / mean_value
print("Ratio of standard deviation to mean:", ratio)