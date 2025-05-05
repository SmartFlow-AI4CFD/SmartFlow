import numpy as np
import matplotlib.pyplot as plt

def read_data_file(filename):
    """
    Read a file with float values into a 2D numpy array.
    
    Args:
        filename (str): Path to the data file
        
    Returns:
        numpy.ndarray: 2D array of float values
    """
    try:
        data = np.loadtxt(filename)
        return data
    except Exception as e:
        print(f"Error reading file: {e}")
        return None

def plot_column(data, column_index, title=None):
    """
    Plot a specific column from the data array.
    
    Args:
        data (numpy.ndarray): 2D array of float values
        column_index (int): Index of the column to plot
        title (str, optional): Plot title
    """
    if data is None or column_index >= data.shape[1]:
        print("Invalid data or column index")
        return
    
    plt.figure(figsize=(10, 6))
    plt.plot(data[:, column_index])
    plt.xlabel('Row Index')
    plt.ylabel(f'Column {column_index} Value')
    plt.title(title or f'Plot of Column {column_index}')
    plt.grid(True)
    plt.ylim(0.9, 1.1)
    plt.savefig(f'agent_{column_index}_action_vs_step.png')

def main():
    # Get filename from user input
    filename = input("Enter the path to the data file: ")
    
    # Read the data
    data = read_data_file(filename)
    
    if data is not None:
        print(f"Successfully read data with shape: {data.shape}")
        print(f"The file has {data.shape[1]} columns and {data.shape[0]} rows.")
        
        # Get column to plot from user input
        try:
            col_index = int(input(f"Enter the column index to plot (0 to {data.shape[1]-1}): "))
            if 0 <= col_index < data.shape[1]:
                plot_column(data, col_index)
            else:
                print(f"Invalid column index. Must be between 0 and {data.shape[1]-1}")
        except ValueError:
            print("Please enter a valid integer for the column index.")

if __name__ == "__main__":
    main()
