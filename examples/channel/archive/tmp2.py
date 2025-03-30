import numpy as np


def read_line_from_file(file_path, line_number=65):
    """
    Read a specific line from a file and convert it to a numpy array
    
    Args:
        file_path (str): Path to the file
        line_number (int): Line number to read (default: 65)
        
    Returns:
        np.ndarray: Numpy array containing the data from the specified line
    """
    with open(file_path, 'r') as file:
        # Skip to the desired line (line numbers are 1-indexed)
        for _ in range(line_number - 1):
            next(file, None)
        
        # Read the target line
        line = next(file, None)
        
        if line is None:
            raise ValueError(f"Line {line_number} not found in file {file_path}")
            
        # Convert the line to a numpy array (assuming space-separated values)
        # Adjust the delimiter and dtype as needed for your specific data format
        return np.array([float(x) for x in line.strip().split()])


data = np.zeros((6, 96, 19))
line_number = np.array([65, 108, 151])
for i in range(3):
    file_path = "/scratch/maochao/code/SmartSOD2D/examples/channel/experiment/.smartsim/telemetry/experiment/2e1fe22/model/train_0/train_0.out"
    data_1 = read_line_from_file(file_path, line_number=line_number[i]).reshape(48, 19)
    file_path = "/scratch/maochao/code/SmartSOD2D/examples/channel/experiment/.smartsim/telemetry/experiment/74161d8/model/train_1/train_1.out"
    data_2 = read_line_from_file(file_path, line_number=line_number[i]).reshape(48, 19)
    data[i] = np.vstack([data_1, data_2])

for i in range(3):
    file_path = "/scratch/maochao/code/SmartSOD2D/examples/channel/experiment/.smartsim/telemetry/experiment/7a9b867/model/train_0/train_0.out"
    data_1 = read_line_from_file(file_path, line_number=line_number[i]).reshape(48, 19)
    file_path = "/scratch/maochao/code/SmartSOD2D/examples/channel/experiment/.smartsim/telemetry/experiment/692990d/model/train_1/train_1.out"
    data_2 = read_line_from_file(file_path, line_number=line_number[i]).reshape(48, 19)
    data[i+3] = np.vstack([data_1, data_2])


def read_lines_to_array(file_path, start_line=77, end_line=124):
    """
    Read multiple lines from a file into a numpy array
    
    Args:
        file_path (str): Path to the file
        start_line (int): First line to read (inclusive)
        end_line (int): Last line to read (inclusive)
        
    Returns:
        np.ndarray: Numpy array containing the data from the specified lines
    """
    lines_data = []
    
    with open(file_path, 'r') as file:
        # Skip to the start line
        for _ in range(start_line - 1):
            next(file, None)
        
        # Read lines from start_line to end_line
        for _ in range(end_line - start_line + 1):
            line = next(file, None)
            clean_line = line.replace('[','').replace(']','').strip()
            if line is None:
                break
            
            # Convert each line to array and add to our list
            line_data = np.array([float(x) for x in clean_line.strip().split()])
            lines_data.append(line_data)
    
    # Stack all lines into a single array
    return np.vstack(lines_data)

# Example usage
data_py = np.zeros((6, 96, 19))
file_path = "/scratch/maochao/code/SmartSOD2D/examples/channel/job.out"
start_line = np.array([78, 535, 992, 1461, 1918, 2375])
end_line = start_line + (2830-2375)
for i in range(6):
    data_py[i] = read_lines_to_array(file_path, start_line=start_line[i], end_line=end_line[i]).reshape(96, 19)


for i in range(6):
    difference = (data_py[i]-data[i])/data[i]
    is_significant = np.abs(difference) > 1e-8
    print(is_significant.sum())