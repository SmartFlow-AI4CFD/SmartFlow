from smartsim import Experiment
from smartredis import Client
import numpy as np
import torch

# REDIS_PORT = 6379  # Use default Redis port
REDIS_PORT = 6394  # Use default Redis port

# Start a new Experiment
exp = Experiment("experiment", launcher="local")

# Create and start the Orchestrator database
db = exp.create_database(db_nodes=1, port=REDIS_PORT, interface="lo")
exp.generate(db)
exp.start(db)

# Connect SmartRedis client
client = Client(address=db.get_address()[0], cluster=False)

# Create sample tensor
# sample_array_1 = torch.from_numpy(np.array([np.arange(9.)]))
sample_array_1 = np.array([np.arange(9.)])
client.put_tensor("script-data-1", sample_array_1)
# client.put_tensor("script-data-1", sample_array_1.numpy().astype(np.float32))

# Get and print the output tensor
sample_array_2 = np.array([np.arange(9.)], dtype=np.float64)
sample_array_2[0:9] = client.get_tensor("script-data-1")

# Stop the database
exp.stop(db)

# print("Type of sample_array_1:", sample_array_1.numpy().astype(np.float32).dtype)
# print("Type of sample_array_2:", sample_array_2[0:9].dtype)
print("sample_array_1:", sample_array_1)