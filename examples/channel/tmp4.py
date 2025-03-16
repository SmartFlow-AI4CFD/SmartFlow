from smartsim import Experiment
from smartredis import Client
import numpy as np
import torch
from smartsim.database import Orchestrator
from smartsim.log import get_logger
import os

logger = get_logger(__name__)


port = 6120
exp = Experiment("experiment", launcher="local")
db = Orchestrator(port=port, interface="lo")

# remove db files from previous run if necessary
db.remove_stale_files()

# create an output directory for the database log files
exp.generate(db, overwrite=True)

# startup Orchestrator
exp.start(db)

for db_host in db.hosts:
    logger.info(f"$(smart dbcli) -h {db_host} -p {port} shutdown")

logger.info(f"DB address: {db.get_address()[0]}")


exe_args = {
    # "--tag": self.tag[i],
    # "--action_interval": self.n_cfd_time_steps_per_action,
    # "--agent_interval": self.agent_interval,
    # "--restart_file": restart_file,
}
exe_args = [f"{k}={v}" for k,v in exe_args.items()]

run = exp.create_run_settings(
    exe='./mini',
    exe_args=exe_args,
)

cwd = os.getcwd()

model = exp.create_model(
    name="test_0",
    run_settings=run,
    # path=self.cwd,
    path=os.path.join(cwd, "test_0")
)

exp.start(model)

db_address = db.get_address()[0]
os.environ["SSDB"] = db_address
client = Client(
    address=db_address,
    cluster=db.batch
)

client.poll_tensor('example_tensor', 100, 100000000)
state = client.get_tensor('example_tensor')
client.delete_tensor('example_tensor')

print(state)