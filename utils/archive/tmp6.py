import socket
import subprocess

from smartsim import Experiment
from smartsim.database import Orchestrator
from smartsim.log import get_logger
from smartredis import Client

import os
import time
import sys

logger = get_logger(__name__)

def get_slurm_walltime():
    """
    Get the walltime of the current SLURM job
    """
    cmd = 'squeue -h -j $SLURM_JOBID -o "%L"'
    return subprocess.check_output(cmd, shell=True, text=True)[:-2] # it ends with '\n'


def get_slurm_hosts():
    """
    Get the host list from the SLURM_JOB_NODELIST environment variable
    """
    hostslist_str = subprocess.check_output(
        "scontrol show hostnames", shell=True, text=True
    )
    return list(set(hostslist_str.split("\n")[:-1]))  # returns unique name of hosts

# allow "auto"
launcher = "auto"
run_command = "auto"
port = 6323
n_vec_envs = 1
walltime = get_slurm_walltime()
hosts = get_slurm_hosts()

print(hosts)

exp = Experiment("mini-experiment", launcher=launcher)
print(f"exp._launcher: {exp._launcher}")

db = exp.create_database(
    port=port,
    interface='ib0',
    hosts=hosts[0],
    run_command=run_command,
)

# db = Orchestrator(
#         launcher=launcher,
#         port=port,
#         db_nodes=1, # SlurmOrchestrator supports multiple databases per node
#         batch=False, # false if it is launched in an interactive batch job
#         time=walltime,  # this is necessary, otherwise the orchestrator wont run properly
#         interface="ib0",
#         hosts=hosts[0],  # specify hostnames of nodes to launch on (without ip-addresses)
#         run_command=run_command,  # ie. mpirun, srun, etc
#         # db_per_host=1,  # number of database shards per system host (MPMD), defaults to 1
#         # single_cmd=True,  # run all shards with one (MPMD) command, defaults to True
# )

# remove db files from previous run if necessary
logger.info("Removing stale files from old database...")
db.remove_stale_files()

# create an output directory for the database log files
logger.info("Creating output directory for database log files...")
exp.generate(db, overwrite=True)

# startup Orchestrator
logger.info("Starting database...")
exp.start(db)

logger.info("If the SmartRedis database isn't stopping properly you can use this command to stop it from the command line:")
for db_host in db.hosts:
    logger.info(f"$(smart dbcli) -h {db_host} -p {port} shutdown")

logger.info(f"DB address: {db.get_address()[0]}")


db_address = db.get_address()[0]
os.environ["SSDB"] = db_address
client = Client(
    address=db_address,
    cluster=db.batch
)


ensemble = []
for i in range(n_vec_envs):
    # mpirun -n 2 /scratch/maochao/code/CaLES/build/cales --action_interval=10 --agent_interval=4 --restart_file=../restart/fld.bin
    # run_args = {
    #     'report-bindings': None,
    # }
    print(hosts[i])

    if i == 0:
        exe = '/leonardo/home/userexternal/mxiao000/code/SmartFlow/examples/channel/mini_key1'
    else:
        exe = '/leonardo/home/userexternal/mxiao000/code/SmartFlow/examples/channel/mini_key2'

    run_args = {
        'mpi': 'pmix_v3',
        'nodelist': hosts[i+1],
        'distribution': 'block:block:block,Pack',
        'cpu-bind': 'verbose',
        # 'exclusive': None,
    }

    run_settings = exp.create_run_settings(
        exe=exe,
        run_command=run_command, # mpirun works, but srun doesn't even for a single node
        run_args=run_args
    )
    run_settings.set_tasks(4)
    
    print(f"model_{i} creating")
    model = exp.create_model(
        name=f"env_{i}",
        run_settings=run_settings,
        # path=self.cwd,
    )
    print(f"model_{i} created")

    ensemble.append(model)

    # exp.start(model, block=False)


for i in range(n_vec_envs):
    print(f"model_{i} starting")
    exp.start(ensemble[i], block=False, summary=False) # non-blocking start of CFD solvers
    print(f"model_{i} started")


for i in range(n_vec_envs):
    if i == 0:
        key = "key1"
    else:
        key = "key2"
    print(f"model_{i} polling")
    client.poll_tensor(key, 100, 1000000)
    print(f"model_{i} polled")
    print(f"model_{i} getting")
    data = client.get_tensor(key)
    print(f"model_{i} got")
    print(data)

# time.sleep(10)  # wait for the CFD solvers to finish

exp.stop(db)
print("Done")


# smartflow, 
# mpirun n1(y)  n2(sometimes 1 env started, but not always)
# srun   n1(y)  n2(sometimes 1 env started, but not always)
