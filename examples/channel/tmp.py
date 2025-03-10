from smartsim import Experiment
from smartredis import Client
import numpy as np
import time
from smartsim.database import Orchestrator
import subprocess

def get_slurm_walltime():
    """
    Get the walltime of the current SLURM job
    """
    cmd = 'squeue -h -j $SLURM_JOBID -o "%L"'
    return subprocess.check_output(cmd, shell=True, text=True)[:-2] # it ends with '\n'


REDIS_PORT = 6379  # Use default Redis port

# Start a new Experiment
exp = Experiment("experiment", launcher="slurm")

# Create and start the Orchestrator database
db = Orchestrator(
    launcher="slurm",
    port=REDIS_PORT,
    db_nodes=1, # SlurmOrchestrator supports multiple databases per node
    batch=False, # false if it is launched in an interactive batch job
    time=get_slurm_walltime(),  # this is necessary, otherwise the orchestrator wont run properly
    interface="lo",
    hosts=['soaring'],  # specify hostnames of nodes to launch on (without ip-addresses)
    run_command='mpirun',  # ie. mpirun, srun, etc
    db_per_host=1,  # number of database shards per system host (MPMD), defaults to 1
    single_cmd=True,  # run all shards with one (MPMD) command, defaults to True
)

exp.generate(db)
exp.start(db)

# Connect SmartRedis client
client = Client(address=db.get_address()[0], cluster=False)

ensemble = []
cfd_n_envs = 1
tag = [str(i) for i in range(cfd_n_envs)]
n_tasks_per_env = 1
cwd = "/home/mchao/code/SmartSOD2D/examples/channel"
for i in range(cfd_n_envs):
    exe_args = {
        "--tag": tag[i],
    }
    exe_args = [f"{k}={v}" for k,v in exe_args.items()]
    run_args = {
        'report-bindings': None
    }
    run = exp.create_run_settings(
        exe='cales',
        exe_args=exe_args,
        run_command='mpirun',
        run_args=run_args
    )
    run.set_tasks(n_tasks_per_env)

    model = exp.create_model(
        name="ensemble-" + str(i),
        run_settings=run,
        path=cwd
    )
    ensemble.append(model)

for i in range(cfd_n_envs):
    exp.start(ensemble[i], block=False, summary=False) # non-blocking start of CFD solvers
time.sleep(1)

# Stop the database
exp.stop(db)
print("Experiment finished")