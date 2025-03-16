# hello_world.py
from smartsim import Experiment

exp = Experiment("hello_world_exp", launcher="slurm")
run = exp.create_run_settings(exe="echo", exe_args="Hello World!")
run.set_tasks(8)
run.set_tasks_per_node(4)

model = exp.create_model("hello_world", run)
exp.start(model, block=True, summary=True)

print(exp.get_status(model))
