from smartsim import Experiment

exp = Experiment("simple", launcher="local")

settings = exp.create_run_settings("echo", exe_args="Hello World1")
model = exp.create_model("hello_world1", settings, path="hello_world1")

exp.start(model, block=True)
print(exp.get_status(model))


settings = exp.create_run_settings("echo", exe_args="Hello World2")
model = exp.create_model("hello_world2", settings, path="hello_world1")

exp.start(model, block=True)
print(exp.get_status(model))