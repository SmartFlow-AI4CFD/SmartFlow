# run_db_local.py
from smartsim import Experiment

exp = Experiment("local-db", launcher="local")
db = exp.create_database(port=6780,       # database port
                         interface="lo")  # network interface to use

# by default, SmartSim never blocks execution after the database is launched.
exp.start(db)

# launch models, analysis, training, inference sessions, etc
# that communicate with the database using the SmartRedis clients

# stop the database
exp.stop(db)

print("Done.")