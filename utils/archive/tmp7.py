class MyResource:
    def __init__(self, name):
        self.name = name
        print(f"{self.name}: Initialized")

    def __enter__(self):
        print(f"{self.name}: Entering context")
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        print(f"{self.name}: Exiting context")
        if exc_type:
            print(f"{self.name}: An exception occurred: {exc_type.__name__} - {exc_val}")
        # Return False to propagate exceptions if any
        return False

    def __del__(self):
        print(f"{self.name}: Object deleted (cleanup in __del__)")

def main():
    # Using MyResource as a context manager
    with MyResource("ContextResource") as res:
        print(f"{res.name}: In the context block")
    
    # Creating an instance without a 'with' statement so __exit__ is not used.
    res2 = MyResource("NonContextResource")
    print(f"{res2.name}: Doing some work outside context")

    # Explicitly delete the object to see __del__
    del res2

    # Force garbage collection (optional) to see __del__ output immediately
    # import gc
    # gc.collect()

if __name__ == '__main__':
    main()



# test
# slurm, compute nodes 2 srun(yes) mpirun (no)
# local mpirun(yes)
# compute node 1, srun(), mpirun(yes)