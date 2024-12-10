#!/usr/bin/env python3

import os
import glob
import shutil
import random
import numpy as np

from tf_agents.specs import array_spec
from tf_agents.environments import py_environment
from tf_agents.trajectories import time_step as ts

from smartredis import Client
from smartsim.settings import MpirunSettings
from smartsim.settings.settings import create_batch_settings
from smartsim.log import get_logger
from smartsod2d.utils import n_witness_points, n_rectangles, numpy_str, bcolors

logger = get_logger(__name__)

class SodEnvBase(py_environment.PyEnvironment):
    """
    SOD2D environment(s) extending the tf_agents.environments.PyEnvironment class.
    Fields:
    - exp: SmartSim experiment class instance
    - db: SmartSim orchestrator class instance
    - hosts: list of host nodes
    - sod_exe: Sod2D executable
    - cwd: working directory
    - launcher: "local", "slurm", "slurm-split"
    - run_command: "mpirun" ("srun" still not working)
    - cluster_account: project account for Alvis run
    - sod2d_modules: modules file for SOD2D when SOD2D simulations are launched as external jobs
    - episode_walltime: environment walltime for Alvis run
    - cfd_n_envs: number of cdf simulations
    - marl_n_envs: number of marl environments inside a cfd simulation
    - witness_xyz: number of witness points in the (x, y, x) directions
    - marl_neighbors: number of witness blocks selected to compose the state
    - n_tasks_per_env: number of processes (MPI ranks) per environment
    - f_action: action frequency ("how often we send a new action signal)
    - t_episode: episode elapsed time
    - t_begin_control: time to start control
    - action_bounds: bounds for action values
    - dtype: data type for the model
    - sod_dtype: data type for arrays to be sent to Sod2D (actions)
    - poll_time: time waiting for an array to appear in the database (in seconds)
    - witness_file: witness points file
    - rectangle_file: actuators surfaces file
    - sod_JSON: Name of the JSON file used to launch Sod2D (without .json extension)

    This base class is used as template for custom classes that implement its own MARL state, reward, and action functions.
    In summary, users need to defined a class inherited from this one that implements:
        - _redistribute_state(): update self._state_marl
        - _get_reward(): updates self._reward
        - _set_action(): updates self._action, self._action_znmf, and writes action into DB.
    """

    def __init__( # pylint: disable=super-init-not-called
        self,
        exp,
        db,
        hosts,
        sod_exe,
        cwd,
        launcher = "local",
        run_command = "mpirun",
        cluster_account = None,
        sod2d_modules = None,
        episode_walltime = None,
        cfd_n_envs = 2,
        marl_n_envs = 5,
        witness_xyz = (6, 4, 10),
        marl_neighbors = 1,
        n_tasks_per_env = 1,
        n_nodes = 1,
        n_tasks_per_node = 1,
        f_action = 1.0,
        t_episode = 10.0,
        t_begin_control = 0.0,
        action_bounds = (-0.05, 0.05),
        reward_norm = 1.0,
        reward_beta = 0.5,
        mode = "collect",
        dtype = np.float32,
        sod_dtype = np.float64,
        poll_time = 3600, # 2 minutes
        copy_executable = False,
        dump_data_flag = True,
        witness_file = "witness.txt",
        rectangle_file = "rectangleControl.txt",
        time_key = "time",
        step_type_key = "step_type",
        state_key = "state",
        state_size_key = "state_size",
        action_key = "action",
        action_size_key = "action_size",
        reward_key = "reward",
        sod_JSON = "BLFlowSolverIncompDRL",
    ):
        self.exp = exp
        self.db = db
        self.sod_exe = sod_exe
        self.cwd = cwd
        self.launcher = launcher
        self.cluster_account = cluster_account
        self.sod2d_modules = sod2d_modules
        self.episode_walltime = episode_walltime
        self.witness_file = witness_file
        self.rectangle_file = rectangle_file
        self.cfd_n_envs = cfd_n_envs
        self.marl_n_envs = marl_n_envs
        self.n_envs = cfd_n_envs * marl_n_envs
        self.witness_xyz = witness_xyz
        self.marl_neighbors = marl_neighbors
        self.n_tasks_per_env = n_tasks_per_env
        self.run_command = run_command
        self.dtype = dtype
        self.sod_dtype = sod_dtype
        self.poll_time = poll_time * 10 # poll_time in [s], but we query in every 100[ms]
        self.action_bounds = action_bounds
        self.reward_norm = reward_norm
        self.reward_beta = reward_beta
        self.mode = mode
        self.dump_data_flag = dump_data_flag
        self.dump_data_path = os.path.join(self.cwd, "train") if mode == "collect" else os.path.join(self.cwd, "eval")
        self.sod_JSON = sod_JSON

        # preliminary checks
        if 2 * marl_neighbors + 1 > marl_n_envs:
            raise ValueError(f"Number of witness blocks selected to compose the state exceed the number total witness blocks:\n \
                Witness blocks selected: {2 * marl_neighbors + 1}\n \
                Total witness blocks: {marl_n_envs}\n")

        # manage directories
        if self.mode == "eval" and os.path.exists(self.dump_data_path):
            counter = 0
            path = self.dump_data_path + f"_{counter}"
            while os.path.exists(path):
                counter += 1
                path = self.dump_data_path + f"_{counter}"
            os.rename(self.dump_data_path, path)
            logger.info(f"{bcolors.WARNING}The data path `{self.dump_data_path}` exists. Moving it to `{path}`{bcolors.ENDC}")
        if self.dump_data_flag:
            if not os.path.exists(os.path.join(self.dump_data_path, "state")):
                os.makedirs(os.path.join(self.dump_data_path, "state"))
            if not os.path.exists(os.path.join(self.dump_data_path, "reward")):
                os.makedirs(os.path.join(self.dump_data_path, "reward"))
            if not os.path.exists(os.path.join(self.dump_data_path, "action")):
                os.makedirs(os.path.join(self.dump_data_path, "action"))

        # generate ensemble keys
        self.time_key = ["ensemble_" + str(i) + "." + time_key for i in range(self.cfd_n_envs)]
        self.step_type_key = ["ensemble_" + str(i) + "." + step_type_key for i in range(self.cfd_n_envs)]
        self.state_key = ["ensemble_" + str(i) + "." + state_key for i in range(self.cfd_n_envs)]
        self.action_key = ["ensemble_" + str(i) + "." + action_key for i in range(self.cfd_n_envs)]
        self.reward_key = ["ensemble_" + str(i) + "." + reward_key for i in range(self.cfd_n_envs)]
        self.state_size_key = state_size_key
        self.action_size_key = action_size_key

        # connect Python Redis client to an orchestrator database
        self.db_address = db.get_address()[0]
        os.environ["SSDB"] = self.db_address
        self.client = Client(address=self.db_address, cluster=self.db.batch)

        # create SOD2D executable arguments
        self.tag = [str(i) for i in range(self.cfd_n_envs)] # environment tags [0, 1, ..., cfd_n_envs - 1]
        self.f_action = [str(f_action) for _ in range(self.cfd_n_envs)]
        self.t_episode = [str(t_episode) for _ in range(self.cfd_n_envs)]
        self.t_begin_control = [str(t_begin_control) for _ in range(self.cfd_n_envs)]

        # create SOD2D ensemble models inside experiment
        self.ensemble = None
        self._episode_ended = False
        self.envs_initialised = False

        # arrays with known shapes
        self._time = np.zeros(self.cfd_n_envs, dtype=self.dtype)
        self._step_type = -np.ones(self.cfd_n_envs, dtype=int) # init status in -1

        # create and allocate array objects
        self.n_state = n_witness_points(os.path.join(self.cwd, self.witness_file))
        self.n_state_marl = int((2 * self.marl_neighbors + 1) * (self.n_state / self.marl_n_envs))
        if self.marl_n_envs > 1:
            self.n_action = 1
        else:
            n_action = n_rectangles(os.path.join(self.cwd, self.rectangle_file))
            assert np.mod(n_action, 2) == 0, "Number of actions is not divisible by 2, so zero-net-mass-flux \
                strategy cannot be implemented"
            self.n_action = n_action // 2

        self._state = np.zeros((self.cfd_n_envs, self.n_state), dtype=self.dtype)
        self._state_marl = np.zeros((self.n_envs, self.n_state_marl), dtype=self.dtype)
        self._action = np.zeros((self.cfd_n_envs, self.n_action * self.marl_n_envs), dtype=self.sod_dtype)
        self._action_znmf = np.zeros((self.cfd_n_envs, 2 * self.n_action * self.marl_n_envs), dtype=self.sod_dtype)
        self._local_reward = np.zeros((self.cfd_n_envs, self.marl_n_envs))
        self._reward = np.zeros(self.n_envs)
        self._episode_global_step= -1

        # define initial state and action array properties. Omit batch dimension (shape is per (marl) environment)
        self._observation_spec = array_spec.ArraySpec(shape=(self.n_state_marl,), dtype=self.dtype, name="observation")
        self._action_spec = array_spec.BoundedArraySpec(shape=(self.n_action,), dtype=self.dtype, name="action",
            minimum=action_bounds[0], maximum=action_bounds[1])


    def stop(self):
        """
        Stops all SOD2D instances inside launched in this environment.
        """
        if self.exp: self._stop_exp()


    def start(self, new_ensemble=False, restart_file=0, global_step=0):
        """
        Starts all SOD2D instances with configuration specified in initialization.
        """
        if not self.ensemble or new_ensemble:
            self.ensemble = self._create_mpmd_ensemble(restart_file)
            logger.info(f"New ensemble created")
        else:
            self.ensemble.run_settings = self._generate_mpmd_settings(restart_file)
            logger.info(f"Updated ensemble args for SOD2D simulations")


        self.exp.start(self.ensemble, block=False) # non-blocking start of Sod2D solver(s)
        self.envs_initialised = False

        # Check simulations have started
        status = self.get_status()
        logger.info(f"Initial status: {status}")
        assert np.all(status > 0), "SOD2D environments could not start."
        self._episode_global_step = global_step

        # Assert that the same state and action sizes are captured equally in Sod2D and here
        n_state = self._get_n_state()
        n_action = self._get_n_action()

        if n_state != self.n_state or n_action != self.n_action * 2 * self.marl_n_envs:
            raise ValueError(f"State or action size differs between SOD2D and the Python environment: \n \
                SOD2D n_state: {n_state}\n Python env n_state: {self.n_state}\n \
                SOD2D n_action: {n_action}\n Python env n_action * 2 * MARL_envs: {self.n_action * 2 * self.marl_n_envs}")

        # Get the initial state and reward
        self._get_state() # updates self._state
        self._redistribute_state() # updates self._state_marl
        self._get_reward() # updates self._reward

        # Write RL data into disk
        if self.dump_data_flag:
            self._dump_rl_data()


    def _generate_mpmd_settings(self, restart_file):
        # restart files are copied within SOD2D to the output_* folder
        if restart_file == 3:
            restart_step = [random.choice(["1", "2"]) for _ in range(self.cfd_n_envs)]
        else:
            restart_step = [str(restart_file) for _ in range(self.cfd_n_envs)]
        # set SOD2D exe arguments
        sod_args = {"restart_step": restart_step, "f_action": self.f_action,
            "t_episode": self.t_episode, "t_begin_control": self.t_begin_control}
        for i in range(self.cfd_n_envs):
            exe_args = [self.sod_JSON] + [f"--{k}={v[i]}" for k,v in sod_args.items()]
            run = MpirunSettings(exe=self.sod_exe, exe_args=exe_args)
            run.set_tasks(self.n_tasks_per_env)
            if i == 0:
                f_mpmd = run
            else:
                f_mpmd.make_mpmd(run)
        return f_mpmd


    def _create_mpmd_ensemble(self, restart_file):
        f_mpmd = self._generate_mpmd_settings(restart_file)
        batch_settings = None
        # Slurm split configuration. Example for the Alvis cluster. To be generalised.
        #  - SOD2D simulations will be launched as external Slurm jobs loading the modules contained in `sod2d_modules`
        #  - there are 4xA100 GPUs per node in Alvis.
        if self.launcher == "slurm-split":
            gpus_required = self.cfd_n_envs * self.n_tasks_per_env
            n_nodes = int(np.ceil(gpus_required / 4))
            gpus_per_node = '4' if gpus_required >= 4 else gpus_required
            batch_settings = create_batch_settings("slurm", nodes=n_nodes, account=self.cluster_account, time=self.episode_walltime,
                    batch_args={'ntasks':gpus_required, 'gpus-per-node':'A100:' + str(gpus_per_node)})
            batch_settings.add_preamble([". " + self.sod2d_modules])
        return self.exp.create_model("ensemble", f_mpmd, batch_settings=batch_settings, path=self.cwd)


    def _get_n_state(self):
        try:
            self.client.poll_tensor(self.state_size_key, 100, self.poll_time) # poll every 100ms for 1000 times (100s in total)
            state_size = self.client.get_tensor(self.state_size_key)
        except Exception as exc:
            raise Warning(f"Could not read state size from key: {self.state_size_key}") from exc
        logger.debug(f"Read state_size: {state_size}")
        return state_size[0]


    def _get_n_action(self):
        try:
            self.client.poll_tensor(self.action_size_key, 100, self.poll_time)
            action_size = self.client.get_tensor(self.action_size_key)
            logger.debug(f"Read action_size: {action_size}")
        except Exception as exc:
            raise Warning(f"Could not read action size from key: {self.action_size_key}") from exc
        return action_size[0]


    def _get_state(self):
        """
        Get current flow state from the database.
        """
        for i in range(self.cfd_n_envs):
            if self._step_type[i] > 0: # environment still running
                self.client.poll_tensor(self.state_key[i], 100, self.poll_time)
                try:
                    self._state[i, :] = self.client.get_tensor(self.state_key[i])
                    self.client.delete_tensor(self.state_key[i])
                    logger.debug(f"[Env {i}] Got state: {numpy_str(self._state[i, :5])}")
                except Exception as exc:
                    raise Warning(f"Could not read state from key: {self.state_key[i]}") from exc


    def _redistribute_state(self):
        """
        Redistribute state across MARL pseudo-environments.
        Make sure the witness points are written in such that the first moving coordinate is x, then y, and last z.

        The redistributed state is saved in variable self._state_marl
        """
        raise NotImplementedError


    def _get_reward(self):
        """
        Obtain the local unprocessed value of the reward from each CFD environment and compute the local/global reward for the problem at hand
        """
        raise NotImplementedError


    def _get_time(self):
        for i in range(self.cfd_n_envs):
            self.client.poll_tensor(self.time_key[i], 100, self.poll_time)
            try:
                self._time[i] = self.client.get_tensor(self.time_key[i])[0]
                self.client.delete_tensor(self.time_key[i])
                logger.debug(f"[Env {i}] Got time: {numpy_str(self._time[i])}")
            except Exception as exc:
                raise Warning(f"Could not read time from key: {self.time_key[i]}") from exc


    def get_status(self):
        """
        Reads the step_type tensors from database (one for each environment).
        Once created, these tensor are never deleted afterwards
        Types of step_type:
            0: Ended
            1: Initialized
            2: Running
        Returns array with step_type from every environment.
        """
        if not self.envs_initialised: # initialising environments - poll and wait for them to get started
            for i in range(self.cfd_n_envs):
                self.client.poll_tensor(self.step_type_key[i], 100, self.poll_time)
                try:
                    self._step_type[i] = self.client.get_tensor(self.step_type_key[i])[0]
                except Exception as exc:
                    raise Warning(f"Could not read step type from key: {self.step_type_key[i]}.") from exc
            if np.any(self._step_type < 0):
                raise ValueError(f"Environments could not be initialized, or initial step_type could not be read from database. \
                    \n step_type = {self._step_type}")
            self.envs_initialised = True
        else:
            for i in range(self.cfd_n_envs):
                try:
                    self._step_type[i] = self.client.get_tensor(self.step_type_key[i])[0]
                except Exception as exc:
                    raise Warning(f"Could not read step type from key: {self.step_type_key[i]}") from exc
        return self._step_type


    def _set_action(self, action):
        """
        Write actions for each environment to be polled by the corresponding Sod2D environment.
        Action clipping must be performed within the environment: https://github.com/tensorflow/agents/issues/216 when using PPO
        """
        ## Sample structure for this function
        # scale actions and reshape for SOD2D
        # apply zero-net-mass-flow strategy
        # write action into database
        # for i in range(self.cfd_n_envs):
        #     self.client.put_tensor(self.action_key[i], self._action_znmf[i, ...].astype(self.sod_dtype))
        #     logger.debug(f"[Env {i}] Writing action: {numpy_str(self._action[i, :])}")
        raise NotImplementedError


    def _dump_rl_data(self):
        """Write RL data into disk."""
        for i in range(self.cfd_n_envs):
            with open(os.path.join(self.dump_data_path , "state", f"state_env{i}_eps{self._episode_global_step}.txt"),'a') as f:
                np.savetxt(f, self._state[i, :][np.newaxis], fmt='%.4f', delimiter=' ')
            f.close()
            with open(os.path.join(self.dump_data_path , "reward", f"local_reward_env{i}_eps{self._episode_global_step}.txt"),'a') as f:
                np.savetxt(f, self._local_reward[i, :][np.newaxis], fmt='%.4f', delimiter=' ')
            f.close()
            with open(os.path.join(self.dump_data_path , "action", f"action_env{i}_eps{self._episode_global_step}.txt"),'a') as f:
                np.savetxt(f, self._action[i, :][np.newaxis], fmt='%.4f', delimiter=' ')
            f.close()


    def _stop_exp(self):
        """
        Stop SOD2D experiment (ensemble of models) with SmartSim
        """
        # delete data from database
        self.client.delete_tensor(self.state_size_key)
        self.client.delete_tensor(self.action_size_key)
        for i in range(self.cfd_n_envs):
            self.client.delete_tensor(self.step_type_key[i])
        # stop ensemble run
        self.exp.stop(self.ensemble)
        # move ensemble log files
        for f in glob.glob("ensemble-*"):
            shutil.move(f, "sod2d_exp/")


    def __del__(self):
        """
        Properly finalize all launched SOD2D instances within the SmartSim experiment.
        """
        self.stop()


    # PyEnvironment methods override

    # https://www.tensorflow.org/agents/tutorials/2_environments_tutorial
    # https://www.tensorflow.org/agents/api_docs/python/tf_agents/trajectories/TimeStep
    # https://www.tensorflow.org/agents/api_docs/python/tf_agents/trajectories/restart

    # We need to override the following methods
    #     - _reset(self)
    #     - _step(self)
    #     - action_spec(self)
    #     - observation_spec(self)
    #     - batched
    #     - batch_size

    def _reset(self):
        self._episode_ended = False
        return ts.restart(np.array(self._state_marl, dtype=self.dtype), batch_size = self.n_envs)


    def _step(self, action):
        """
        Performs a step (single or multiple dts) in SOD2D environments.
        Here we apply an action and return the new time_step
        """
        # the last action ended the episode. Ignore the current actions, relaunch the environments, and start a new episode
        if self._episode_ended:
            logger.info("Last action ended the episode. Relaunching the environments and starting a new episode")
            return self.reset()

        # send predicted actions into DB for SOD2D to collect and step
        self._set_action(action)

        # poll new state and reward. This waits for the SOD2D to finish the action period and send the state and reward data
        self._get_state() # updates self._state
        self._redistribute_state() # updates self._state_marl
        self._get_reward() # updates self._reward

        # write RL data into disk
        if self.dump_data_flag: self._dump_rl_data()

        # determine if simulation finished
        status = self.get_status()
        self._episode_ended = np.all(status == 0)

        # Return transition if episode ended
        if self._episode_ended:
            logger.debug(f"Episode ended. status = {status}")
            return ts.termination(np.array(self._state_marl, dtype=self.dtype), self._reward)
        # else discount is later multiplied with global discount from user input
        return ts.transition(np.array(self._state_marl, dtype=self.dtype), self._reward, discount=np.ones((self.n_envs,)))


    def observation_spec(self):
        return self._observation_spec


    def action_spec(self):
        return self._action_spec


    @property
    def batched(self):
        """
        Multi-environment flag. Override batched property to indicate that this environment is batched.
        Even if n_envs=1, we run it as a muti-environment (for consistency).
        """
        return True


    @property
    def batch_size(self):
        """
        Override batch size property according to chosen batch size
        """
        return self.n_envs

    # End PyEnvironment methods override
