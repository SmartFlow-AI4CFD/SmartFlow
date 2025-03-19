#!/usr/bin/env python3

import os
import glob
import shutil
import random
import numpy as np
from typing import Any, List, Dict, Union, Optional, Tuple, Sequence

from smartredis import Client
from smartsim.log import get_logger
from smartflow.init_smartsim import init_smartsim

import gymnasium as gym
from gymnasium import spaces

import time
import subprocess

from stable_baselines3.common.vec_env import VecEnv
from stable_baselines3.common.vec_env.base_vec_env import VecEnvObs, VecEnvStepReturn, VecEnvIndices

from abc import abstractmethod

logger = get_logger(__name__)


class CFDEnv(VecEnv):
    """
    An asynchronous, vectorized CFD environment.

    :param num_envs: Number of environments
    :param observation_space: Observation space
    :param action_space: Action space
    """

    def __init__(
        self,
        conf,
        runtime,
    ):
        # Define agents before using in dones/rewards dicts, and one pseudo-env has one agent
        self.possible_agents = [f"agent_{i}" for i in range(conf.environment.n_pseudo_envs)]
        self.agents = self.possible_agents[:]
        
        # Set render_mode early to avoid warnings from VecEnv
        self.render_mode = None
        
        # Store configuration
        self.conf = conf
        
        # Initialize required parameters from config
        self.mode = conf.runner.mode
        self.n_action_steps_per_pseudo_env_episode = conf.runner.n_action_steps_per_pseudo_env_episode
        self.n_vec_envs = conf.environment.n_vec_envs
        self.n_pseudo_envs_per_env = conf.environment.n_pseudo_envs_per_env
        self.n_pseudo_envs = conf.environment.n_pseudo_envs
        self.n_tasks_per_env = conf.environment.n_tasks_per_env
        self.witness_file = conf.environment.witness_file
        self.rectangle_file = conf.environment.rectangle_file
        self.time_key = conf.environment.time_key
        self.step_type_key = conf.environment.step_type_key
        self.state_key = conf.environment.state_key
        self.action_key = conf.environment.action_key
        self.reward_key = conf.environment.reward_key
        self.state_size_key = conf.environment.state_size_key
        self.action_size_key = conf.environment.action_size_key
        self.dtype = conf.environment.dtype
        self.cfd_dtype = conf.environment.cfd_dtype
        self.poll_time = conf.environment.poll_time
        self.env_names = conf.environment.env_names
        self.dump_data_flag = conf.environment.dump_data_flag
        self.cwd = os.getcwd()
        self.total_training_iterations = conf.runner.total_training_iterations
        
        # Action related parameters
        self.n_cfd_time_steps_per_action = conf.environment.n_cfd_time_steps_per_action
        self.agent_interval = conf.environment.agent_interval
        self.action_bounds = conf.environment.action_bounds
        self.reward_beta = conf.environment.reward_beta
        self.t_action = conf.environment.t_action
        self.f_action = conf.environment.f_action
        self.t_episode = conf.environment.t_episode
        self.t_begin_control = conf.environment.t_begin_control

        self.training_iteration = 0
        
        # Data paths
        self.dump_data_path = os.path.join(self.cwd, "dump_data")

        self.runtime = runtime
        self.client = Client(
            address=self.runtime.db_entry, 
            cluster=False,
        )

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
        self.time_key = ["ensemble_" + str(i) + "." + self.time_key for i in range(self.n_vec_envs)]
        self.step_type_key = ["ensemble_" + str(i) + "." + self.step_type_key for i in range(self.n_vec_envs)]
        self.state_key = ["ensemble_" + str(i) + "." + self.state_key for i in range(self.n_vec_envs)]
        self.action_key = ["ensemble_" + str(i) + "." + self.action_key for i in range(self.n_vec_envs)]
        self.reward_key = ["ensemble_" + str(i) + "." + self.reward_key for i in range(self.n_vec_envs)]

        # create exe arguments
        self.tag = [str(i) for i in range(self.n_vec_envs)]
        self.f_action = [str(self.f_action) for _ in range(self.n_vec_envs)]
        self.t_episode = [str(self.t_episode) for _ in range(self.n_vec_envs)]
        self.t_begin_control = [str(self.t_begin_control) for _ in range(self.n_vec_envs)]

        # create ensemble models inside experiment
        self.ensemble = None
        self._episode_ended = False
        self.envs_initialised = False

        self.n_3 = 16 # half channel, so we split the channel into 2 parts in the z direction
        self.n_state_marl = 3
        self.n_state = self.n_pseudo_envs_per_env*self.n_state_marl
        self.n_action = 1
        self.n_reward = 3 + self.n_3
        
        # Track whether the environment has terminated for each agent
        self.dones = {agent: False for agent in self.possible_agents}
        self.rewards = {agent: 0.0 for agent in self.possible_agents}
        self.infos = {agent: {} for agent in self.possible_agents}

        self._state = np.zeros((self.n_vec_envs, self.n_state), dtype=self.dtype)
        self._state_marl = np.zeros((self.n_pseudo_envs, self.n_state_marl), dtype=self.dtype)
        self._action = np.zeros((self.n_vec_envs, self.n_pseudo_envs_per_env * self.n_action), dtype=self.dtype)
        self._local_reward = np.zeros((self.n_vec_envs, self.n_pseudo_envs_per_env * self.n_reward))
        self._reward = np.zeros(self.n_pseudo_envs)
        self._episode_global_step = 0

        # data files
        self.n_envs = len(self.env_names)
        self.envs = [{} for _ in range(self.n_envs)]
        for i in range(self.n_envs):
            env_path = os.path.join(self.cwd, "environments", self.env_names[i])
            self.envs[i]["directory"] = env_path
            self.envs[i]["restart_files"] = glob.glob(os.path.join(env_path, "fld_*.bin"))
            self.envs[i]["stats"] = np.loadtxt(os.path.join(env_path, "stats.txt"))
            self.envs[i]["stats_file"] = glob.glob(os.path.join(env_path, "stats-single-point-chan-?????.out"))[0]
            self.envs[i]["tauw_ref"] = self.envs[i]["stats"][1]**2
            self.envs[i]["ref_vel"] = np.loadtxt(self.envs[i]["stats_file"], usecols=2, max_rows=self.n_3)
            zf = np.loadtxt(self.envs[i]["stats_file"], usecols=1, max_rows=self.n_3)
            dzf = np.zeros(self.n_3)
            dzf[0] = zf[0] - 0.0
            for j in range(1, self.n_3):
                dzf[j] = zf[j] - zf[j-1]
            self.envs[i]["ref_dzf"] = dzf
            self.envs[i]["input.nml"] = os.path.join(env_path, "input.nml")
            self.envs[i]["input.py"] = os.path.join(env_path, "input.py")
        
        # Create proper spaces for VecEnv
        observation_space = spaces.Box(
            low=-np.inf,
            high=np.inf,
            shape=(self.n_state_marl,),
            dtype=np.float32
        )
        
        action_space = spaces.Box(
            low=conf.environment.action_bounds[0],
            high=conf.environment.action_bounds[1],
            shape=(self.n_action,),
            dtype=np.float32
        )
        
        # Initialize VecEnv parent class
        super().__init__(
            num_envs=self.n_pseudo_envs,
            observation_space=observation_space,
            action_space=action_space
        )
        
        # Add step_async required variables
        self._actions = None
        self._waiting = False


    def reset(self) -> VecEnvObs:
        """
        Reset all the environments and return an array of
        observations, or a tuple of observation arrays.

        If step_async is still doing work, that work will
        be cancelled and step_wait() should not be called
        until step_async() is invoked again.

        :return: observation
        """
        
        # Reset dones and rewards
        self.dones = {agent: False for agent in self.agents}
        self.rewards = {agent: 0.0 for agent in self.agents}
        self.infos = {agent: {} for agent in self.agents}

        self.episode_steps = 0
        self.episode_rewards = np.zeros(self.n_pseudo_envs)

        # Close the current CFD simulations
        self._stop_exp()

        # Start the simulation with a new ensemble
        restart_file = self.conf.runner.restart_file if hasattr(self.conf.runner, "restart_file") else 0
        self.ensemble = self._start_exp(restart_file=restart_file, global_step=0)

        self._get_state()
        self._redistribute_state()
        
        # Return numpy array observations instead of dict for VecEnv compatibility
        observations = np.zeros((self.n_pseudo_envs, self.n_state_marl), dtype=self.dtype)
        for i in range(self.n_pseudo_envs):
            observations[i] = self._state_marl[i]
        
        return observations
    

    def step_async(self, actions: np.ndarray) -> None:
        """
        Tell all the environments to start taking a step
        with the given actions.
        Call step_wait() to get the results of the step.

        You should not call this if a step_async run is
        already pending.
        """
        if self._waiting:
            raise ValueError("Async step already in progress")
            
        self._actions = actions
        self._waiting = True

        # Set actions in the CFD environment
        self._set_action(self._actions)


    def step_wait(self) -> VecEnvStepReturn:
        """
        Wait for the step taken with step_async().

        :return: observation, reward, done, information
        """
        if not self._waiting:
            raise ValueError("No async step in progress")

        # Poll new state and reward
        self._get_state()
        self._redistribute_state()
        self._get_reward()
        
        # Format VecEnv return values - observations, rewards, dones, infos
        observations = np.zeros((self.n_pseudo_envs, self.n_state_marl), dtype=self.dtype)
        rewards = np.zeros(self.n_pseudo_envs)
        dones = np.zeros(self.n_pseudo_envs, dtype=bool)
        infos = [{} for _ in range(self.n_pseudo_envs)]
        
        for i in range(self.n_pseudo_envs):
            observations[i] = self._state_marl[i]
            rewards[i] = self._reward[i]

        self.episode_steps += 1
        self.episode_rewards += rewards

        # Check if episode has ended
        self._episode_global_step += 1
        if self._episode_global_step >= self.n_action_steps_per_pseudo_env_episode:
            dones[:] = True
        
        # Write RL data to disk if enabled
        if self.dump_data_flag:
            self._dump_rl_data()
        
        self._waiting = False

        if all(dones):
            for i in range(self.n_pseudo_envs):
                infos[i]["terminal_observation"] = observations[i]
                infos[i]["episode"] = dict(
                    r=self.episode_rewards[i],
                    l=self.episode_steps
                )
            self.training_iteration += 1
            if self.training_iteration >= self.total_training_iterations:
                self.close()
            else:
                observations = self.reset()

        return observations, rewards, dones, infos

    def close(self) -> None:
        """
        Clean up the environment's resources.
        """
        self._stop_exp()

    def get_attr(self, attr_name: str, indices: VecEnvIndices = None) -> list[Any]:
        """
        Return attribute from vectorized environment.

        :param attr_name: The name of the attribute whose value to return
        :param indices: Indices of envs to get attribute from
        :return: List of values of 'attr_name' in all environments
        """
        if indices is None:
            indices = range(self.n_pseudo_envs)
        return [getattr(self, attr_name) for _ in indices]

    def set_attr(self, attr_name: str, value: Any, indices: VecEnvIndices = None) -> None:
        """
        Set attribute inside vectorized environmrender_modeents.

        :param attr_name: The name of attribute to assign new value
        :param value: Value to assign to `attr_name`
        :param indices: Indices of envs to assign value
        :return:
        """
        if indices is None:
            indices = range(self.n_pseudo_envs)
        for _ in indices:
            setattr(self, attr_name, value)

    def env_method(self, method_name: str, *method_args, indices: VecEnvIndices = None, **method_kwargs) -> list[Any]:
        """
        Call instance methods of vectorized environments.

        :param method_name: The name of the environment method to invoke.
        :param indices: Indices of envs whose method to call
        :param method_args: Any positional arguments to provide in the call
        :param method_kwargs: Any keyword arguments to provide in the call
        :return: List of items returned by the environment's method call
        """
        if indices is None:
            indices = range(self.n_pseudo_envs)
        return [getattr(self, method_name)(*method_args, **method_kwargs) for _ in indices]

    def env_is_wrapped(self, wrapper_class: type[gym.Wrapper], indices: VecEnvIndices = None) -> list[bool]:
        """
        Check if environments are wrapped with a given wrapper.

        :param method_name: The name of the environment method to invoke.
        :param indices: Indices of envs whose method to call
        :param method_args: Any positional arguments to provide in the call
        :param method_kwargs: Any keyword arguments to provide in the call
        :return: True if the env is wrapped, False otherwise, for each env queried.
        """
        if indices is None:
            indices = range(self.n_pseudo_envs)
        return [False for _ in indices]  # No wrappers used in this environment
        
    def get_images(self) -> Sequence[Optional[np.ndarray]]:
        """
        Return RGB images from each environment
        """
        return [None for _ in range(self.n_pseudo_envs)]  # No rendering support
    

    # Internal methods below
    
    def _get_state(self):
        """
        Get current flow state from the database.
        """
        for i in range(self.n_vec_envs):
            print(f"Polling state from key: {self.state_key[i]}")
            self.client.poll_tensor(self.state_key[i], 100, self.poll_time)
            print(f"Getting state from key: {self.state_key[i]}")
            try:
                self._state[i, :] = self.client.get_tensor(self.state_key[i])
                self.client.delete_tensor(self.state_key[i])
                logger.info(f"[Env {i}] Got state: {self._state[i, :5]}")
            except Exception as exc:
                raise Warning(f"Could not read state from key: {self.state_key[i]}") from exc


    def _start_exp(self, restart_file=0, global_step=0):
        """Start CFD instances within runtime environment.
            
            Returns:
                List of `smartsim` handles for each started CFD environment.
        """
        seed = self.conf.environment.seed
        rng = random.Random(seed)
        idx = rng.randint(0, self.n_envs-1)
        logger.info(f"Using seed {seed} to select environment {idx} from {self.n_envs} available environments")

        logger.info(f"Using restart files from folder {self.env_names[idx]}")
        self.tauw_ref = self.envs[idx]["tauw_ref"]
        self.ref_vel = self.envs[idx]["ref_vel"]
        self.ref_dzf = self.envs[idx]["ref_dzf"]


        exe_args = []
        exe_name = []
        for i in range(self.n_vec_envs):
            folder_name = f"train_{i}" if self.mode == "train" else f"eval_{i}"
            folder_path = os.path.join(self.cwd, "experiment" ,folder_name)
            if not os.path.exists(folder_path):
                os.makedirs(folder_path)

            if os.path.exists(self.envs[idx]["input.nml"]):
                target_path = os.path.join(folder_path, "input.nml")
                if os.path.exists(target_path):
                    os.remove(target_path)
                shutil.copy(self.envs[idx]["input.nml"], target_path)
            if os.path.exists(self.envs[idx]["input.py"]):
                target_path = os.path.join(folder_path, "input.py")
                if os.path.exists(target_path):
                    os.remove(target_path)
                shutil.copy(self.envs[idx]["input.py"], target_path)
            
            restart_file = rng.choice(self.envs[idx]["restart_files"])
            target_path = os.path.join(folder_path, "fld.bin")
            if os.path.exists(target_path):
                os.remove(target_path)
            shutil.copy(restart_file, target_path)
            logger.info(f"Selected restart file: {os.path.basename(restart_file)}")

            # mpirun -n 2 /scratch/maochao/code/CaLES/build/cales --action_interval=10 --agent_interval=4 --restart_file=../restart/fld.bin
            this_exe_args = {
                "--tag": self.tag[i],
                "--action_interval": self.n_cfd_time_steps_per_action,
                "--agent_interval": self.agent_interval,
                # "--restart_file": restart_file,
            }
            this_exe_args = [f"{k}={v}" for k,v in this_exe_args.items()]
            exe_args.append(this_exe_args)
            exe_name.append(f"train_{self.training_iteration*self.n_vec_envs + i}")


        # Launch executables in runtime
        return self.runtime.launch_models(
            '/scratch/maochao/code/CaLES/build/cales',
            exe_args,
            exe_name,
            self.n_tasks_per_env,
            self.n_vec_envs,
            launcher=self.conf.smartsim.run_command,
        )


    def _dump_rl_data(self):
        """Write RL data into disk."""
        for i in range(self.n_vec_envs):
            with open(os.path.join(self.dump_data_path , "state", f"state_env{i}_eps{self._episode_global_step}.txt"),'a') as f:
                np.savetxt(f, self._state[i, :][np.newaxis], fmt='%.8f', delimiter=' ')
            f.close()
            with open(os.path.join(self.dump_data_path , "reward", f"local_reward_env{i}_eps{self._episode_global_step}.txt"),'a') as f:
                np.savetxt(f, self._local_reward[i, :][np.newaxis], fmt='%.8f', delimiter=' ')
            f.close()
            with open(os.path.join(self.dump_data_path , "action", f"action_env{i}_eps{self._episode_global_step}.txt"),'a') as f:
                np.savetxt(f, self._action[i, :][np.newaxis], fmt='%.8f', delimiter=' ')
            f.close()


    def _stop_exp(self):
        """
        Stop all CFD instances.
        """
        if self.ensemble:
            for i in range(self.n_vec_envs):
                if i < len(self.ensemble) and self.ensemble[i] is not None and not self.runtime.exp.finished(self.ensemble[i]):
                    self.runtime.exp.stop(self.ensemble[i])

    
    def _redistribute_state(self):
        """
        Redistribute state across MARL pseudo-environments.
        """


    def _get_reward(self):
        """
        Obtain the local reward (already computed in SOD2D) from each CFD environment and compute the local/global reward for the problem at hand
        It is better to compute the global reward in python
        """


    def _set_action(self, action):
        """
        Write actions for each environment to be polled by the corresponding SOD2D environment.
        """