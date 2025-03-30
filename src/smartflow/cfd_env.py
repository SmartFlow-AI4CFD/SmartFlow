#!/usr/bin/env python3

import os
import glob
import shutil
import random
import numpy as np
from typing import Any, List, Dict, Union, Optional, Tuple, Sequence

from smartredis import Client
from smartsim.log import get_logger

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
    def __init__(self, conf, runtime):

        self.n_total_agents = conf.environment.n_total_agents
        self.agent_state_dim = conf.environment.agent_state_dim
        self.agent_action_dim = conf.environment.agent_action_dim
        self.action_bounds = conf.environment.action_bounds

        observation_space = spaces.Box(
            low=-np.inf,
            high=np.inf,
            shape=(self.agent_state_dim,),
            dtype=np.float32
        )

        action_space = spaces.Box(
            low=self.action_bounds[0],
            high=self.action_bounds[1],
            shape=(self.agent_action_dim,),
            dtype=np.float32
        )

        super().__init__(
            num_envs=self.n_total_agents,
            observation_space=observation_space,
            action_space=action_space
        )

        # Define agents before using in dones/rewards dicts, and one pseudo-env has one agent
        self.possible_agents = [f"agent_{i}" for i in range(self.n_total_agents)]
        self.agents = self.possible_agents[:]
        
        # Set render_mode early to avoid warnings from VecEnv
        self.render_mode = None
        self.conf = conf
        self.runtime = runtime
        
        # Initialize required parameters from config
        self.mode = conf.runner.mode
        self.steps_per_episode = conf.runner.steps_per_episode
        self.n_action_steps_per_batch = conf.runner.steps_per_batch
        self.n_cfds = conf.environment.n_cfds
        self.agents_per_cfd = conf.environment.agents_per_cfd
        self.tasks_per_cfd = conf.environment.tasks_per_cfd
        self.cfd_dtype = conf.environment.cfd_dtype
        self.poll_time = conf.environment.poll_time
        self.save_trajectories = conf.environment.save_trajectories
        self.trajectory_path = conf.environment.trajectory_path
        self.total_iterations = conf.runner.n_iterations
        self.cfd_state_dim = conf.environment.cfd_state_dim
        self.cfd_action_dim = conf.environment.cfd_action_dim
        self.cfd_reward_dim = conf.environment.cfd_reward_dim
        self.exe = conf.environment.executable_path
        self.cfd_steps_per_action = conf.environment.cfd_steps_per_action
        self.agent_interval = conf.environment.agent_interval
        self.reward_beta = conf.environment.reward_beta
        self.cwd = os.getcwd()

        # Initialize parameters applicable to all cases
        self.n_cases = len(conf.environment.case_names)
        self.cases = [{} for _ in range(self.n_cases)]
        for i in range(self.n_cases):
            self.cases[i]["name"] = conf.environment.case_names[i]
            case_path = os.path.join(conf.environment.case_folder, conf.environment.case_names[i])
            self.cases[i]["path"] = case_path

        # Initialize counters
        self.iteration = 0
        self._global_step = 0

        # Initialize smartredis client
        self.client = Client(
            address=self.runtime.db_entry, 
            cluster=False,
        )

        # Initialize states, actions, and rewards by CFD and agent, respectively
        self._cfd_states = np.zeros((self.n_cfds, self.agents_per_cfd * self.cfd_state_dim))
        self._cfd_actions = np.zeros((self.n_cfds, self.agents_per_cfd * self.cfd_action_dim))
        self._cfd_rewards = np.zeros((self.n_cfds, self.agents_per_cfd * self.cfd_reward_dim))
        self._agent_states = np.zeros((self.n_total_agents, self.agent_state_dim))
        self._agent_actions = np.zeros((self.n_total_agents, self.agent_action_dim))
        self._agent_rewards = np.zeros(self.n_total_agents)

        self.dones = {agent: False for agent in self.possible_agents}
        self.rewards = {agent: 0.0 for agent in self.possible_agents}
        self.infos = {agent: {} for agent in self.possible_agents}
        self.models = [None for _ in range(self.n_cfds)]

        self.state_key = [None for _ in range(self.n_cfds)]
        self.action_key = [None for _ in range(self.n_cfds)]
        self.reward_key = [None for _ in range(self.n_cfds)]
        
        # Initialize step_async required variables
        self._actions = None
        self._waiting = False

        # Initialize random number generators
        seeds = self.seed(self.conf.runner.seed)[::self.agents_per_cfd]
        self.case_selector = random.Random(seeds[0])
        self.restart_selectors = [random.Random(seed) for seed in seeds]

        # Initialize trajectory saving
        if self.save_trajectories:
            os.makedirs(os.path.join(self.trajectory_path, "state" ), exist_ok=True)
            os.makedirs(os.path.join(self.trajectory_path, "action"), exist_ok=True)
            os.makedirs(os.path.join(self.trajectory_path, "reward"), exist_ok=True)


    def reset(self) -> VecEnvObs:
        """
        Reset all the environments and return an array of
        observations, or a tuple of observation arrays.

        If step_async is still doing work, that work will
        be cancelled and step_wait() should not be called
        until step_async() is invoked again.

        :return: observations
        """
        # Update keys for the current iteration
        self.envs = [{} for _ in range(self.n_cfds)]
        for i in range(self.n_cfds):
            env_idx = self.iteration * self.n_cfds + i
            self.envs[i]["exe"] = self.conf.environment.executable_path
            self.envs[i]["n_tasks"] = self.tasks_per_cfd
            self.envs[i]["exe_name"] = f"env_{env_idx}"
            exe_path = os.path.join("envs", f"env_{env_idx:03d}")
            if os.path.exists(exe_path):
                shutil.rmtree(exe_path)
            os.makedirs(exe_path)
            self.envs[i]["exe_path"] = exe_path
            self.state_key[i]  = f"env_{(self.iteration * self.n_cfds + i):03d}.state"
            self.action_key[i] = f"env_{(self.iteration * self.n_cfds + i):03d}.action"
            self.reward_key[i] = f"env_{(self.iteration * self.n_cfds + i):03d}.reward"

        # Reset dones and rewards
        self.dones = {agent: False for agent in self.agents}
        self.rewards = {agent: 0.0 for agent in self.agents}
        self.infos = {agent: {} for agent in self.agents}

        self.episode_steps = 0
        self.episode_rewards = np.zeros(self.n_total_agents)
        self._global_step += self.n_action_steps_per_batch

        # Close the current CFD simulations
        self._stop_envs()

        # Start the simulation with new models
        self._create_envs()
        self.models = self._start_envs()
        self._get_state()
        self._recalculate_state()
        
        # Return numpy array observations instead of dict for VecEnv compatibility
        observations = np.zeros((self.n_total_agents, self.agent_state_dim))
        for i in range(self.n_total_agents):
            observations[i] = self._agent_states[i]
        
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
            
        self._agent_actions = actions
        self._waiting = True

        # Set actions in the CFD environment
        self._recalculate_action()
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
        self._recalculate_state()
        self._get_reward()
        self._recalculate_reward()
        
        # Format VecEnv return values - observations, rewards, dones, infos
        observations = np.zeros((self.n_total_agents, self.agent_state_dim))
        rewards = np.zeros(self.n_total_agents)
        dones = np.zeros(self.n_total_agents, dtype=bool)
        infos = [{} for _ in range(self.n_total_agents)]
        
        for i in range(self.n_total_agents):
            observations[i] = self._agent_states[i]
            rewards[i] = self._agent_rewards[i]

        self.episode_steps += 1
        self.episode_rewards += rewards

        # Check if episode has ended
        if self.episode_steps >= self.steps_per_episode:
            dones[:] = True
        
        # Write RL data to disk if enabled
        if self.save_trajectories:
            self._save_trajectories()
        
        self._waiting = False

        if all(dones):
            for i in range(self.n_total_agents):
                infos[i]["terminal_observation"] = observations[i]
                infos[i]["episode"] = dict(
                    r=self.episode_rewards[i],
                    l=self.episode_steps
                )
            self.iteration += 1
            if self.iteration >= self.total_iterations:
                self.close()
            else:
                observations = self.reset()

        return observations, rewards, dones, infos


    def close(self) -> None:
        """
        Clean up the environment's resources.
        """
        self._stop_envs()


    def get_attr(self, attr_name: str, indices: VecEnvIndices = None) -> list[Any]:
        """
        Return attribute from vectorized environment.

        :param attr_name: The name of the attribute whose value to return
        :param indices: Indices of envs to get attribute from
        :return: List of values of 'attr_name' in all environments
        """
        if indices is None:
            indices = range(self.n_total_agents)
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
            indices = range(self.n_total_agents)
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
            indices = range(self.n_total_agents)
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
            indices = range(self.n_total_agents)
        return [False for _ in indices]  # No wrappers used in this environment
        

    def get_images(self) -> Sequence[Optional[np.ndarray]]:
        """
        Return RGB images from each environment
        """
        return [None for _ in range(self.n_total_agents)]  # No rendering support
    

    # Internal methods below
    
    def _get_state(self):
        """
        Get current flow state from the database.
        """
        for i in range(self.n_cfds):
            self.client.poll_tensor(self.state_key[i], 100, self.poll_time)
            try:
                self._cfd_states[i, :] = self.client.get_tensor(self.state_key[i])
                self.client.delete_tensor(self.state_key[i])
            except Exception as exc:
                raise Warning(f"Could not read state from key: {self.state_key[i]}") from exc
            
    
    def _get_reward(self):
        """
        Obtain the local reward from each CFD environment and compute the local/global reward for the problem at hand
        It is better to compute the global reward in python
        """
        for i in range(self.n_cfds):
            self.client.poll_tensor(self.reward_key[i], 100, self.poll_time)
            try:
                self._cfd_rewards[i, :] = self.client.get_tensor(self.reward_key[i])
                self.client.delete_tensor(self.reward_key[i])
            except Exception as exc:
                raise Warning(f"Could not read reward from key: {self.reward_key[i]}") from exc
            

    def _set_action(self, action):
        """
        Write actions for each environment to be polled by the corresponding SOD2D environments.
        """
        for i in range(self.n_cfds):
            self.client.put_tensor(self.action_key[i], self._cfd_actions[i, :].astype(self.cfd_dtype))
            
    
    @abstractmethod    
    def _create_envs(self):
        """Create CFD instances within runtime environment.
            
            Returns:
                List of `smartsim` handles for each started CFD environment.
        """

        raise NotImplementedError
    

    def _start_envs(self):
        """Start CFD instances within runtime environment.
            
            Returns:
                List of `smartsim` handles for each started CFD environment.
        """
        # Launch executables in runtime
        return self.runtime.launch_models(
            exe=[env["exe"] for env in self.envs],
            exe_path=[env["exe_path"] for env in self.envs],
            exe_args=[env["exe_args"] for env in self.envs],
            exe_name=[env["exe_name"] for env in self.envs],
            n_procs=[env["n_tasks"] for env in self.envs],
            n_exe=self.n_cfds,
            launcher=self.conf.smartsim.run_command,
        )    


    def _stop_envs(self):
        """
        Stop all running CFD instances and clean up resources.
        
        This method iterates through all CFD environments and stops them if they
        are running. It handles errors gracefully and logs the stopping process.
        """
        for i in range(self.n_cfds):
            if self.models[i] is None:
                continue
            elif not self.runtime.exp.finished(self.models[i]):
                self.runtime.exp.stop(self.models[i])

        self.models = [None for _ in range(self.n_cfds)]


    @abstractmethod
    def _recalculate_state(self):
        """
        Redistribute state.
        """
        raise NotImplementedError
    

    @abstractmethod
    def _recalculate_reward(self):
        """
        Recalculate reward.
        """
        raise NotImplementedError
    

    @abstractmethod
    def _recalculate_action(self):
        """
        Rescale action.
        """
        raise NotImplementedError
    

    def _save_trajectories(self):
        """Write RL trajectory data into disk following the conventional order: state, action, reward."""
        agents_per_cfd = self.agents_per_cfd
        for i in range(self.n_cfds):
            agent_indices = slice(i * agents_per_cfd, (i + 1) * agents_per_cfd)
            with open(os.path.join(self.trajectory_path, f"state/env{i}_eps{self._global_step}.txt"),'a') as f:
                np.savetxt(f, self._agent_states[agent_indices], fmt='%.8f', delimiter=' ')
            with open(os.path.join(self.trajectory_path, f"action/env{i}_eps{self._global_step}.txt"),'a') as f:
                np.savetxt(f, self._agent_actions[agent_indices], fmt='%.8f', delimiter=' ')
            with open(os.path.join(self.trajectory_path, f"reward/env{i}_eps{self._global_step}.txt"),'a') as f:
                np.savetxt(f, self._agent_rewards[agent_indices], fmt='%.8f', delimiter=' ')