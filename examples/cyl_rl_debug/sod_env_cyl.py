#!/usr/bin/env python3

import numpy as np

from smartsim.log import get_logger
from smartsod2d.utils import numpy_str

from smartsod2d.sod_env import SodEnvBase

logger = get_logger(__name__)

class SodEnvCyl(SodEnvBase):
    """
    SOD2D environment(s) extending the tf_agents.environments.PyEnvironment class.
    The function to implement are: _redistribute_state, _get_reward, _set_action
    Inherits from the SodEnvBase environment, the functions defined here are specific for the reduction of the
    recirculation bubble in a turbulent boundary layer.
    """
    def _redistribute_state(self):
        """
        Redistribute state across MARL pseudo-environments.
        
        NEW VERSION --> Not necessary to specify "witness_xyz = (nwit_x,nwit_y,nwit_z)"
        Make sure the witness points are written in blocks in the z direction.
        Inside each z-block, xy points must be sorted always in the same order
        """

        # Number of witness points per pseudo-environment without neighbors
        n_state_psenv = int(self.n_state/self.marl_n_envs)              

        # Extend origianl state array with the periodic marl neighbors
        end_columns = self._state[:, -self.marl_neighbors*n_state_psenv:]
        start_columns = self._state[:, :self.marl_neighbors*n_state_psenv]
    
        aux_state = np.append(end_columns, self._state, axis=1)         # Append end columns to the starting ones
        aux_state = np.append(aux_state, start_columns, axis=1)         # Append initial columns to the end ones

        # Distribute original state taking groups of "n_state_marls" on each row and append it to the state_marl array
        for i in range(self.cfd_n_envs):
            for j in range(self.marl_n_envs):
                self._state_marl[i*self.marl_n_envs+j,:] = aux_state[i, (j*n_state_psenv):(j*n_state_psenv)+self.n_state_marl]


    def _get_reward(self):
        """
        Obtain the local reward (already computed in SOD2D) from each CFD environment and compute the local/global reward for the problem at hand
        """
        for i in range(self.cfd_n_envs):
            if self._step_type[i] > 0: # environment still running
                self.client.poll_tensor(self.reward_key[i], 100, self.poll_time)
                try:
                    reward = self.client.get_tensor(self.reward_key[i])
                    self.client.delete_tensor(self.reward_key[i])
                    logger.debug(f"[Env {i}] Got reward: {numpy_str(reward)}")

                    local_reward = reward / self.reward_norm
                    self._local_reward[i, :] = local_reward
                    global_reward = np.mean(local_reward)
                    for j in range(self.marl_n_envs):
                        self._reward[i * self.marl_n_envs + j] = self.reward_beta * global_reward + (1.0 - self.reward_beta) * local_reward[j]
                except Exception as exc:
                    raise Warning(f"Could not read reward from key: {self.reward_key[i]}") from exc


    def _set_action(self, action):
        """
        Write actions for each environment to be polled by the corresponding Sod2D environment.
        Action clipping must be performed within the environment: https://github.com/tensorflow/agents/issues/216 when using PPO
        """
        # scale actions and reshape for SOD2D
        action = action * self.action_bounds[1]
        action = np.clip(action, self.action_bounds[0], self.action_bounds[1])
        for i in range(self.cfd_n_envs):
            for j in range(self.marl_n_envs):
                self._action[i, j] = action[i * self.marl_n_envs + j]

        # apply zero-net-mass-flow strategy (For the cylinder it is not necessary to invert Q, as jets are located at the top and bottom surfaces)
        self._action_znmf = np.repeat(self._action, 2, axis=-1)

        # write action into database
        for i in range(self.cfd_n_envs):
            self.client.put_tensor(self.action_key[i], self._action_znmf[i, ...].astype(self.sod_dtype))
            logger.debug(f"[Env {i}] Writing (half) action: {numpy_str(self._action[i, :])}")