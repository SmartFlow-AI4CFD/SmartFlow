#!/usr/bin/env python3

import numpy as np
import os
import glob

from smartsim.log import get_logger
from smartflow.cfd_env import CFDEnv

logger = get_logger(__name__)


class CustomEnv(CFDEnv):

    def _redistribute_state(self):
        """
        Redistribute state across MARL pseudo-environments.
        """

        n_state_psenv = int(self.n_state/self.n_pseudo_envs_per_env)

        # Distribute original state taking groups of "n_state_marls" on each row and append it to the state_marl array
        # n_state_psenv = n_state_marl, if there is no neighbor
        for i in range(self.n_vec_envs):
            for j in range(self.n_pseudo_envs_per_env):
                self._state_marl[i*self.n_pseudo_envs_per_env+j,:] = self._state[i, (j*n_state_psenv):(j*n_state_psenv)+self.n_state_marl]


    def _get_reward(self):
        """
        Obtain the local reward (already computed in SOD2D) from each CFD environment and compute the local/global reward for the problem at hand
        It is better to compute the global reward in python
        """
        for i in range(self.n_vec_envs):
            self.client.poll_tensor(self.reward_key[i], 100, self.poll_time)
            try:
                self._local_reward[i, :] = self.client.get_tensor(self.reward_key[i])
                self.client.delete_tensor(self.reward_key[i])

                reward = self._local_reward[i, :].reshape(self.n_pseudo_envs_per_env, self.n_reward)

                # u_profile
                vel_profile_err = np.zeros(self.n_pseudo_envs_per_env)
                for j in range(self.n_pseudo_envs_per_env):
                    vel_profile_err[j] = np.sum(self.ref_dzf[0:6]*(reward[j,3:3+6] - self.ref_vel[0:6])**2) # 0.15
                
                vel_profile_err_global = np.sum(self.ref_dzf[0:6]*(np.mean(reward[:,3:3+6],axis=0) - self.ref_vel[0:6])**2)
                
                #
                rl_0 = -0.0*np.abs(        reward[:,0]  - 0.8045) # u, not used any more
                rg_0 = -0.0*np.abs(np.mean(reward[:,0]) - 0.8045)


                rl_1 = -1.0*50.0 *np.abs(        reward[:,1]  - self.tauw_ref) # tauw, 25000
                rg_1 = -1.0*50.0 *np.abs(np.mean(reward[:,1]) - self.tauw_ref)

                
                rl_2 = -0.0*100.0 *        reward[:,2] # u_profile_err  # can be nan even multiplied by zero
                rg_2 = -0.0*100.0 *np.mean(reward[:,2]) # 100 should be increased here, not used any more
                rl_3 = -0.0*100.0 *vel_profile_err[:] # u_profile_err
                rg_3 = -0.0*500.0 *vel_profile_err_global
                rl_4 =  0.0 # tauw_rms
                rg_4 = -0.0*500.0*np.abs(np.sqrt(np.mean((reward[:,1] - self.tauw_ref)**2)) - self.tauw_ref/5.0) # 500

                local_reward  =  rl_0 + rl_1 + rl_3 + rl_4
                global_reward =  rg_0 + rg_1 + rg_3 + rg_4


                # logger.info(f"rl_1: {rl_1}, rg_1: {rg_1}, rl_3: {rl_3}, rg_3: {rg_3}, rl_4: {rl_4}, rg_4: {rg_4}")
                # logger.info(f"rl_1: {rl_1}, rg_1: {rg_1}, rl_3: {rl_3}, rg_3: {rg_3}")
                # print(f"rl_1: {rl_1}, rg_1: {rg_1}, rl_3: {rl_3}, rg_3: {rg_3}")

                for j in range(self.n_pseudo_envs_per_env):
                    self._reward[i * self.n_pseudo_envs_per_env + j] = self.reward_beta * global_reward + (1.0 - self.reward_beta) * local_reward[j]
                logger.info(f"[Env {i}] Global reward: {global_reward}")
            except Exception as exc:
                raise Warning(f"Could not read reward from key: {self.reward_key[i]}") from exc


    def _set_action(self, action):
        """
        Write actions for each environment to be polled by the corresponding SOD2D environment.
        """

        lower_bound = 0.001
        upper_bound = 0.009
        scaled_action = lower_bound + 0.5 * (action + 1) * (upper_bound - lower_bound)

        for i in range(self.n_vec_envs):
            for j in range(self.n_pseudo_envs_per_env):
                for k in range(self.n_action):   # action is a single value
                    self._action[i, j * self.n_action + k] = scaled_action[i * self.n_pseudo_envs_per_env + j, k]

        # write action into database
        for i in range(self.n_vec_envs):
            self.client.put_tensor(self.action_key[i], self._action[i, :].astype(self.cfd_dtype))
            print(f"Put action: {self._action[i, :].astype(self.cfd_dtype)}")
