#!/usr/bin/env python3

import numpy as np
import os
import glob
import shutil

from smartsim.log import get_logger
from smartflow.cfd_env import CFDEnv

logger = get_logger(__name__)


class ChannelEnv(CFDEnv):
    """
    Channel environment.
    """
    def __init__(self, conf, runtime):
        super().__init__(conf=conf, runtime=runtime)
        
        for i in range(self.n_cases):
            n_cells = conf.extras.n_cells
            case_path = self.cases[i]["path"]
            self.cases[i]["n_cells"] = n_cells
            self.cases[i]["restart_files"] = glob.glob(os.path.join(case_path, "fld_*.bin"))
            self.cases[i]["stats"] = np.loadtxt(os.path.join(case_path, "stats.txt"))
            self.cases[i]["stats_file"] = glob.glob(os.path.join(case_path, "stats-single-point-chan-?????.out"))[0]
            self.cases[i]["tauw_ref"] = self.cases[i]["stats"][1]**2
            self.cases[i]["vel_ref"] = np.loadtxt(self.cases[i]["stats_file"], usecols=2, max_rows=n_cells)
            zf = np.loadtxt(self.cases[i]["stats_file"], usecols=1, max_rows=n_cells)
            dzf = np.zeros(n_cells)
            dzf[0] = zf[0] - 0.0
            for j in range(1, n_cells):
                dzf[j] = zf[j] - zf[j-1]
            self.cases[i]["dzf_ref"] = dzf
            self.cases[i]["input.nml"] = os.path.join(case_path, "input.nml")
            self.cases[i]["input.py"] = os.path.join(case_path, "input.py")


    def _create_envs(self):
        """
        Start CFD instances within runtime environment. 
        We try to do as little modifications to the CFD solvers.
        """
        # TODO: Move basic key-value pairs of envs to the parent class
        case_idx = self.case_selector.randint(0, self.n_cases-1)
        self.tauw_ref = self.cases[case_idx]["tauw_ref"]
        self.vel_ref = self.cases[case_idx]["vel_ref"]
        self.dzf_ref = self.cases[case_idx]["dzf_ref"]

        for i in range(self.n_cfds):
            env_idx = self.envs[i]["env_idx"]
            this_exe_args = {
                "--tag": f"env_{env_idx:03d}",
                "--action_interval": self.cfd_steps_per_action,
                "--agent_interval": self.agent_interval,
                "--tauw_ref": self.tauw_ref,
            }
            this_exe_args = [f"{k}={v}" for k,v in this_exe_args.items()]
            self.envs[i]["exe_args"] = this_exe_args

            for file_name in ["input.nml", "input.py"]:
                dst = os.path.join(self.envs[i]["exe_path"], file_name)
                if os.path.lexists(dst):
                    os.remove(dst)
                os.symlink(self.cases[case_idx][file_name], dst)

            restart_file = self.restart_selectors[i].choice(self.cases[case_idx]["restart_files"])
            fld_bin_path = os.path.join(self.envs[i]["exe_path"], "fld.bin")
            if os.path.lexists(fld_bin_path):
                os.remove(fld_bin_path)
            os.symlink(restart_file, fld_bin_path)
            print(f"Restart file: {restart_file} for env {i}.")


    def _recalculate_state(self):
        """
        Redistribute state.
        """
        # TODO: Better to remove the indices specification from the environment; just leave it to the user for flexibility
        agents_per_cfd = self.agents_per_cfd
        for i in range(self.n_cfds):
            for j in range(agents_per_cfd):
                for k in range(self.agent_state_dim):
                    start_idx = j * self.cfd_state_dim
                    self._agent_states[i * agents_per_cfd + j, k] = self._cfd_states[i, start_idx + k]


    def _recalculate_reward(self):
        """
        Recalculate reward.
        """
        agents_per_cfd = self.agents_per_cfd
        for i in range(self.n_cfds):
            rewards = self._cfd_rewards[i, :].reshape(agents_per_cfd, self.cfd_reward_dim)

            tauw1 = rewards[:, 0]
            tauw1_prev = rewards[:, 1]
            tauw_ref = self.tauw_ref
            
            bonus = np.where(abs(tauw_ref - tauw1) / tauw_ref < 0.01, 1.0, 0.0)
            local_rewards = -1.0 / tauw_ref * (abs(tauw_ref - tauw1) - abs(tauw_ref - tauw1_prev)) + bonus
            global_reward = 0.0

            agent_indices = slice(i * agents_per_cfd, (i + 1) * agents_per_cfd)
            self._agent_rewards[agent_indices] = self.reward_beta * global_reward + (1.0 - self.reward_beta) * local_rewards


            # # u_profile
            # vel_profile_err = np.sum(self.dzf_ref[0:6] * (reward[:, 3:3+6] - self.vel_ref[0:6])**2, axis=1)
            # vel_profile_err_global = np.sum(self.dzf_ref[0:6] * (np.mean(reward[:, 3:3+6], axis=0) - self.vel_ref[0:6])**2)
            
            # rl_0 = -0.0 * np.abs(reward[:, 0] - 0.8045)  # u, not used any more
            # rg_0 = -0.0 * np.abs(np.mean(reward[:, 0]) - 0.8045)

            # rl_1 = -1.0 * 50.0 * np.abs(reward[:, 1] - self.tauw_ref)  # tauw, 25000
            # rg_1 = -1.0 * 50.0 * np.abs(np.mean(reward[:, 1]) - self.tauw_ref)

            # rl_2 = -0.0 * 100.0 * reward[:, 2]  # u_profile_err  # can be nan even multiplied by zero
            # rg_2 = -0.0 * 100.0 * np.mean(reward[:, 2])  # 100 should be increased here, not used any more
            # rl_3 = -0.0 * 100.0 * vel_profile_err[:]  # u_profile_err
            # rg_3 = -0.0 * 500.0 * vel_profile_err_global
            # rl_4 =  0.0  # tauw_rms
            # rg_4 = -0.0 * 500.0 * np.abs(np.sqrt(np.mean((reward[:, 1] - self.tauw_ref)**2)) - self.tauw_ref / 5.0)  # 500

            # local_rewards = rl_0 + rl_1 + rl_3 + rl_4
            # global_reward = rg_0 + rg_1 + rg_3 + rg_4


    def _recalculate_action(self):
        """
        Redistribute action.
        """
        lower_bound = 0.9
        upper_bound = 1.1
        scaled_actions = lower_bound + 0.5 * (self._agent_actions + 1) * (upper_bound - lower_bound)

        agents_per_cfd = self.agents_per_cfd
        for i in range(self.n_cfds):
            for j in range(agents_per_cfd):
                for k in range(self.cfd_action_dim):
                    n = (i * agents_per_cfd + j) * self.cfd_action_dim
                    self._cfd_actions[i, j * self.cfd_action_dim + k] = scaled_actions[n, k]

        self._scaled_agent_actions = scaled_actions