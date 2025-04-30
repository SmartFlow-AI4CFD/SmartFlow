#!/usr/bin/env python3

import os
import glob
import numpy as np
from smartflow.cfd_env import CFDEnv

class ChannelEnv(CFDEnv):
    """
    Channel environment.
    """
    def __init__(self, conf, runtime):
        super().__init__(conf=conf, runtime=runtime)

        for i in range(self.n_cases):
            case_path = self.cases[i]["path"]
            self.cases[i]["input.nml"] = os.path.join(case_path, "input.nml")
            self.cases[i]["input.py"] = os.path.join(case_path, "input.py")
            self.cases[i]["restart_files"] = glob.glob(os.path.join(case_path, "fld_*.bin"))
            self.cases[i]["stats"] = np.loadtxt(os.path.join(case_path, "stats.txt"))
            self.cases[i]["tauw_ref"] = self.cases[i]["stats"][1]**2
            self.cases[i]["n_cells"] = conf.extras.n_cells
            self.cases[i]["tauw_min_percent"] = conf.extras.tauw_min_percent
            self.cases[i]["tauw_max_percent"] = conf.extras.tauw_max_percent
            self.cases[i]["hwm_min"] = conf.extras.hwm_min
            self.cases[i]["hwm_max"] = conf.extras.hwm_max
            self.cases[i]["kap_log"] = conf.extras.kap_log
            # self.cases[i]["stats_file"] = glob.glob(os.path.join(case_path, "stats-single-point-chan-?????.out"))[0]
            # self.cases[i]["vel_ref"] = np.loadtxt(self.cases[i]["stats_file"], usecols=2, max_rows=n_cells)
            # zf = np.loadtxt(self.cases[i]["stats_file"], usecols=1, max_rows=n_cells)
            # dzf = np.zeros(n_cells)
            # dzf[0] = zf[0] - 0.0
            # for j in range(1, n_cells):
            #     dzf[j] = zf[j] - zf[j-1]
            # self.cases[i]["dzf_ref"] = dzf


    def _create_envs(self):
        """
        Start CFD instances within runtime environment. 
        We try to do as little modifications to the CFD solvers.
        """
        # TODO: Move basic key-value pairs of envs to the parent class

        for i in range(self.n_cfds):
            case_idx = self.envs[i]["case_idx"]
            tauw_ref = self.cases[case_idx]["tauw_ref"]
            tauw_min_percent = self.cases[case_idx]["tauw_min_percent"]
            tauw_max_percent = self.cases[case_idx]["tauw_max_percent"]
            hwm_min = self.cases[case_idx]["hwm_min"]
            hwm_max = self.cases[case_idx]["hwm_max"]
            kap_log = self.cases[case_idx]["kap_log"]
            # self.vel_ref = self.cases[case_idx]["vel_ref"]
            # self.dzf_ref = self.cases[case_idx]["dzf_ref"]
            env_name = self.envs[i]["env_name"]
            this_exe_args = {
                "--tag": env_name,
                "--action_interval": self.cfd_steps_per_action,
                "--agent_interval": self.agent_interval,
                "--tauw_ref_min": tauw_ref * tauw_min_percent,
                "--tauw_ref_max": tauw_ref * tauw_max_percent,
                "--hwm_min": hwm_min,
                "--hwm_max": hwm_max,
                "--cfd_seed": self.cfd_seeds[i],
                "--kap_log": kap_log,
            }
            this_exe_args = [f"{k}={v}" for k, v in this_exe_args.items()]
            self.envs[i]["exe_args"] = this_exe_args

            for file_name in ["input.nml", "input.py"]:
                dst = os.path.join(self.envs[i]["exe_path"], file_name)
                if os.path.lexists(dst):
                    os.remove(dst)
                os.symlink(self.cases[case_idx][file_name], dst)

            restart_file = self.cfd_selectors[i].choice(self.cases[case_idx]["restart_files"])
            fld_bin_path = os.path.join(self.envs[i]["exe_path"], "fld.bin")
            if os.path.lexists(fld_bin_path):
                os.remove(fld_bin_path)
            os.symlink(restart_file, fld_bin_path)
            print(f"Restart file: {restart_file} for env {i}.")


    def _redistribute_state(self, cfd_states):
        """
        Redistribute state.
        
        Args:
            cfd_states (numpy.ndarray): The CFD states array
            
        Returns:
            numpy.ndarray: The agent states array
        """
        agents_per_cfd = self.agents_per_cfd
        agent_states = np.zeros((self.n_agents, self.agent_state_dim))
        
        for i in range(self.n_cfds):
            for j in range(agents_per_cfd):
                for k in range(self.agent_state_dim):
                    kk = self.cfd_state_indices[k]
                    agent_states[i * agents_per_cfd + j, k] = cfd_states[i, j * self.cfd_state_dim + kk]
        
        return agent_states


    def _recalculate_reward(self, cfd_rewards):
        """
        Recalculate reward.
        
        Args:
            cfd_rewards (numpy.ndarray): The CFD rewards array
            
        Returns:
            numpy.ndarray: The agent rewards array
        """
        agents_per_cfd = self.agents_per_cfd
        agent_rewards = np.zeros(self.n_agents)
        
        for i in range(self.n_cfds):
            case_idx = self.envs[i]["case_idx"]
            tauw_ref = self.cases[case_idx]["tauw_ref"]
            rewards = cfd_rewards[i, :].reshape(agents_per_cfd, self.cfd_reward_dim)

            tauw1 = rewards[:, 0]
            tauw1_prev = rewards[:, 1]

            bonus = np.where(abs(tauw_ref - tauw1) / tauw_ref < 0.01, 1.0, 0.0)
            local_rewards = -1.0 / tauw_ref * (abs(tauw_ref - tauw1) - abs(tauw_ref - tauw1_prev)) + bonus
            global_reward = 0.0

            agent_indices = slice(i * agents_per_cfd, (i + 1) * agents_per_cfd)
            agent_rewards[agent_indices] = self.reward_beta * global_reward + (1.0 - self.reward_beta) * local_rewards

        return agent_rewards


    def _scale_action(self, agent_actions):
        """
        Scale action.
        
        Args:
            agent_actions (numpy.ndarray): The agent actions array
            
        Returns:
            numpy.ndarray: The scaled agent actions array
        """
        lower_bound = self.action_scale_min
        upper_bound = self.action_scale_max
        scaled_actions = lower_bound + 0.5 * (agent_actions + 1) * (upper_bound - lower_bound)
        return scaled_actions
