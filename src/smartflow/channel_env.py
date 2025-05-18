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

        self.action_start_step = conf.environment.action_start_step
        self.action_start_time = conf.environment.action_start_time
        self.cfd_steps_per_action = conf.environment.cfd_steps_per_action
        self.time_duration_per_action = conf.environment.time_duration_per_action
        self.agent_interval = conf.environment.agent_interval
        self.reward_beta = conf.environment.reward_beta
        self.reward_definition = conf.environment.reward_definition
        self.state_definition = conf.environment.state_definition

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
            env_idx = self.envs[i]["env_idx"]
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
                "--action_start_step": self.action_start_step,
                "--action_start_time": self.action_start_time,
                "--cfd_steps_per_action": self.cfd_steps_per_action,
                "--time_duration_per_action": self.time_duration_per_action,
                "--agent_interval": self.agent_interval,
                "--tauw_ref_min": tauw_ref * tauw_min_percent,
                "--tauw_ref_max": tauw_ref * tauw_max_percent,
                "--hwm_min": hwm_min,
                "--hwm_max": hwm_max,
                "--cfd_seed": self.cfd_seeds[i] + env_idx, # Each CFD instance has a different seed
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
    

    def _recalculate_state(self, cfd_states):
        """
        Recalculate state.
        
        Args:
            cfd_states (numpy.ndarray): The CFD states array
            
        Returns:
            numpy.ndarray: The agent states array
        """
        agents_per_cfd = self.agents_per_cfd
        agent_states = np.zeros((self.n_agents, self.agent_state_dim))

        for i in range(self.n_cfds):
            case_idx = self.envs[i]["case_idx"]
            states = cfd_states[i, :].reshape(agents_per_cfd, self.cfd_state_dim)

            hwm_plus = states[:, 0]
            vel_h_plus = states[:, 1]
            dveldz_plus = states[:, 2]

            agent_indices = slice(i * agents_per_cfd, (i + 1) * agents_per_cfd)
            if self.state_definition == "default":
                agent_states[agent_indices, 0] = hwm_plus
                agent_states[agent_indices, 1] = vel_h_plus
                agent_states[agent_indices, 2] = dveldz_plus
            
            elif self.state_definition == "hwm+vel":
                agent_states[agent_indices, 0] = hwm_plus
                agent_states[agent_indices, 1] = vel_h_plus

            elif self.state_definition == "log(hwm)+vel":
                agent_states[agent_indices, 0] = np.log(hwm_plus)
                agent_states[agent_indices, 1] = vel_h_plus
                
            elif self.state_definition == "kap+b":
                kap = 1.0 / (hwm_plus * dveldz_plus)
                b = vel_h_plus - 1.0 / kap * np.log(hwm_plus)
                agent_states[agent_indices, 0] = 1.0 / kap
                agent_states[agent_indices, 1] = b

            elif self.state_definition == "kap_corrected+b":
                kap = 1.0 / (hwm_plus * dveldz_plus)
                b = vel_h_plus - 1.0 / kap * np.log(hwm_plus)
                kap_log = self.cases[case_idx]["kap_log"]
                agent_states[agent_indices, 0] = (1.0 / kap - 1.0 / kap_log) * np.log(hwm_plus)
                agent_states[agent_indices, 1] = b
            
            else:
                raise ValueError(f"Unknown state definition: {self.state_definition}.")
        
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

            if self.reward_definition == "default":
                bonus = np.where(abs(tauw_ref - tauw1) / tauw_ref < 0.01, 1.0, 0.0)
                local_rewards = -1.0 / tauw_ref * (abs(tauw_ref - tauw1) - abs(tauw_ref - tauw1_prev)) + bonus
            
            elif self.reward_definition == "bonus_off":
                local_rewards = -1.0 / tauw_ref * (abs(tauw_ref - tauw1) - abs(tauw_ref - tauw1_prev))

            elif self.reward_definition == "improvement_off":
                local_rewards = np.where(abs(tauw_ref - tauw1) / tauw_ref < 0.01, 1.0, 0.0)

            elif self.reward_definition == "simple_reward":
                local_rewards = -abs(tauw_ref - tauw1) / tauw_ref
            
            else:
                raise ValueError(f"Unknown reward definition: {self.reward_definition}.")

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
