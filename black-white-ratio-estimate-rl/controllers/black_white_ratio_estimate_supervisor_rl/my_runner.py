
import os
import json
import time
import traceback
from copy import deepcopy

import wandb
import numpy as np
import pandas as pd
import rospy
from std_msgs.msg import String

from base_runner import BaseRunner
from black_white_ratio_estimate_supervisor_rl import Epuck2Supervisor

class MyRunner(BaseRunner):
    def __init__(self, config):
        super(MyRunner, self).__init__(config)
        rospy.init_node('runner_publisher', anonymous=True)
        self.actions = [None] * self.num_agents
        self.action_subscribers = [
            rospy.Subscriber(f'action_{i}', String, self.action_callback, callback_args=i) 
            for i in range(self.num_agents)
        ]
        self.received_actions = [False] * self.num_agents
        self.publishers = [
            rospy.Publisher(f'agent_{i}', String, queue_size=10) 
            for i in range(self.num_agents)
        ]

        print("\n Env-{}, Algo-{}, runs total num timesteps-{}/{}. \n".format(
            self.env_name, self.algorithm_name,
            self.total_train_steps, self.max_steps,
        ))
        self.log_clear()
    
    def action_callback(self, data, agent_id):
        """Callback function for receiving actions from the agents."""
        # print(f"data: {data}, type: {type(data)}")
        self.actions[agent_id] = json.loads(data.data)
        self.received_actions[agent_id] = True
        
    def run(self):
        # train
        env_info = self.collect_rollout(phase="train")
        # print(env_info)
        for k, v in env_info.items():
            self.env_infos[k].append(v)
        # # save
        # if self.use_save and (self.total_train_steps - self.last_save_T) / self.save_interval >= 1:
        #     self.last_save_T = self.total_train_steps
        #     self.save()
        # log
        if self.total_train_steps > 5 and ((self.total_train_steps - self.last_log_T) / self.log_interval) >= 1:
            self.last_log_T = self.total_train_steps
            self.log()
        # eval
        if self.use_eval and ((self.total_train_steps - self.last_eval_T) / self.eval_interval) >= 1:
            self.last_eval_T = self.total_train_steps
            print("start evaluation")
            self.eval()

        return self.total_train_steps
    
    def eval(self):
        """Collect episodes to evaluate the policy."""
        eval_infos = {
            f'episode_r_{i}': [] for i in range(self.num_agents)
        }
        eval_infos['episode_time'] = []
        eval_infos['ratio_estimate'] = []
        eval_infos['exploration_rate'] = []
        for i in range(4):  # action space
            eval_infos[f'action_{i}'] = []

        for _ in range(self.args.num_eval_episodes):
            env_info = self.collect_rollout(phase="eval")
            for k, v in env_info.items():
                eval_infos[k].append(v)

        self.log_env(eval_infos, suffix="eval_")

    def collect_rollout(self, phase="train"):
        """
        Collect a rollout and store it in the buffer. All agents share a single policy.

        :return env_info: (dict) contains information about the rollout (total rewards, etc).
        """
        assert phase in ["train", "eval"], f"phase must be either 'train' or 'eval', but got {phase}."
        env_info = {
            f'episode_r_{i}': 0 for i in range(self.num_agents)
        }
        env: Epuck2Supervisor = self.env
        env.reset()
        local_step = 0
        init_actions = [0] * self.num_agents  # forward
        state, map_info, reward, done, info = env.step(init_actions)
        episode_states, episode_actions, episode_rewards = [], [], []
        episode_map_info = []
        episode_states.append(state)  # avail info in [:-1, ], shape of [steps, num_agents, state_dim=11]
        episode_map_info.append(map_info)  # avail info in [:-1, ]
        episode_actions.append(init_actions)  # avail info in [1:, ]
        episode_rewards.append(reward)  # avail info in [1:, ]

        episode_dones = {local_step: done}  # avail info in [1:, ]
        episode_infos = {local_step: info}  # avail info in [1:, ]

        while local_step < self.episode_length and not rospy.is_shutdown():
            local_step += 1
            if local_step == self.args.exclude_steps:
                env.reset_visited()
            for i in range(self.num_agents):
                agent_data = {
                    'state': state[i].tolist(),
                    'map': map_info.tolist(),
                    'reward': float(reward[i]),
                    'done': done[i],
                    'info': info[i],
                    'phase': phase,
                    'total_steps': self.total_train_steps,
                    'save_dir': self.save_dir
                }
                message = json.dumps(agent_data)
                # rospy.loginfo(f'Publishing data for agent_{i}: {message}')
                self.publishers[i].publish(message)
            
            rate = rospy.Rate(10)  # 10Hz
            while not all(self.received_actions) and not rospy.is_shutdown():
                rate.sleep()
            # print(f"======== info: {info[0]}, actions: {self.actions[0]} ==========")
            s_, map_info_, r_, done_, info_ = env.step(self.actions)
            # import pdb; pdb.set_trace()
            episode_states.append(s_)
            episode_map_info.append(map_info_)
            episode_actions.append(deepcopy(self.actions))
            episode_rewards.append(r_)
            episode_dones[local_step] = done_
            episode_infos[local_step] = info_
            
            state = s_
            map_info = map_info_
            reward = r_
            done = done_
            info = info_
            
            if all(done_):
                break
        
        if phase == "train":
            self.total_train_steps += local_step * self.num_agents
        
        for i in range(self.num_agents):
            agent_data = {
                'state': state[i].tolist(),
                'map': map_info.tolist(),
                'reward': float(reward[i]),
                'done': done[i],
                'info': info[i],
                'phase': phase,
                'total_steps': self.total_train_steps,
                'save_dir': self.save_dir
            }
            message = json.dumps(agent_data)
            # rospy.loginfo(f'Publishing data for agent_{i}: {message}')
            self.publishers[i].publish(message)
        if all(done):
            time.sleep(10)
            
        for i in range(self.num_agents):
            env_info[f'episode_r_{i}'] = np.sum(np.array(episode_rewards)[1 + self.args.exclude_steps:, i])
        action_collections = [a for episode in episode_actions for a in episode]
        env_info['episode_time'] = env.get_episode_time()
        env_info['ratio_estimate'] = env.get_ratio_estimation()
        env_info['exploration_rate'] = sum(env.env_tiles.tiles_visited) / (env.col * env.row)
        for i in range(4):  # action space
            env_info[f'action_{i}'] = action_collections.count(i) / len(action_collections)
        return env_info

    def log(self):
        """See parent class."""
        print("\n Env-{}, Algo-{}, runs total num timesteps-{}/{}. \n".format(
            self.env_name, self.algorithm_name,
            self.total_train_steps, self.max_steps,
        ))
        # for p_id, train_info in zip(self.policy_ids, self.train_infos):
        #     self.log_train(p_id, train_info)

        self.log_env(self.env_infos)
        self.log_clear()

    def log_clear(self):
        """See parent class."""
        self.env_infos = {
            f'episode_r_{i}': [] for i in range(self.num_agents)
        }
        self.env_infos['episode_time'] = []
        self.env_infos['ratio_estimate'] = []
        self.env_infos['exploration_rate'] = []
        for i in range(4):  # action space
            self.env_infos[f'action_{i}'] = []
    
    def log_env(self, env_info, suffix=None):
        """
        Log information related to the environment.
        :param env_info: (dict) contains logging information related to the environment.
        :param suffix: (str) optional string to add to end of keys in env_info when logging. 
        """
        data_env = []
        data_env.append(self.total_train_steps)
        for k, v in env_info.items():
            # if k == 'ratio_estimate':
            #     mean_v_each = np.mean(v, axis=0).tolist()
            #     data_env.extend(mean_v_each)
            #     mean_v = np.mean(mean_v_each)
            #     std_v = np.std(mean_v_each)
            #     data_env.append(mean_v)
            #     data_env.append(std_v)
            #     suffix_k = k if suffix is None else suffix + k 
            #     print(suffix_k + " is " + str(mean_v_each))
            #     print(suffix_k + "_mean is " + str(mean_v))
            #     print(suffix_k + "_std is " + str(std_v))
            #     self.tb_logger(f"{suffix_k}_mean", mean_v, self.total_train_steps)
            #     self.tb_logger(f"{suffix_k}_std", std_v, self.total_train_steps)
            # else:
            if len(v) > 0:
                v = np.mean(v)
                data_env.append(v)
                suffix_k = k if suffix is None else suffix + k 
                print(suffix_k + " is " + str(v))
                if self.use_wandb:
                    wandb.log({suffix_k: v}, step=self.total_train_steps)
                else:
                    self.tb_logger(suffix_k, v, self.total_train_steps)
                    # self.writter.add_scalars("episode_info", {suffix_k: v}, self.total_train_steps)
        print()
        if suffix=="eval_":
            progress_filename = os.path.join(self.run_dir,'progress_eval.csv')
        else:
            progress_filename = os.path.join(self.run_dir,'progress.csv')
        df = pd.DataFrame([data_env])
        df.to_csv(progress_filename,mode='a',header=False,index=False) 
        
    def log_train(self, policy_id, train_info):
        """
        Log information related to training.
        :param policy_id: (str) policy id corresponding to the information contained in train_info.
        :param train_info: (dict) contains logging information related to training.
        """
        data_env = []
        data_env.append(self.total_train_steps)
        for k, v in train_info.items():
            policy_k = str(policy_id) + '/' + k
            data_env.append(v)
            if self.use_wandb:
                wandb.log({policy_k: v}, step=self.total_train_steps)
            else:
                self.writter.add_scalars(policy_k, {policy_k: v}, self.total_train_steps)
                
        progress_filename = os.path.join(self.run_dir,'progress_train.csv')
        df = pd.DataFrame([data_env])
        df.to_csv(progress_filename,mode='a',header=False,index=False)   