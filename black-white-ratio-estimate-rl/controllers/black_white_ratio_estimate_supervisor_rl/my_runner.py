
import os
import json
import time
import traceback
from copy import deepcopy

from tqdm import tqdm
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
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
        self.publishers_state = [
            rospy.Publisher(f'state_agent_{i}', String, queue_size=10) 
            for i in range(self.num_agents)
        ]
        self.publishers_reward = [
            rospy.Publisher(f'reward_agent_{i}', String, queue_size=10)
            for i in range(self.num_agents)
        ]

        print("Env-{}, Algo-{}, runs total num timesteps-{}/{}. \n".format(
            self.env_name, self.algorithm_name,
            self.train_count, self.max_episodes
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
        print("Env-{}, Algo-{}, runs total num timesteps-{}/{}. \n".format(
            self.env_name, self.algorithm_name,
            self.train_count, self.max_episodes
        ))
        self.log_env(env_info)

        return self.train_count
    
    def eval(self):
        """Collect episodes to evaluate the policy."""
        print("Start Evaluation")
        time.sleep(self.args.aggregation_time)
        columns = ['step']
        columns.extend([f'local_ratio_{i}' for i in range(self.num_agents)])
        columns.extend([f'contribution_{i}' for i in range(self.num_agents)])
        columns.append('global_ratio')
        for i in tqdm(range(self.args.num_eval_episodes), desc="Evaluating"):
            progress_filename = os.path.join(
                self.eval_dir,f'progress_eval_{i}_{self.args.ratio_update_method}.csv'
            )
            df = pd.DataFrame(columns=columns)
            df.to_csv(progress_filename,index=False)
            env_info = self.collect_rollout(phase="eval")
            self.log_env(env_info, suffix="eval_", eval_count=i)

    def collect_rollout(self, phase="train"):
        """
        Collect a rollout and store it in the buffer. All agents share a single policy.

        :return env_info: (dict) contains information about the rollout (total rewards, etc).
        """
        assert phase in ["train", "eval"], f"phase must be either 'train' or 'eval', but got {phase}."
        env: Epuck2Supervisor = self.env
        state = env.reset()
        map_info = env.get_map_info()
        
        episode_states = []
        episode_actions = []
        episode_rewards = []
        episode_dones = []
        contributions = [0] * self.num_agents
        local_step = 0
        start_time = time.time()

        while not rospy.is_shutdown() and local_step < self.buffer_length:
            local_step += 1
            for i in range(self.num_agents):
                agent_data = {
                    'step': local_step,
                    'state': state[i].tolist(),
                    'map': map_info[i].tolist(),
                    'phase': phase,
                    'total_steps': self.total_train_steps,
                    'save_dir': self.save_dir,
                    'log_dir': self.log_dir,
                }
                message = json.dumps(agent_data)
                # rospy.loginfo(f'Publishing data for agent_{i}: {message}')
                self.publishers_state[i].publish(message)
                time.sleep(0.01)
            
            rate = rospy.Rate(10)  # 10Hz
            while not all(self.received_actions) and not rospy.is_shutdown():
                rate.sleep()
            s_, map_info_, r, done, c, _ = env.step(self.actions, phase, self.train_count)
            self.received_actions = [False] * self.num_agents
            for i in range(self.num_agents):
                agent_data = {
                    'reward': float(r[i]),
                    'done': done[i],
                    'next_state': s_[i].tolist(),
                    'next_map': map_info_[i].tolist(),
                    'phase': phase,
                    'total_steps': self.total_train_steps,
                    'contribution': c[i],
                }
                contributions[i] += c[i]
                # rospy.loginfo(f'Publishing data for agent_{i}: {r[i]}')
                self.publishers_reward[i].publish(json.dumps(agent_data))
                time.sleep(0.01)

            # episode_states.append(state)
            # episode_map_info.append(map_info_)
            copy_action = deepcopy(self.actions)
            if self.args.byz_num > 0 and "action" in self.args.byz_style:
                copy_action = copy_action[:-self.args.byz_num]
            episode_actions.append(copy_action)
            episode_rewards.append(r)
            # episode_dones.append(done)
            
            state = s_
            map_info = map_info_
            
            if all(done):
                break
        
        acceleration_rate = env.get_episode_time() / (time.time() - start_time)
        print(f"\033[0;31macceleration rate: {acceleration_rate}\033[0m")
        if phase == "train":
            self.train_count += 1
            # img_name = f'/{phase}_{self.train_count}.jpg'
            self.total_train_steps += local_step * self.num_agents
        elif phase == "eval":
            self.eval_count += 1
            # img_name = f'/{phase}_{self.eval_count}.jpg'
        # env.exportImage(self.image_dir + img_name, 100)
        time.sleep(10)

        env_info = {}
        if phase == "train":
            ratios = env.get_ratio_estimation()
            exp_ratios = env.get_exploration_ratio()
            env_info['global_r_total'] = env.global_reward_total
            env_info['global_r_mean'] = env.global_reward_total / env.aggregation_count
            env_info['global_r_last'] = env.global_reward_last
            for i in range(self.num_agents):
                env_info[f'{i}_episode_r'] = np.sum(np.array(episode_rewards)[:, i])
                env_info[f'{i}_episode_r_mean'] = np.mean(np.array(episode_rewards)[:, i])
                env_info[f'{i}_ratio_estimate'] = ratios[i]
                env_info[f'{i}_exploration_ratio'] = exp_ratios[i]
            env_info['episode_steps'] = local_step
            action_collections = [a for episode in episode_actions for a in episode]
            env_info['global_ratio'] = env.global_ratio_estimation
            env_info['episode_time'] = env.get_episode_time()
            for i in range(4):  # action space
                env_info[f'action_{i}'] = action_collections.count(i) / len(action_collections)
            env_info['convergence'] = env.global_ratio_list
        elif phase == "eval":
            env_info['step'] = list(range(1, local_step + 1))
            for i in range(self.num_agents):
                env_info[f'local_ratio_{i}'] = env.local_ratio_dict[i]
                env_info[f'contribution_{i}'] = contributions[i]
            env_info['global_ratio'] = env.global_ratio_list
        return env_info

    def log(self):
        """See parent class."""
        self.log_env(self.env_infos)
        self.log_clear()

    def log_clear(self):
        """See parent class."""
        self.env_infos = {}
        for i in range(self.num_agents):
            self.env_infos[f'{i}_episode_r'] = []
            self.env_infos[f'{i}_episode_r_mean'] = []
            self.env_infos[f'{i}_ratio_estimate'] = []
            self.env_infos[f'{i}_exploration_ratio'] = []
        self.env_infos['episode_steps'] = []
        self.env_infos['global_ratio'] = []
        self.env_infos['episode_time'] = []
        for i in range(4):  # action space
            self.env_infos[f'action_{i}'] = []
    
    def log_env(self, env_info, suffix=None, eval_count=None):
        """
        Log information related to the environment.
        :param env_info: (dict) contains logging information related to the environment.
        :param suffix: (str) optional string to add to end of keys in env_info when logging. 
        """
        if suffix == "eval_":
            progress_filename = os.path.join(
                self.eval_dir,f'progress_eval_{eval_count}_{self.args.ratio_update_method}.csv'
            )
            df = pd.DataFrame({'step': env_info['step']})
            for i in range(self.num_agents):
                df[f'local_ratio_{i}'] = env_info[f'local_ratio_{i}']
            for i in range(self.num_agents):
                df[f'contribution_{i}'] = env_info[f'contribution_{i}']
                self.tb_writer.add_scalar(f"eval/contribution_{i}", env_info[f'contribution_{i}'], eval_count)
            global_ratio_dict = dict(env_info['global_ratio'])
            df['global_ratio'] = df['step'].map(global_ratio_dict)
            df.to_csv(progress_filename,mode='a',header=False,index=False)
            self.tb_writer.add_scalar("eval/steps", env_info['step'][-1], eval_count)
            self.tb_writer.add_scalar("eval/global_ratio", env_info['global_ratio'][-1][-1], eval_count)
            self.tb_writer.flush()
            try:
                self.plot_eval(env_info, eval_count)
            except Exception as e:
                print(f"Error in plotting eval: {e}")
        else:
            data_env = []
            data_env.append(self.total_train_steps)
            for k, v in env_info.items():
                suffix_k = k if suffix is None else suffix + k
                if k != "convergence":
                    data_env.append(v)
                    if "ratio_estimate" in suffix_k:
                        print(f"\033[0;31m{suffix_k} is {v}\033[0m", end=" ")  # red
                    elif "exploration_ratio" in suffix_k:
                        print(f"\033[0;32m{suffix_k} is {v}\033[0m")  # green
                    else:
                        print(suffix_k + " is " + str(v))
                if self.use_wandb:
                    import wandb
                    wandb.log({suffix_k: v}, step=self.total_train_steps)
                else:
                    if "episode_r_mean" in suffix_k:
                        self.tb_writer.add_scalar(f"episode_r_mean/{suffix_k}", v, self.train_count)
                    elif "episode_r" in suffix_k:
                        self.tb_writer.add_scalar(f"episode_r/{suffix_k}", v, self.train_count)
                    elif "exploration_ratio" in suffix_k:
                        self.tb_writer.add_scalar(f"exploration_ratio/{suffix_k}", v, self.train_count)
                    elif "ratio_estimate" in suffix_k:
                        self.tb_writer.add_scalar(f"ratio_estimate/{suffix_k}", v, self.train_count)
                    elif "action" in suffix_k:
                        self.tb_writer.add_scalar(f"action/{suffix_k}", v, self.train_count)
                    elif "global_r" in suffix_k:
                        self.tb_writer.add_scalar(f"global_r/{suffix_k}", v, self.train_count)
                    elif "convergence" in suffix_k:
                        for s, r in v:
                            self.tb_writer.add_scalar(f"convergence/{self.train_count}", r, s)
                    else:
                        self.tb_writer.add_scalar(suffix_k, v, self.train_count)
            self.tb_writer.flush()
            print()
            progress_filename = os.path.join(self.run_dir,'progress.csv')
            df = pd.DataFrame([data_env])
            df.to_csv(progress_filename,mode='a',header=False,index=False)

    def plot_eval(self, env_info, eval_count):
        plt.figure(figsize=(10, 6))
        for i in range(self.num_agents):
            alpha = 0.5 if i in self.env.byz_robots else 1
            plt.plot(env_info['step'], env_info[f'local_ratio_{i}'], label=f'local_ratio_{i}', linewidth=1.5, alpha=alpha)

        x, y = zip(*env_info['global_ratio'])
        plt.plot(x, y, label='global_ratio', marker='o', linestyle='--', linewidth=2)

        plt.xlabel('step')
        plt.ylabel('ratio')
        plt.ylim(-0.02, 1.02)
        plt.title('local ratios and global ratio over steps')
        plt.legend()
        plt.grid()
        plt.savefig(str(self.eval_dir) + f'/progress_eval_{eval_count}_{self.args.ratio_update_method}.png')
        plt.savefig(str(self.eval_dir) + f'/progress_eval_{eval_count}_{self.args.ratio_update_method}.eps')
        plt.close()
