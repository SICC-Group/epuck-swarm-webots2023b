import os
import time

import wandb
import torch
import numpy as np
import pandas as pd
from tensorboardX import SummaryWriter
from tensorboard_logger import configure, log_value

class BaseRunner(object):
    """Base class for training recurrent policies."""
    def __init__(self, config):
        """
        Base class for training recurrent policies.
        :param config: (dict) Config dictionary containing parameters for training.
        """
        self.args = config["args"]
        self.env_infos = {}
        self.algorithm_name = self.args.algorithm_name
        self.env_name = self.args.env_name
        self.max_steps = self.args.max_steps
        self.episode_length = self.args.episode_length
        self.use_wandb = self.args.use_wandb
        self.use_eval = self.args.use_eval
        self.eval_interval = self.args.eval_interval
        self.save_interval = self.args.save_interval
        self.log_interval = self.args.log_interval
        self.train_count, self.eval_count = 0, 0
        self.gamma = self.args.gamma
        self.use_save = self.args.use_save
        # self.total_env_steps = 0  # total environment interactions collected during training
        self.total_train_steps = 0  # number of gradient updates performed
        self.last_train_episode = 0  # last episode after which a gradient update was performed
        self.last_eval_T = 0  # last episode after which a eval run was conducted
        self.last_save_T = 0  # last epsiode after which the models were saved
        self.last_log_T = 0 # last timestep after which information was logged

        self.num_agents = config["num_agents"]
        self.agent_ids = [i for i in range(self.num_agents)]
        self.env = config["env"]
        # no parallel envs
        self.num_envs = 1
        self.action_repr_updating = True
        # dir
        self.model_dir = self.args.model_dir
        if self.use_wandb:
            self.save_dir = str(wandb.run.dir)
            self.run_dir = config["run_dir"]
        else:
            self.run_dir = config["run_dir"]
            self.log_dir = str(self.run_dir / 'logs')
            if not os.path.exists(self.log_dir):
                os.makedirs(self.log_dir)
            configure(self.log_dir)
            self.tb_logger = log_value

            self.image_dir = str(self.run_dir / 'images')
            if not os.path.exists(self.image_dir):
                os.makedirs(self.image_dir)

            self.save_dir = str(self.run_dir / 'models')
            if not os.path.exists(self.save_dir):
                os.makedirs(self.save_dir)
        
        
    def run(self):
        """Collect a training episode and perform appropriate training, saving, logging, and evaluation steps."""
        raise NotImplementedError
     
    def save(self):
        """Save all policies to the path specified by the config."""
        for pid in self.policy_ids:
            policy_critic = self.policies[pid].critic
            critic_save_path = self.save_dir + '/' + str(pid)
            if not os.path.exists(critic_save_path):
                os.makedirs(critic_save_path)
            torch.save(policy_critic.state_dict(),
                       critic_save_path + '/critic.pt')

            policy_actor = self.policies[pid].actor
            actor_save_path = self.save_dir + '/' + str(pid)
            if not os.path.exists(actor_save_path):
                os.makedirs(actor_save_path)
            torch.save(policy_actor.state_dict(),actor_save_path + '/actor.pt')

    def restore(self):
        """Load policies policies from pretrained models specified by path in config."""
        for pid in self.policy_ids:
            path = str(self.model_dir) + str(pid)
            print("load the pretrained model from {}".format(path))
            policy_critic_state_dict = torch.load(path + '/critic.pt')
            policy_actor_state_dict = torch.load(path + '/actor.pt')

            self.policies[pid].critic.load_state_dict(policy_critic_state_dict)
            self.policies[pid].actor.load_state_dict(policy_actor_state_dict)

    def log(self):
        """Log relevent training and rollout colleciton information."""
        raise NotImplementedError

    def log_clear(self):
        """Clear logging variables so they do not contain stale information."""
        raise NotImplementedError

    def log_env(self):
        """Log information related to the environment."""
        raise NotImplementedError
        
    def log_train(self):
        """Log information related to training."""
        raise NotImplementedError
        
    def collect_rollout(self):
        """Collect a rollout and store it in the buffer."""
        raise NotImplementedError