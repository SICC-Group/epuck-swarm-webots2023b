import argparse

parser = argparse.ArgumentParser()

# aegnt info
parser.add_argument("--id", type=int, default=0)
parser.add_argument("--num_agents", type=int, default=6,
                  help="number of robots in the swarm")
parser.add_argument("--byz_num", type=int, default=0,
                  help="number of byzantine robots in the swarm")
parser.add_argument("--byz_style", type=str, default="", 
                    choices=["", "ratio-1", "ratio-0.6", "ratio-0.3", "ratio-0", "ratio-r",
                             "action-0", "action-1", "action-2", "action-3",
                             "grad-noise", "grad-signflip", "grad-alie",
                             "grad-ipm-3", "grad-ipm-5", "grad-ipm-7"])
parser.add_argument("--range0", type=float, default=0.25,
                  help="communication range for common robots")
parser.add_argument("--range1", type=float, default=1,
                  help="communication range for ranger robots")
parser.add_argument("--ranger_robots", type=int, default=0,
                  help="number of ranger robots")
parser.add_argument("--group_number", type=int, default=1,
                  help="number of groups in the swarm")

# webots simulation env info
parser.add_argument('--exclude_steps', type=int, default=5,
                    help="the gs values are not stable in the first some steps")
parser.add_argument("--black_ratio", type=float, default=0.4,
                  help="the ratio of black tiles in the environment")
parser.add_argument("--time_step", type=int, default=640,
                  help="simulation time step in webots")
parser.add_argument("--frequency_ratio", type=int, default=20, 
                    help="the ratio of frequency differences that "
                    "may exist among different sensors")
parser.add_argument("--ps_threshold", type=int, default=80,
                    help="ps threshold for obstacle avoidance")
parser.add_argument("--center_threshold", type=int, default=0.047)
parser.add_argument('--col', type=int, default=20)
parser.add_argument('--row', type=int, default=20)

# exp-training info
parser.add_argument('--seed', type=int, default=1)
parser.add_argument('--cuda', action='store_false', default=False)
parser.add_argument('--cuda_deterministic',
                    action='store_false', default=True)
parser.add_argument('--n_training_threads', type=int,
                    default=1, help="Number of torch threads for training")
parser.add_argument('--use_wandb', action='store_true', default=False)
parser.add_argument('--env_name', type=str, default="ratio_estimation")
parser.add_argument('--algorithm_name', type=str, default="A3C")
parser.add_argument('--max_episodes', type=int, default=80,
                    help="Number of env steps to train for")
parser.add_argument('--buffer_length', type=int, default=5000)
parser.add_argument('--update_ratio_steps', type=int, default=100)
parser.add_argument('--model_dir', type=str, default=None)
parser.add_argument('--ratio_update_method', type=str, default='all',
                    choices=['threshold', 'shapley', 'all'],
                    help="The method to update the global ratio in evaluation"
                    "options: threshold, shapley, all")

# FRL info
parser.add_argument("--collision_distance", type=float, default=0.1)
parser.add_argument('--done_exploration', type=float, default=0.5)
parser.add_argument('--min_exploration', type=float, default=0.3)
parser.add_argument('--done_ratio_difference', type=float, default=0.001)
parser.add_argument("--reward_collision", type=float, default=-1.0)
parser.add_argument("--reward_exploration", type=float, default=5.0,
                    help="reward for the exploration of each tile")
parser.add_argument('--reward_exploration_ratio', type=int, default=100)
parser.add_argument('--reward_time', type=float, default=-0.1,
                    help="reward (or punishment) given at each time step")
parser.add_argument('--reward_repeat', type=float, default=-1.0,)
parser.add_argument('--reward_local_ratio', type=float, default=3.0)
parser.add_argument('--reward_global_ratio', type=float, default=3.0)
parser.add_argument('--lr', type=float, default=1e-3)
parser.add_argument('--gamma', type=float, default=0.99)
parser.add_argument('--batch_size', type=int, default=32)
parser.add_argument('--reward_normalize', action='store_false', default=False)

parser.add_argument('--grad_norm_init', type=int, default=800)
parser.add_argument('--norm_decay_steps', type=int, default=12)
parser.add_argument('--grad_norm_min', type=float, default=100)
parser.add_argument('--grad_aggregation', type=str, default="mean", 
                    choices=["mean", "median", "multikrum", "clustering", "signguard"])
parser.add_argument('--aggregation_time', type=int, default=100)

# eval
parser.add_argument('--use_eval', action='store_true',
                    default=True, help="Whether to conduct the evaluation")
parser.add_argument('--eval_interval', type=int,  default=5,  # 2000
                    help="After how many episodes the model should be evaled")
parser.add_argument('--num_eval_episodes', type=int, default=10,
                    help="How many episodes to collect for each eval")
parser.add_argument('--eval_sync', default=True, action='store_false')

# save parameters 
parser.add_argument('--use_save', action='store_false',
                    default=True, help="Whether to save the model")
parser.add_argument('--save_interval', type=int, default=2000,)  # 2000

# log parameters
parser.add_argument('--log_interval', type=int, default=1)  # 2000
