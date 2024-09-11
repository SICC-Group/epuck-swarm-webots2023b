import argparse

parser = argparse.ArgumentParser()

# aegnt info
parser.add_argument("--id", type=int, default=0)
parser.add_argument("--num_agents", type=int, default=6,
                  help="number of robots in the swarm")
parser.add_argument("--byz_num", type=int, default=0,
                  help="number of byzantine robots in the swarm")
parser.add_argument("--byz_style", type=int, default=2,
                  help="Specify the behavior value. "
                  "If the value is 0, Byzantine behavior will continuously output 0. "
                  "If the value is 1, Byzantine behavior will continuously output 1. "
                  "If the value is anything else, Byzantine behavior will output a random value.")
parser.add_argument("--range0", type=float, default=0.25,
                  help="communication range for common robots")
parser.add_argument("--range1", type=float, default=1,
                  help="communication range for ranger robots")
parser.add_argument("--ranger_robots", type=int, default=0,
                  help="number of ranger robots")
parser.add_argument("--group_number", type=int, default=1,
                  help="number of groups in the swarm")

# webots simulation env info
parser.add_argument("--black_ratio", type=float, default=0.4,
                  help="the ratio of black tiles in the environment")
parser.add_argument("--time_step", type=int, default=100,
                  help="simulation time step in webots")
parser.add_argument("--frequency_ratio", type=int, default=1, 
                    help="the ratio of frequency differences that "
                    "may exist among different sensors")
parser.add_argument("--dist_threshold", type=int, default=80,
                    help="distance threshold for obstacle avoidance")
parser.add_argument("--variance_threshold", type=int, default=0.001)

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
# parser.add_argument('--experiment_name', type=str, default="exp")
###############################################################################
parser.add_argument('--max_steps', type=int, default=1000000,  #200000
                    help="Number of env steps to train for")
parser.add_argument('--episode_length', type=int, default=1000, #300
                    help="Max length for any episode")

# parser.add_argument('--t_max', type=int, default=2000000)
parser.add_argument('--buffer_length', type=int, default=800)
###############################################################################
parser.add_argument('--model_dir', type=str, default=None)

# RL info
parser.add_argument("--collision_distance", type=float, default=0.04)
parser.add_argument("--collision_reward", type=float, default=-0.5)
parser.add_argument("--reward_exploration", type=float, default=1.0)
parser.add_argument('--reward_time', type=float, default=-0.1,
                    help="reward (or punishment) given at each time step")
parser.add_argument('--lr', type=float, default=1e-3)
parser.add_argument('--gamma', type=float, default=0.99)
parser.add_argument('--batch_size', type=int, default=32)

# train
# parser.add_argument('--train_episode_interval', type=int, default=1,
#                     help="Number of env steps between updates to actor/critic")
# parser.add_argument('--train_time_interval', type=int, default=1)
# parser.add_argument('--episode_t_max', type=float, default=1200.0)

# eval
parser.add_argument('--use_eval', action='store_false',
                    default=True, help="Whether to conduct the evaluation")
parser.add_argument('--eval_interval', type=int,  default=20000,  # 2000
                    help="After how many episodes the model should be evaled")
parser.add_argument('--num_eval_episodes', type=int, default=1, #  5
                    help="How many episodes to collect for each eval")
parser.add_argument('--eval_sync', default=True, action='store_false')
# parser.add_argument('--eval_interval', type=int, default=3000,
#                     help="After how many seconds the model should be evaled")

# save parameters 
parser.add_argument('--use_save', action='store_false',
                    default=True, help="Whether to save the model")
parser.add_argument('--save_interval', type=int, default=2,)  # 2000
# parser.add_argument('--save_interval', type=int, default=3000,
#                     help="After how many seconds of training the policy model should be saved")

# log parameters
parser.add_argument('--log_interval', type=int, default=4000)  # 2000
# parser.add_argument('--log_interval', type=int, default=3000,
#                     help="After how many seconds of training the policy model should be saved")
