import sys
import os
import socket
import setproctitle
sys.path.append("/usr/local/webots/lib/controller/python")
sys.path.append("/usr/local/webots/lib/controller/python/controller")
sys.path.append('../')
# sys.path.append('../..')

import numpy as np
import pandas as pd
from pathlib import Path
import wandb
import torch

current_dir = os.path.dirname(os.path.abspath(__file__))
exp_path = os.path.join(current_dir, "..", "..")
sys.path.append(exp_path)
save_path = os.path.join(current_dir, "..", "..", "results")
from config import parser
from black_white_ratio_estimate_supervisor_rl import Epuck2Supervisor
from my_runner import MyRunner as Runner


def main(args_):
    args, unknown = parser.parse_known_args(args_)
    
    # cuda and # threads
    if args.cuda and torch.cuda.is_available():
        device = torch.device("cuda:0")
        torch.set_num_threads(args.n_training_threads)
        if args.cuda_deterministic:
            torch.backends.cudnn.benchmark = False
            torch.backends.cudnn.deterministic = True
    else:
        device = torch.device("cpu")
        torch.set_num_threads(args.n_training_threads)

    # setup file to output tensorboard, hyperparameters, and saved models
    run_dir = os.path.join(
        os.path.split(os.path.dirname(os.path.abspath(__file__)))[0], "..",
        "results", args.env_name, args.algorithm_name
    )
    run_dir = Path(run_dir)
    if not run_dir.exists():
        os.makedirs(str(run_dir))

    if args.use_wandb:
        # init wandb
        run = wandb.init(config=args,
                         project=args.env_name,
                         entity=args.user_name,
                         notes=socket.gethostname(),
                         name=str(args.algorithm_name) + "_" +
                         str(args.experiment_name) +
                         "_seed" + str(args.seed),
                         dir=str(run_dir),
                         job_type="training",
                         reinit=False)
    else:
        exist_run_nums = [int(str(folder.name).split('run')[1]) 
                          for folder in run_dir.iterdir() 
                          if str(folder.name).startswith('run')]
        if len(exist_run_nums) == 0:
            curr_run = 'run1'
        else:
            curr_run = 'run%i' % (max(exist_run_nums) + 1)
        
        run_dir = run_dir / curr_run
        if not run_dir.exists():
            os.makedirs(str(run_dir))

    # setproctitle.setproctitle(str(args.algorithm_name) + "-" + str(
    #     args.env_name) + "-" + str(args.experiment_name) + "@" + str(args.user_name))

    # set seeds
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)
    np.random.seed(args.seed)

    env = Epuck2Supervisor(args, save_path, col=args.col, row=args.row)
    num_agents = args.num_agents
               
    config = {"args": args,
              "env": env,
              "num_agents": num_agents,
              "device": device,
              "run_dir": run_dir,}

    total_train_steps = 0
    runner = Runner(config=config)
    
    progress_filename = os.path.join(run_dir,'config.csv')
    df = pd.DataFrame(list(args.__dict__.items()),columns=['Name', 'Value'])
    df.to_csv(progress_filename,index=False)
    
    columns = ['step']
    for i in range(num_agents):
        columns.append(f'episode_reward_{i}')
    columns.append('episode_time')
    columns.append(f'ratio_estimation')
    columns.append(f'exploration_rate')
    for i in range(4):
        columns.append(f'action_{i}')
    
    progress_filename = os.path.join(run_dir,'progress.csv')
    df = pd.DataFrame(columns=columns)
    df.to_csv(progress_filename,index=False)
    
    progress_filename = os.path.join(run_dir,'progress_eval.csv')
    df = pd.DataFrame(columns=columns)
    df.to_csv(progress_filename,index=False)
    
    # progress_filename_train = os.path.join(run_dir,'progress_train.csv')
    # df = pd.DataFrame(columns=['step','loss','Q_tot','grad_norm']) 
    # df.to_csv(progress_filename_train,index=False)
    
    # progress_filename_train = os.path.join(run_dir,'progress_train_adj.csv')
    # df = pd.DataFrame(columns=['step','advantage','clamp_ratio','rl_loss','entropy_loss','grad_norm']) 
    # df.to_csv(progress_filename_train,index=False)

    try:
        while total_train_steps < args.max_steps:
            total_train_steps = runner.run()
    except KeyboardInterrupt:
        print("=== {0:>8} is aborted by keyboard interrupt".format('Main'))
    env.close()

    if args.use_wandb:
        run.finish()
    else:
        # runner.writter.export_scalars_to_json(str(runner.log_dir + '/summary.json'))
        # runner.writter.close()
        pass


if __name__ == "__main__":
    main(sys.argv[1:])
    # tmux kill-session -t workers
