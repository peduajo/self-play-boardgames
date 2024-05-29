import torch_tensorrt
import argparse

from utils.selfplay import SelfPlayEnv, SelfPlayEnvExploiter
from utils.callbacks import SelfPlayCallback, StartMainRolloutCallback

from stable_baselines3.common.env_checker import check_env
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.vec_env import DummyVecEnv, SubprocVecEnv
from stable_baselines3 import PPO
from environments.connect4.model import CustomActorCriticPolicy, CustomCNN, DistillNet, DistillNetTrain
import torch
from typing import Callable
import os
import torch.optim as optim

from stable_baselines3.common.utils import set_random_seed

import gymnasium as gym

import config

def make_env(env_id: str, rank: int, seed: int = 0, distill = False, exploiter = False) -> Callable:
    """
    Utility function for multiprocessed env.

    :param env_id: (str) the environment ID
    :param num_env: (int) the number of environment you wish to have in subprocesses
    :param seed: (int) the inital seed for RNG
    :param rank: (int) index of the subprocess
    :return: (Callable)
    """

    def _init() -> gym.Env:
        if exploiter:
            env = SelfPlayEnvExploiter(device, False, rank, distill)
        else:
            env = SelfPlayEnv(device, False, rank, distill)

        env.reset(seed=seed + rank)
        return env

    set_random_seed(seed)
    return _init

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Process some integers.")
    parser.add_argument("--exploiter", action="store_true", default=False,
                        help="Activate exploiter mode")
    parser.add_argument("--num-cpu", type=int, default=16, 
                        help="Number of CPUs to use (default: 16)")
    parser.add_argument("--distill", action="store_true", default=False,
                        help="Enable distillation")
    parser.add_argument("--experiment-name", type=str, 
                        help="Name for tensorboard experiment")


    args = parser.parse_args()
    device = torch.device("cuda")

    exploiter = args.exploiter
    num_cpu = args.num_cpu
    distill = args.distill

    print(f"EXPLOITER: {exploiter}")

    if distill:

        dist_ref = DistillNet(6, 128, 6, 7).to(device)
        dist_ref.eval()

        compile_spec = {"inputs":[torch_tensorrt.Input(min_shape=[1, 5, 6, 7],
                                                    opt_shape=[num_cpu, 5, 6, 7],
                                                    max_shape=[1024, 5, 6, 7])],
                        "enabled_precisions": torch.half,
                        "workspace_size" : 1 << 22
                        }
        
        dist_ref_serv = torch_tensorrt.compile(dist_ref, **compile_spec)


        dist_trn = DistillNetTrain().to(device)
        distill_opt = optim.Adam(dist_trn.parameters(), lr=1e-5)

        policy_kwargs = dict(
            features_extractor_class=CustomCNN,
            features_extractor_kwargs=dict(features_dim=128),
            distill = distill,
            optimizer_dist = distill_opt,
            dist_ref_model = dist_ref_serv,
            dist_trn_model = dist_trn
        )

    else:
        policy_kwargs = dict(
            features_extractor_class=CustomCNN,
            features_extractor_kwargs=dict(features_dim=128)
        )


    #vec_env = make_vec_env(env_id=SelfPlayEnv, n_envs=1, env_kwargs={'opponent_type':'mostly_best',
    #                                                                 'device':device,
    #                                                                 'determinist': False})
    vec_env = SubprocVecEnv([make_env(None, i, distill=distill, exploiter=exploiter) for i in range(num_cpu)])
    #vec_env = SelfPlayEnv(device, False, 0)

    #test_env = make_vec_env(env_id=SelfPlayEnv, n_envs=4, env_kwargs={'opponent_type':'mostly_best',
    #                                                                  'device':device,
    #                                                                  'determinist': True})

    if exploiter:
        test_env = SelfPlayEnvExploiter(device,
                            True,
                            0,
                            distill,
                            mode='eval')
        eval_games = config.GAMES_PER_OPPONENT_EVAL
    else:
        test_env = SelfPlayEnv(device,
                            True,
                            0,
                            distill,
                            mode='eval')
        eval_games = config.EVAL_GAMES

    callback_args = {
        'eval_env': test_env,
        'eval_freq' : 2050 * 3,
        'n_eval_episodes' : eval_games,
        'deterministic' : False,
        'render' : False,
        'verbose' : 0
    }
    threshold = config.THRESHOLD_EXP if exploiter else config.THRESHOLD
    eval_callback = SelfPlayCallback(threshold, exploiter ,**callback_args)
    path_best_model = os.path.join(config.TMPMODELDIR, "best_model.zip")

    batch_size = 512 if exploiter else 1024

    if os.path.exists(path_best_model):
        print("Continue training from checkpoint...")
        model = PPO.load(path_best_model, env=vec_env)
    else:
        model = PPO(CustomActorCriticPolicy,
                    env=vec_env,
                    policy_kwargs=policy_kwargs,
                    tensorboard_log="./runs/",
                    target_kl = 1.0,
                    batch_size=batch_size,
                    ent_coef=0.01,
                    gamma=0.99)

    callbacks_list = [eval_callback]
    if not exploiter:
        start_rollout_callback = StartMainRolloutCallback()
        callbacks_list.append(start_rollout_callback)

    model.learn(total_timesteps=1e9, tb_log_name=f"./{args.experiment_name}/", callback=callbacks_list)