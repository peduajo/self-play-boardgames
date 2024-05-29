from utils.agents_test import Agent

import config
from utils.files import load_model, get_best_model_name
from environments.connect4.connect4 import Connect4Env
from stable_baselines3.common.evaluation import evaluate_policy
from stable_baselines3 import PPO
import torch
from utils.selfplay import SelfPlayEnv

if __name__ == '__main__':
    env = Connect4Env()

    best_model_name = get_best_model_name()
    best_model = load_model(env, best_model_name)
    #best_model = load_model(env, "_model_00087_0.91_32960000_.pkl")
    best_model.eval()

    human_model = load_model(env, "_model_00001_0.928_984000_.ts")
    human_model.eval()

    device = torch.device("cuda")

    opponent = Agent('ppo_opponent', device=device, model = best_model.to(device), distill=False)
    human = Agent('ppo_opponent', device=device, model = human_model.to(device), distill=False)

    _ = env.reset()
    done = False 

    while not done:
        env.render()

        if env.current_player_num == 0:
            action = opponent.choose_action(env, choose_best_action=True, mask_invalid_actions=True)
            print("turno best model")
        else:
            action = int(input('\nPlease choose an action: '))
            #action = human.choose_action(env, choose_best_action=True, mask_invalid_actions=True)

        obs, reward, done, _ = env.step(action)

    env.render()
    if reward == 0:
        print('Draw!')
    else:
        if env.current_player_num == 1:
            print('Win O!')
        else:
            print('Win X!')
            

    #m = PPO.load('logs/best_model')

    #test_env = SelfPlayEnv(device, True, 0, mode='eval')
    #mean_reward, std_reward = evaluate_policy(m, test_env, n_eval_episodes=200, deterministic=True)
    #print(f"mean_reward={mean_reward:.2f} +/- {std_reward}")