import numpy as np
import random
import time
import pdb
import json
import pdb 

from utils.files import load_model, load_all_models, get_best_model_name, load_best_model
from utils.agents import Agent

from environments.connect4.connect4 import Connect4Env

import os 
from stable_baselines3 import PPO
from environments.connect4.model import CustomActorCriticPolicy, CustomCNN

import config


class SelfPlayEnvExploiter(Connect4Env):
    # wrapper over the normal single player env, but loads the best self play model
    def __init__(self, device, determinist, rank, distill, mode='train'):
        super(SelfPlayEnvExploiter, self).__init__()

        self.distill = distill
        self.device = device

        self.best_model, self.best_model_name = load_best_model(self)
        print(f"BEST_MODEL_NAME: {self.best_model_name}")
        self.opponent_agent = Agent('ppo_opponent', self.device, self.best_model.to(self.device), distill=self.distill) 

        self.device = device
        self.determinist = determinist
        self.mode = mode 

        self.idx_game = 0

        self.count_wins = 0
        self.count_wins_x = 0
        self.count_wins_o = 0
        self.count_draws = 0
        self.count_loses = 0

    def setup_opponents(self):
        # incremental load of new model
        best_model_name = get_best_model_name()
        if self.best_model_name != best_model_name:
            print(f"Loading new best model: {best_model_name}!")
            self.best_model = load_model(self, best_model_name)
            self.best_model_name = best_model_name

            self.opponent_agent = Agent('ppo_opponent', self.device, self.best_model.to(self.device), distill=self.distill) 


        if self.mode == 'train': 
            self.agent_player_num = np.random.choice(self.n_players)

        elif self.mode == 'eval':
            self.agent_player_num = self.idx_game % 2 
            self.idx_game += 1

        self.agents = [self.opponent_agent] * self.n_players
        self.agents[self.agent_player_num] = None


    def reset(self, seed = 0):
        super(SelfPlayEnvExploiter, self).reset()
        self.setup_opponents()

        if self.current_player_num != self.agent_player_num:   
            self.continue_game()

        return self.observation, {}

    @property
    def current_agent(self):
        return self.agents[self.current_player_num]

    def continue_game(self):
        observation = None
        reward = None
        done = None

        while self.current_player_num != self.agent_player_num:

            action = self.current_agent.choose_action(self, choose_best_action = self.determinist, mask_invalid_actions = True)
            observation, reward, done, _ = super(SelfPlayEnvExploiter, self).step(action)
            if done:
                break

        return observation, reward, done, None

    def print_results(self):
        pct_wins = self.count_wins / config.GAMES_PER_OPPONENT_EVAL
        pct_loses = self.count_loses / config.GAMES_PER_OPPONENT_EVAL
        pct_draws = self.count_draws / config.GAMES_PER_OPPONENT_EVAL

        pct_wins_o = self.count_wins_o / config.GAMES_PER_OPPONENT_EVAL
        pct_wins_x = self.count_wins_x / config.GAMES_PER_OPPONENT_EVAL 

        print(f"% wins: {pct_wins}, % loses: {pct_loses}, % draws: {pct_draws}, % wins X: {pct_wins_x}, % wins O: {pct_wins_o}")

        self.count_loses = 0
        self.count_draws = 0
        self.count_wins = 0

        self.count_wins_o = 0
        self.count_wins_x = 0


    def step(self, action):
        observation, reward, done, _ = super(SelfPlayEnvExploiter, self).step(action)

        if not done:
            package = self.continue_game()
            if package[0] is not None:
                observation, reward, done, _ = package


        agent_reward = reward[self.agent_player_num]
        #logger.debug(f'\nReward To Agent: {agent_reward}')

        if done:
            if self.mode == 'eval':

                self.render()
        
                if agent_reward == -1:
                    self.count_loses += 1
                elif agent_reward == 0:
                    self.count_draws += 1 
                elif agent_reward == 1:
                    self.count_wins += 1
                    if self.agent_player_num == 0:
                        self.count_wins_x += 1
                    else:
                        self.count_wins_o += 1


        return observation, agent_reward, done, False, {} 


class SelfPlayEnv(Connect4Env):
    def __init__(self, device, determinist, rank, distill, mode='train'):
        super(SelfPlayEnv, self).__init__()

        model_filenames = os.listdir(config.MODELDIR)
        self.distill = distill
        self.rank = rank

        if 'base.ts' not in model_filenames:
            if rank > 0:
                time.sleep(15)
            else:
                self.save_base_model()

        self.opponent_models, self.opponent_names = load_all_models(self)
        self.best_model_name_main = get_best_model_name()
        self.best_model_name_exp = get_best_model_name(exploiter=True)
        self.device = device
        self.determinist = determinist
        self.mode = mode 

        self.idx_game = 0
        self.idx_opponent = 1

        self.count_wins = 0
        self.count_wins_x = 0
        self.count_wins_o = 0
        self.count_draws = 0
        self.count_loses = 0

        self.count_wins_vs_best = 0
        self.count_matches_vs_best = 0

        self.dict_wins = {name:0 for name in self.opponent_names}
        self.dict_games = {name:0 for name in self.opponent_names}

        self.dict_winrates = {}

    def save_base_model(self):
        name = "base.ts"
        policy_kwargs = dict(
            features_extractor_class=CustomCNN,
            features_extractor_kwargs=dict(features_dim=128)
        )

        ppo_model = PPO(CustomActorCriticPolicy, env=self, policy_kwargs=policy_kwargs)
        print(f'Saving {name} PPO model...')
        path = os.path.join(config.MODELDIR, name)
        input_shape = (1, 5, 6, 7)
        ppo_model.policy.export_tensorrt(input_shape, path)

    
    def check_exploiter_model(self):
        new_model_name_exp = None 
        best_model_name_exp = get_best_model_name(exploiter=True)
        if self.best_model_name_exp != best_model_name_exp:
            print(f"Loading new best model exploiter: {best_model_name_exp}!")
            self.opponent_models.append(load_model(self, best_model_name_exp, exploiter=True))
            self.opponent_names.append(best_model_name_exp)
            self.best_model_name_exp = best_model_name_exp
            self.dict_wins[best_model_name_exp] = 0
            self.dict_games[best_model_name_exp] = 0
            new_model_name_exp = best_model_name_exp

        if new_model_name_exp is not None and self.rank == 0:
            if os.path.exists(config.WINRATES_DICT_PATH):
                with open(config.WINRATES_DICT_PATH, 'r') as file:
                    dict_winrates = json.load(file)
                
                if new_model_name_exp not in dict_winrates.keys():
                    dict_winrates[new_model_name_exp] = 0.05

                    with open(config.WINRATES_DICT_PATH, 'w') as file:
                        json.dump(dict_winrates, file, indent=4)


    def setup_opponents(self):
        # incremental load of new model
        best_model_name_main = get_best_model_name()

        if self.best_model_name_main != best_model_name_main:
            print(f"Loading new best model main: {best_model_name_main}!")
            self.opponent_models.append(load_model(self, best_model_name_main))
            self.opponent_names.append(best_model_name_main)
            self.best_model_name_main = best_model_name_main
            self.dict_wins[best_model_name_main] = 0
            self.dict_games[best_model_name_main] = 0

        if self.mode == "eval":
            self.check_exploiter_model()

        #la idea es que tengamos una ventana de oponentes cercanos lo suficientemente buenos
        if len(self.opponent_models) > config.WINDOW_MODELS:
            self.opponent_models.pop(0)
            self.opponent_names.pop(0)

        if self.mode == 'train':
            self.load_winrates_data()
            #mixed matchmaking distribution. 20% of times picking top models and 80% with same ELO/rating
            r = random.uniform(0,1)
            if r < 0.2:
                matchmaking_probs = self.matchmaking_probs2
            else:
                matchmaking_probs = self.matchmaking_probs1

            self.opponent_model = np.random.choice(self.opponent_models, p=matchmaking_probs)

            self.opponent_agent = Agent('ppo_opponent', self.device, self.opponent_model.to(self.device), distill=self.distill)   
            self.agent_player_num = np.random.choice(self.n_players)

        elif self.mode == 'eval':
            self.agent_player_num = self.idx_game % 2 
            self.agent_player_str = 'X' if self.agent_player_num == 0 else 'O'

            self.game_vs_best = True if self.idx_opponent == 1 else False

            self.opponent_model = self.opponent_models[-self.idx_opponent]
            self.opponent_agent = Agent('ppo_opponent', self.device, self.opponent_model.to(self.device), distill=self.distill)   
            self.opponent_name = self.opponent_names[-self.idx_opponent]
            print(f"Playing against opponent: {self.opponent_name} with colour: {self.agent_player_str}")
            self.idx_game += 1

            print(f"Current evaluation game: {self.idx_game}")
            if self.idx_game % config.GAMES_PER_OPPONENT_EVAL == 0:
                self.idx_opponent += 1
                if self.idx_opponent > len(self.opponent_models):
                    self.idx_opponent = 1 

        self.agents = [self.opponent_agent] * self.n_players
        self.agents[self.agent_player_num] = None

    
    def load_winrates_data(self):
        if os.path.exists(config.WINRATES_DICT_PATH):
            with open(config.WINRATES_DICT_PATH, 'r') as file:
                dict_winrates = json.load(file)

        else:
            dict_winrates = {name:0.5 for name in self.opponent_names}


        matchmaking_probs_1, matchmaking_probs_2 = [], []
        for name in self.opponent_names:
            winrate = dict_winrates[name]
            prob1 = winrate*(1-winrate)
            matchmaking_probs_1.append(prob1)
            prob2 = (1-winrate)**3
            matchmaking_probs_2.append(prob2)

        matchmaking_probs1 = np.array(matchmaking_probs_1)
        self.matchmaking_probs1 = matchmaking_probs1 / np.sum(matchmaking_probs1)

        matchmaking_probs2 = np.array(matchmaking_probs_2)
        self.matchmaking_probs2 = matchmaking_probs2 / np.sum(matchmaking_probs2)

        assert len(self.matchmaking_probs1) == len(self.opponent_names)


    def reset(self, seed = 0):
        super(SelfPlayEnv, self).reset()
        self.setup_opponents()

        if self.current_player_num != self.agent_player_num:   
            self.continue_game()

        return self.observation, {}

    @property
    def current_agent(self):
        return self.agents[self.current_player_num]

    def continue_game(self):
        observation = None
        reward = None
        done = None

        while self.current_player_num != self.agent_player_num:

            action = self.current_agent.choose_action(self, choose_best_action = self.determinist, mask_invalid_actions = True)
            observation, reward, done, _ = super(SelfPlayEnv, self).step(action)
            if done:
                break

        return observation, reward, done, None

    def print_results(self):
        pct_wins = self.count_wins / config.EVAL_GAMES
        pct_loses = self.count_loses / config.EVAL_GAMES
        pct_draws = self.count_draws / config.EVAL_GAMES

        pct_wins_o = self.count_wins_o / config.EVAL_GAMES
        pct_wins_x = self.count_wins_x / config.EVAL_GAMES 

        self.pct_wins_vs_best = self.count_wins_vs_best / float(self.count_matches_vs_best)

        print(f"% wins: {pct_wins}, % loses: {pct_loses}, % draws: {pct_draws}, % wins X: {pct_wins_x}, % wins O: {pct_wins_o}, % wins vs best: {self.pct_wins_vs_best}")

        self.count_loses = 0
        self.count_draws = 0
        self.count_wins = 0

        self.count_wins_o = 0
        self.count_wins_x = 0

        self.idx_game = 0
        self.idx_opponent = 1

        self.count_wins_vs_best = 0
        self.count_matches_vs_best = 0

        self.dict_winrates = {k: float(v)/(self.dict_games[k] + 1e-5) for k,v in self.dict_wins.items()}
        print(self.dict_winrates)

        self.dict_wins = {name:0 for name in self.opponent_names}
        self.dict_games = {name:0 for name in self.opponent_names}

    def step(self, action):
        observation, reward, done, _ = super(SelfPlayEnv, self).step(action)

        if not done:
            package = self.continue_game()
            if package[0] is not None:
                observation, reward, done, _ = package


        agent_reward = reward[self.agent_player_num]

        if done:
            if self.mode == 'eval':
                self.dict_games[self.opponent_name] += 1 

                self.render()

                if self.game_vs_best:
                    self.count_matches_vs_best += 1
        
                if agent_reward == -1:
                    print(f"Loss against opponent: {self.opponent_name} with colour: {self.agent_player_str}")
                    self.count_loses += 1
                elif agent_reward == 0:
                    self.count_draws += 1 
                elif agent_reward == 1:
                    self.count_wins += 1
                    self.dict_wins[self.opponent_name] += 1
                    if self.agent_player_num == 0:
                        self.count_wins_x += 1
                    else:
                        self.count_wins_o += 1

                    if self.game_vs_best:
                        self.count_wins_vs_best += 1
            
            self.opponent_model.to('cpu')

        return observation, agent_reward, done, False, {} 