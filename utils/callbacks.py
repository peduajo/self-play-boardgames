from utils.files import get_best_model_name, get_model_stats

import os 
import config
from stable_baselines3.common.callbacks import EvalCallback, BaseCallback
import numpy as np
import pdb 

import json 
import time



class StartMainRolloutCallback(BaseCallback):
    def __init__(self, *args, **kwargs):
        super(StartMainRolloutCallback, self).__init__(*args, **kwargs)
    
    #esto se hace para añadir exploiters solo al principio 
    def _on_rollout_start(self) -> None:
        self.training_env.env_method('check_exploiter_model')

    def _on_step(self):
        return True  # Continuar con el entrenamiento


class SelfPlayCallback(EvalCallback):
    def __init__(self, threshold, exploiter, *args, **kwargs):
        super(SelfPlayCallback, self).__init__(*args, **kwargs)
        self.generation, self.base_timesteps, self.best_reward = get_model_stats(get_best_model_name())

        #reset best_mean_reward because this is what we use to extract the rewards from the latest evaluation by each agent
        self.best_mean_reward = -np.inf
        self.threshold = threshold # the threshold is a constant
        self.exploiter = exploiter

    def _on_step(self) -> bool:

        if self.eval_freq > 0 and self.n_calls % self.eval_freq == 0:
            result = super(SelfPlayCallback, self)._on_step()

            self.eval_env.env_method('print_results')

            av_reward = np.mean(self.best_mean_reward)
            std_reward = np.std(self.best_mean_reward)
            av_timesteps = np.mean(self.num_timesteps)
            total_episodes = np.sum(self.n_eval_episodes)

            if not self.exploiter:
                dict_winrates = self.eval_env.get_attr('dict_winrates')[0]

            print(f"Eval num_timesteps={self.num_timesteps}, episode_reward={av_reward:.2f} +/- {std_reward:.2f}")
            print(f"Total episodes ran={total_episodes}")
            print(f"Average timesteps={av_timesteps}")

            if result and av_reward > self.threshold:
                self.generation += 1

                #el modelo del exploiter será como el próximo bueno
                av_rewards_str = str(round(av_reward,3))

                if self.exploiter:
                    generation_str = str(len(os.listdir(config.EXPMODELDIR))).zfill(5)
                    new_best_model_filename = f"_model_{generation_str}_{av_rewards_str}_{str(self.base_timesteps + self.num_timesteps)}_exploiter.ts"
                    path = os.path.join(config.EXPMODELDIR, new_best_model_filename)
                else:
                    generation_str = str(len(os.listdir(config.MODELDIR))).zfill(5)
                    new_best_model_filename = f"_model_{generation_str}_{av_rewards_str}_{str(self.base_timesteps + self.num_timesteps)}_.ts"
                    path = os.path.join(config.MODELDIR, new_best_model_filename)

                input_shape = (1, 5, 6, 7)
                self.model.policy.export_tensorrt(input_shape,path)

                self.best_mean_reward = -np.inf

                #se introduce un winrate ficticio bajo para que enfrantarse al nuevo modelo sea paulatino
                if not self.exploiter:
                    dict_winrates[new_best_model_filename] = 0.05

                else:
                    #solo continua el entrenamiento del exploiter cuando se obtenga otro main nuevo
                    continue_training = False 
                    n_main_models_start = len([f for f in os.listdir(os.path.join(config.MODELDIR)) if f.startswith("_model")])

                    while not continue_training:
                        n_main_models_now = len([f for f in os.listdir(os.path.join(config.MODELDIR)) if f.startswith("_model")])

                        if n_main_models_now > n_main_models_start:
                            continue_training = True 
                        
                        else: 
                            time.sleep(1)

            if not self.exploiter:
                #save winrate 
                with open(config.WINRATES_DICT_PATH, 'w') as file:
                    json.dump(dict_winrates, file, indent=4)

        return True 
