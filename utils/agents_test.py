import sys
import numpy as np
np.set_printoptions(threshold=sys.maxsize)
import random
import string

from stable_baselines3.common.policies import ActorCriticPolicy
import torch

def sample_action(action_probs):
    action = torch.distributions.Categorical(action_probs).sample()
    return action


def mask_actions(legal_actions, action_probs):
    masked_action_probs = legal_actions * action_probs
    masked_action_probs /=  masked_action_probs.sum()
    return masked_action_probs


class Agent():
  def __init__(self, name, device, distill=False, model = None):
      self.name = name
      self.id = self.name + '_' + ''.join(random.choice(string.ascii_lowercase) for x in range(5))
      self.model : ActorCriticPolicy = model
      self.points = 0
      self.device = device
      self.distill = distill

  def print_top_actions(self, action_probs):
    top5_action_idx = np.argsort(-action_probs)[:5]
    top5_actions = action_probs[top5_action_idx]
    print(f"Top 5 actions: {[str(i) + ': ' + str(round(a,2))[:5] for i,a in zip(top5_action_idx, top5_actions)]}")

  def choose_action(self, env, choose_best_action, mask_invalid_actions):
      with torch.no_grad():
        obs = torch.FloatTensor(env.observation).to(self.device).unsqueeze(0)
        if self.distill:
          policy_raw, _, _ = self.model(obs)
        else:
          policy_raw, _ = self.model(obs)
        action_probs = torch.softmax(policy_raw, axis=1)

      self.print_top_actions(action_probs.squeeze(0).cpu().numpy())
      
      if mask_invalid_actions:
        action_probs = mask_actions(env.legal_actions.to(self.device), action_probs)

      if choose_best_action:
         action = torch.argmax(action_probs).item()
      else:
         action = sample_action(action_probs)

      return action



