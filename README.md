# self-play-boardgames

Framework for solving board games with reinforcement learning in self-play mode. In this case PPO is used, which is confronted with previous versions of itself. The idea is that this framework can be applied to any board game, where the architecture of the neural network and the representation of the environment have to be changed. Main ideas taken from https://github.com/davidADSP/SIMPLE.

A new version is saved when it has beaten all previous versions a high number of times (determined by a threshold). In addition, previous versions are implemented with TensorRT to make the environment run faster. Ideas from AlphaStar (https://deepmind.google/discover/blog/alphastar-mastering-the-real-time-strategy-game-starcraft-ii/) are taken as intelligent matchmaking to face other agents of the same level to facilitate the convergence of training. The idea of including the group of Exploiters agents is also taken, in this way they face the current main model.

The StableBaselines3 library is used, which has already implemented the PPO model for pytorch, allowing the introduction of Resnet as a customizable architecture. The https://github.com/DLR-RM/stable-baselines3/blob/master/stable_baselines3/common/policies.py file of this library is slightly modified to support the export of the model to TensorRT. If you want to run the repository, you have to change the policies.py file in the root directory to that of the library.

## Installation

Follow these steps to set up your environment and run the project:

### Setting Up the Python Environment

1. Clone the repository:
   ```bash
   https://github.com/peduajo/self-play-boardgames.git
   cd self-play-boardgames
   ```

2. Create the Python environment using the environment.yml file:
   ```bash
   conda env create -f environment.yml
   ```
## Run

### Python side

Run main agent training:

```bash
python train.py 
```

Run exploiter agent training (use other terminal while main is training):

```bash
python train.py --exploiter
```

## Conclusion

In conclusion, I will say that I see several disadvantages of this approach for any board game:
- **Machines**: being a rather slow approach, you need many machines or gpu cores to run in parallel. PPO being an on-policy approach, it is quite data inefficient and needs this.
- **Board game programming**: It is a task that can be simple for games like Connect4, but it gets quite complicated for more complex games like board games where there are cards that have passive effects. Also, if the game is new, finding bugs in it can be challenging.

I think it is more viable to approach each board game independently, either with tree search, as in AlphaZero, or if it includes randomness using some variant like Smooth UCT in Poker: https://www.davidsilver.uk/wp-content/uploads/2020/03/smooth_uct.pdf.