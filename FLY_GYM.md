# Fly brain on gymnasium games

```
pip install numpy "gymnasium[classic-control,toy-text,box2d]" flappy-bird-gymnasium pygame juliacall

python fly_gym.py list                                   # tuned games
python fly_gym.py train cartpole --save cartpole.npz     # about 10 minutes
python fly_gym.py test --load cartpole.npz --episodes 3 --render
python fly_gym.py test --load players/cartpole.npz --episodes 3 --render   # already trained
python fly_gym.py random cartpole                        # baseline
python fly_gym.py train cartpole --no-brain              # control: same learner, no connectome
python fly_gym.py train MountainCarContinuous-v0         # any other gymnasium id
```

One run uses one CPU core and about 300-550 MB. `connections_princeton.csv` must be next to the scripts.

| File | Contents |
|---|---|
| `fly_games.py` | any gymnasium observation/action space <-> numbers the brain can use; presets per game |
| `fly_gym.py` | the two learners, controls, save/load, command line |
| `players/*.npz` | the trained players from the table below |
| `tests/test_fly_gym.py` | fast tests, no Julia: `python -m unittest discover tests` |

## How it learns

- **readout** (default): the connectome is frozen. A linear policy and value estimate on the spike counts of 500
  neurons are trained with PPO from the game's reward.
- **stdp** (`--learner stdp`): the synapses onto the chosen action's output population change by spike-timing
  eligibility x dopamine, where dopamine is the reward prediction error.

MountainCar is the only game with a training hint (potential-based energy shaping). Every score below is the game's
own reward.

## Results

Mean return over 100 frozen games on seeds never used in training (Blackjack: 2,000 games), +- standard error.
**One training seed per game**, so small differences between columns mean little. Higher is better everywhere.
"No brain" is the same learner on the raw inputs, with the same settings.

| Game | Random | Fly brain | No brain | gymnasium's "solved" |
|---|---|---|---|---|
| CartPole-v1 | 20.3 | 427 +- 10 | 444 +- 6 | 475 |
| Acrobot-v1 | -497 | -119 +- 3 | -79 +- 2 | -100 |
| MountainCar-v0 | -200 | -145 +- 3 | -120 +- 1 | -110 |
| LunarLander-v3 | -191 | 99 +- 9 | 167 +- 12 | 200 |
| Pendulum-v1 | -1186 | -834 +- 15 | -947 +- 20 | none |
| FrozenLake-v1 | 0.00 | 0.48 +- 0.05 | 0.65 +- 0.05 | 0.70 |
| CliffWalking-v1 (200-step limit) | -2089 | -26 +- 2 | -15 +- 0 | none |
| Blackjack-v1 | -0.38 | -0.09 +- 0.02 | -0.05 +- 0.02 | none |
| FlappyBird-v0 (gap 150, 100-pipe cap) | 0.3 | 96 +- 4 | 121 +- 1 | none |

Read this as: the fly brain learns all of these games from reward and beats random play on every one, but a linear
readout of the raw inputs is at least as good on every game except Pendulum, where the connectome's nonlinearity may
help (one seed, not confirmed). Hyperparameters were chosen while looking at some of these numbers, so treat them as
slightly optimistic.

Not tuned, only checked to run, save and reload: MountainCarContinuous-v0, Taxi-v4, BipedalWalker-v3.

## Things we found

- **40 ms per decision instead of 20 ms.** Activity takes time to cross the connectome: at 20 ms the spikes mostly
  describe the previous input. A linear decoder reads FrozenLake's state with 70% accuracy at 20 ms and 99% at 40 ms.
- **STDP is weak.** It learns a two-cue bandit (mean reward 0.91, 0.95 and 0.99 of 1 on three seeds) but not CartPole:
  four settings gave 24.3, 21.9, 20.5 and 14.0 against 20.3 for random play.
- **Open problem:** applying dopamine to every excitatory synapse (what `FlyBrain.reward!` does) learned the reverse
  answer on that bandit with this trainer (-0.43, -0.96, -0.57), although `FlyBrain/examples/bandit.jl` reaches 97%.
  Changing exploration, the critic, 40 ms, baseline voting, output choice or dopamine
  delay alone did not fix it. The mode was removed.
- Image observations (CarRacing, Atari frames) are refused: the brain needs a short list of numbers.
