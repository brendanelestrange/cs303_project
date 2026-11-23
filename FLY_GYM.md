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
python fly_viewer.py                                     # watch every trained player in the browser
```

One run uses one CPU core and about 300-550 MB. `connections_princeton.csv` must be next to the scripts.

| File | Contents |
|---|---|
| `fly_games.py` | any gymnasium observation/action space <-> numbers the brain can use; presets per game |
| `fly_gym.py` | the two learners, controls, save/load, command line |
| `players/*.npz` | the trained players from the table below |
| `fly_viewer.py`, `viewer/index.html` | live viewer: the game next to every spike of the brain |
| `tests/` | fast tests, no Julia: `python -m unittest discover tests` |

## Which part of the brain

Training uses 11,795 neurons: the left smell circuit (`AL_L`, `LH_L`, `MB_CA_L`; 8,547 neurons) plus a premotor region,
both lateral accessory lobes (`LAL_L`, `LAL_R`). Change it with `--neuropils`. Saved players remember their own regions.

Adding the LAL does not make it the brain's motor output. Inputs, outputs and read-out neurons are still chosen by
connectivity (most outgoing synapses; most input from the input neurons), and in these five regions the smell circuit
barely reaches the LAL: 411 connections go from smell neurons to LAL neurons, against 47,211 inside the LAL. A linear
decoder reads CartPole's 4 inputs from the usual 500 read-out neurons with R² of about 0.75, and from all 3,366 LAL
neurons with R² of about 0. How many of the 500 read-out neurons are in the LAL:

| CliffWalking | FrozenLake | LunarLander | Acrobot | CartPole, Flappy, MountainCar, Pendulum, Blackjack |
|---|---|---|---|---|
| 206 | 77 | 77 | 8 | 0 |

Games with many inputs (CliffWalking has 24) use up the best-connected smell neurons as inputs, so LAL neurons take
over. To make the LAL a real output, the regions that connect the two would have to be simulated too.

## Watching it play

```
python fly_viewer.py                                    # every player in players/, pick one in the page
python fly_viewer.py --load players/lunarlander.npz      # one player
python fly_viewer.py --game cartpole --learn             # an untrained brain learning by STDP while you watch
```

The page shows the game, each neuron at its FlyWire position lighting up when it fires, a spike raster (inputs,
the neurons pushing each choice, and a sample of every brain region), what the fly senses, the probability of each
choice over the last 60 decisions, and how many inputs and read-out neurons each region has. Click a legend entry to
see one group of neurons alone. It plays exactly the games `fly_gym.py test` plays. Games that decide 50 times a
second run at half speed so each 40 ms decision stays visible. The older Flappy Bird brains (`.jls` from
`flappy_readout.py` and `flappy_fly.py`) still work with `--load`.

## How it learns

- **readout** (default): the connectome is frozen. A linear policy and value estimate on the spike counts of 500
  neurons are trained with PPO from the game's reward.
- **stdp** (`--learner stdp`): the synapses onto the chosen action's output population change by spike-timing
  eligibility x dopamine, where dopamine is the reward prediction error.

MountainCar is the only game with a training hint (potential-based energy shaping). Every score below is the game's
own reward.

## Results

`players/` holds the players trained with the premotor region (default settings, `python fly_gym.py train GAME`).
Both fly-brain columns were measured the same way, with `python fly_gym.py test --load PLAYER --episodes 400`: mean
return over 400 frozen games on seeds never used in training (Blackjack: 5,000 games), +- standard error. The
"smell circuit only" players are the earlier ones (commit c338516, `--neuropils AL_L,LH_L,MB_CA_L`). Random and
"No brain" (the same learner on the raw inputs, same settings) are from 100 games (Blackjack 2,000) and do not depend
on the brain. **One training seed per game.** Higher is better everywhere.

| Game | Random | Fly brain + LAL | Smell circuit only | No brain | gymnasium's "solved" |
|---|---|---|---|---|---|
| CartPole-v1 | 20.3 | 456 +- 3 | 433 +- 5 | 444 +- 6 | 475 |
| Acrobot-v1 | -497 | -117 +- 2 | -118 +- 1 | -79 +- 2 | -100 |
| MountainCar-v0 | -200 | -153 +- 1 | -143 +- 2 | -120 +- 1 | -110 |
| LunarLander-v3 | -191 | 141 +- 6 | 82 +- 5 | 167 +- 12 | 200 |
| Pendulum-v1 | -1186 | -787 +- 8 | -873 +- 8 | -947 +- 20 | none |
| FrozenLake-v1 | 0.00 | 0.52 +- 0.02 | 0.43 +- 0.02 | 0.65 +- 0.05 | 0.70 |
| CliffWalking-v1 (200-step limit) | -2089 | -16.3 +- 0.6 | -30.3 +- 1.6 | -15 +- 0 | none |
| Blackjack-v1 | -0.38 | -0.07 +- 0.01 | -0.08 +- 0.01 | -0.05 +- 0.02 | none |
| FlappyBird-v0 (gap 150) | 0.3 | 99 +- 2 | 88 +- 2 | 121 +- 1 | none |

Read this as: the fly brain learns all of these games from reward and beats random play on every one. The players
with the premotor region are better on six games, about the same on Acrobot and Blackjack, and worse on MountainCar.
That is not evidence that the LAL helps: CartPole, Flappy and Pendulum improved with no LAL neuron read out, and each
column is one training run, whose luck counts too (the end-of-training test of LunarLander gave 163 +- 10 on 100
games, a second test 129 +- 14). A linear readout of the raw inputs is still at least as good on every game except
Pendulum. Hyperparameters were chosen while looking at the earlier numbers, so treat them as slightly optimistic.

Flappy Bird's return is 0.1 per frame alive (1 on a frame that passes a pipe), and games without a time limit of their
own stop after 1,000 frames (`Game.limit`), so a perfect game returns 122.5 (25 pipes), not 100 pipes. The earlier
version of this table said "100-pipe cap". `fly_viewer.py --load something.jls` has no frame limit, like
`flappy_readout.py`.

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
