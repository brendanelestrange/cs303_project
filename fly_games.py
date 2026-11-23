"""Turn any gymnasium game into something the fly brain can play (no Julia in this file).

The brain takes numbers in [-1, 1] (one pair of input populations each: one for positive, one for negative)
and answers with one choice per action "head". A Game does both translations:

  observations  Box               -> scaled to [-1, 1] (by the preset, else measured in random play)
                Discrete(n)       -> n states on ceil(n/2) inputs: state 2k is +1 on input k, state 2k+1 is -1
                MultiBinary, MultiDiscrete, Tuple, Dict -> the same, concatenated
  actions       Discrete(n)       -> one head with n choices
                MultiDiscrete, MultiBinary -> one head per entry
                Box (continuous)  -> one head per dimension, each with --bins evenly spaced values

PRESETS holds settings for games we tuned; every other gymnasium id uses the defaults.
"""
from dataclasses import dataclass, field

import gymnasium
import numpy as np
from gymnasium import spaces


@dataclass
class Preset:
    env_id: str
    note: str = ""
    features: object = None        # obs -> values in about [-1, 1] (clipped). None = automatic
    n_obs: int = 0                 # how many values features() returns
    repeat: int = 1                # game frames per brain decision
    noop: int | None = None        # with repeat > 1: action on the repeated frames (None = hold the chosen action)
    potential: object = None       # obs -> Phi(obs), potential-based shaping used in training only
    bins: int = 5                  # choices per continuous action dimension
    max_steps: int | None = None   # only for games without a time limit of their own
    make: dict = field(default_factory=dict)
    train: dict = field(default_factory=dict)   # defaults for the training flags
    senses: list | None = None     # [(name when an input is < 0, name when it is > 0)]. None = automatic
    moves: list | None = None      # [name of each choice], for a game with one action head. None = automatic


def tiles(columns, n):
    """Sense names for a grid world: state 2k is +1 on input k, state 2k+1 is -1 (see discrete_code)."""
    def name(i):
        return f"row {i // columns + 1} col {i % columns + 1}" if i < n else ""
    return [(name(2 * k + 1), name(2 * k)) for k in range((n + 1) // 2)]


def over(*scale):
    scale = np.array(scale, dtype=float)
    return lambda obs: np.asarray(obs, dtype=float) / scale


def flappy_features(obs):
    import flappy_fly
    return flappy_fly.features(obs)


def mountain_car_energy(obs):
    """Height plus kinetic energy, about 0..100 (the game itself only gives -1 per step). Speed is progress."""
    x, v = obs
    return float(100 * ((np.sin(3 * x) + 1) * 0.5 + 100 * v * v))


PRESETS = {
    "cartpole": Preset("CartPole-v1", "balance a pole on a cart",
                       over(1.2, 1.5, 0.15, 2.0), 4, train=dict(steps=150_000),
                       senses=[("cart left", "cart right"), ("moving left", "moving right"),
                               ("leaning left", "leaning right"), ("tipping left", "tipping right")],
                       moves=["push left", "push right"]),
    "acrobot": Preset("Acrobot-v1", "swing a two-link arm above the line",
                      over(1, 1, 1, 1, 6.0, 12.0), 6, train=dict(steps=200_000),
                      senses=[("arm 1 up", "arm 1 down"), ("arm 1 left", "arm 1 right"), ("folded", "straight"),
                              ("bent cw", "bent ccw"), ("arm 1 turns cw", "arm 1 turns ccw"),
                              ("arm 2 turns cw", "arm 2 turns ccw")],
                      moves=["torque -1", "no torque", "torque +1"]),
    "mountaincar": Preset("MountainCar-v0", "rock a weak car out of a valley (trained with an energy hint)",
                          lambda o: np.array([(o[0] + 0.5) / 0.7, o[1] / 0.05]), 2,
                          potential=mountain_car_energy, train=dict(steps=200_000, lr=0.001),
                          senses=[("left of valley", "right of valley"), ("rolling left", "rolling right")],
                          moves=["push left", "coast", "push right"]),
    "lunarlander": Preset("LunarLander-v3", "land between the flags (needs gymnasium[box2d])",
                          over(0.8, 1.0, 0.8, 0.8, 0.5, 0.8, 1, 1), 8, train=dict(steps=600_000),
                          senses=[("left of pad", "right of pad"), ("below pad", "above pad"),
                                  ("drifting left", "drifting right"), ("falling", "rising"),
                                  ("tilted right", "tilted left"), ("turning cw", "turning ccw"),
                                  ("", "leg 1 down"), ("", "leg 2 down")],
                          moves=["do nothing", "left engine", "main engine", "right engine"]),
    "pendulum": Preset("Pendulum-v1", "swing up and hold a pendulum (continuous torque)",
                       over(1, 1, 6.0), 3, train=dict(steps=300_000, gamma=0.95),
                       senses=[("hanging", "upright"), ("on the right", "on the left"), ("turning cw", "turning ccw")]),
    "frozenlake": Preset("FrozenLake-v1", "cross slippery ice without falling in a hole",
                         train=dict(steps=300_000, entropy=0.01),
                         senses=tiles(4, 16), moves=["left", "down", "right", "up"]),
    "cliffwalking": Preset("CliffWalking-v1", "walk along a cliff edge to the goal",
                           max_steps=200, train=dict(steps=200_000, entropy=0.02),
                           senses=tiles(12, 48), moves=["up", "right", "down", "left"]),
    "blackjack": Preset("Blackjack-v1", "beat the dealer",
                        lambda o: np.array([(o[0] - 14) / 7, (o[1] - 5.5) / 4.5, 1.0 if o[2] else -1.0]), 3,
                        train=dict(steps=100_000),
                        senses=[("hand under 14", "hand over 14"), ("dealer low", "dealer high"),
                                ("no usable ace", "usable ace")],
                        moves=["stick", "hit"]),
    "flappy": Preset("FlappyBird-v0", "Flappy Bird (same features as flappy_fly.py; set --gap)",
                     flappy_features, 4, repeat=3, noop=0,
                     make=dict(use_lidar=False, pipe_gap=150, score_limit=100),
                     train=dict(steps=200_000, gamma=0.95),
                     senses=[("above gap", "below gap"), ("rising", "falling"), ("pipe far", "pipe close"),
                             ("high", "low")],
                     moves=["glide", "flap"]),
}


def discrete_code(n, i):
    v = np.zeros((n + 1) // 2)
    v[i // 2] = -1.0 if i % 2 else 1.0
    return v


def encoder(space):
    """(which values still need scaling, obs -> vector) for an observation space."""
    if isinstance(space, spaces.Discrete):
        n, start = int(space.n), int(space.start)
        return np.zeros((n + 1) // 2, bool), lambda o: discrete_code(n, int(o) - start)
    if isinstance(space, spaces.MultiBinary):
        n = int(np.prod(space.shape))
        return np.zeros(n, bool), lambda o: 2.0 * np.ravel(o) - 1
    if isinstance(space, spaces.MultiDiscrete):
        ns, starts = np.ravel(space.nvec).astype(int), np.ravel(space.start).astype(int)
        return (np.zeros(sum((n + 1) // 2 for n in ns), bool),
                lambda o: np.concatenate([discrete_code(n, int(i) - s) for n, s, i in zip(ns, starts, np.ravel(o))]))
    if isinstance(space, spaces.Box):
        n = int(np.prod(space.shape))
        if len(space.shape) >= 2 and n > 512:
            raise ValueError(f"image observations {space.shape} are not supported: the brain needs a short list of "
                             "numbers (for Atari use obs_type='ram')")
        return np.ones(n, bool), lambda o: np.ravel(o).astype(float)
    if isinstance(space, (spaces.Tuple, spaces.Dict)):
        keys = list(range(len(space.spaces))) if isinstance(space, spaces.Tuple) else list(space.spaces)
        parts = [(k, *encoder(space.spaces[k])) for k in keys]
        return (np.concatenate([mask for _, mask, _ in parts]),
                lambda o: np.concatenate([fn(o[k]) for k, _, fn in parts]))
    raise ValueError(f"observation space {space} is not supported")


def sense_names(space):
    """[(name when < 0, name when > 0)] for every value encoder(space) produces."""
    if isinstance(space, spaces.Discrete):
        n, start = int(space.n), int(space.start)
        names = [f"state {i + start}" for i in range(n)] + [""]
        return [(names[2 * k + 1], names[2 * k]) for k in range((n + 1) // 2)]
    if isinstance(space, spaces.MultiBinary):
        return [(f"{k} off", f"{k} on") for k in range(int(np.prod(space.shape)))]
    if isinstance(space, spaces.Box):
        return [(f"value {k} low", f"value {k} high") for k in range(int(np.prod(space.shape)))]
    if isinstance(space, spaces.MultiDiscrete):
        parts = [(k, spaces.Discrete(int(n), start=int(first)))
                 for k, (n, first) in enumerate(zip(np.ravel(space.nvec), np.ravel(space.start)))]
    else:
        keys = range(len(space.spaces)) if isinstance(space, spaces.Tuple) else list(space.spaces)
        parts = [(k, space.spaces[k]) for k in keys]
    return [tuple(f"{k}: {n}" if n else "" for n in pair) for k, part in parts for pair in sense_names(part)]


def move_names(space, bins):
    """[[name per choice] per head] for action_heads(space, bins)."""
    heads, act = action_heads(space, bins)
    if isinstance(space, spaces.Discrete):
        return [[f"action {i + int(space.start)}" for i in range(heads[0])]]
    if isinstance(space, spaces.Box):
        grid = np.array([np.ravel(act([c] * len(heads))) for c in range(bins)])
        return [[(f"{d}: " if len(heads) > 1 else "") + f"{grid[c, d]:+.3g}" for c in range(bins)]
                for d in range(len(heads))]
    if isinstance(space, spaces.MultiBinary):
        return [[f"{d}: off", f"{d}: on"] for d in range(len(heads))]
    starts = np.ravel(space.start).astype(int)
    return [[f"{d}: {c + starts[d]}" for c in range(n)] for d, n in enumerate(heads)]


def action_heads(space, bins):
    """(choices per head, [choice per head] -> env action) for an action space."""
    if isinstance(space, spaces.Discrete):
        return [int(space.n)], lambda c: int(c[0]) + int(space.start)
    if isinstance(space, spaces.MultiDiscrete):
        return ([int(n) for n in np.ravel(space.nvec)],
                lambda c: (np.reshape(c, space.shape) + space.start).astype(space.dtype))
    if isinstance(space, spaces.MultiBinary):
        return [2] * int(np.prod(space.shape)), lambda c: np.reshape(c, space.shape).astype(space.dtype)
    if isinstance(space, spaces.Box):
        lo = np.where(np.isfinite(space.low), space.low, -1).ravel()
        hi = np.where(np.isfinite(space.high), space.high, 1).ravel()
        grid = np.linspace(lo, hi, bins)                     # grid[choice, dimension]
        return ([bins] * lo.size,
                lambda c: grid[np.asarray(c), np.arange(lo.size)].reshape(space.shape).astype(space.dtype))
    raise ValueError(f"action space {space} is not supported")


class Game:
    """A gymnasium environment with brain-sized inputs and outputs."""

    def __init__(self, name, render=False, bins=None, make=None):
        """render: False, True (a window) or "rgb_array" (frames for fly_viewer.py)."""
        self.name = name
        self.on_frame = None           # called after every game frame, so a viewer sees the repeated ones too
        self.preset = p = PRESETS.get(name.lower()) or Preset(name)
        kwargs = {**p.make, **(make or {})}
        if p.env_id.startswith("FlappyBird"):
            import flappy_bird_gymnasium  # noqa: F401  (registers the env)
        if p.env_id not in gymnasium.registry and ":" not in p.env_id:
            raise ValueError(f"no game called '{name}'. Presets: {', '.join(PRESETS)}; "
                             "or use any gymnasium id, for example CartPole-v1")
        if render:
            kwargs["render_mode"] = "human" if render is True else render
        self.env = gymnasium.make(p.env_id, **kwargs)
        # some games never end on their own: random play on a cliff can wander for thousands of steps
        self.limit = None if self.env.spec.max_episode_steps else (p.max_steps or 1000)
        self.heads, self.to_action = action_heads(self.env.action_space, bins or p.bins)
        self.bins = bins or p.bins
        if p.features:
            self.encode, self.mask = p.features, np.zeros(p.n_obs, bool)
        else:
            self.mask, self.encode = encoder(self.env.observation_space)
        self.n_obs = len(self.mask)
        self.centre, self.scale = np.zeros(self.n_obs), np.ones(self.n_obs)

    @property
    def senses(self):
        return self.preset.senses or sense_names(self.env.observation_space)

    @property
    def moves(self):
        return [self.preset.moves] if self.preset.moves else move_names(self.env.action_space, self.bins)

    @property
    def needs_calibration(self):
        return bool(self.mask.any())

    def calibrate(self, steps=2000, seed=0):
        """Measure the range of every unscaled input while acting at random."""
        rng = np.random.default_rng(seed)
        rows, ep = [], 0
        while len(rows) < steps:
            obs, _ = self.env.reset(seed=800_000_000 + seed * 10_000 + ep)
            ep, t, done = ep + 1, 0, False
            rows.append(self.encode(obs))
            while not done:
                obs, _, term, trunc, _ = self.env.step(self.to_action(self.random_choice(rng)))
                t += 1
                rows.append(self.encode(obs))
                done = term or trunc or t == self.limit
        rows = np.array(rows)
        sd = rows.std(0)
        self.centre = np.where(self.mask, rows.mean(0), 0.0)
        self.scale = np.where(self.mask & (sd > 1e-6), 2 * sd, 1.0)

    def random_choice(self, rng):
        return [int(rng.integers(n)) for n in self.heads]

    def look(self, obs):
        return np.clip((self.encode(obs) - self.centre) / self.scale, -1, 1)

    def reset(self, seed):
        self.obs, self.info = self.env.reset(seed=seed)
        self.t = 0
        return self.look(self.obs)

    def step(self, choice, gamma=1.0):
        """Returns (inputs, game reward, shaping bonus, terminated, truncated)."""
        p = self.preset
        before = p.potential(self.obs) if p.potential else 0.0
        action = self.to_action(choice)
        reward = 0.0
        for k in range(p.repeat):
            a = action if k == 0 or p.noop is None else p.noop
            self.obs, r, term, trunc, self.info = self.env.step(a)
            self.t += 1
            reward += float(r)
            if self.on_frame:
                self.on_frame()
            trunc = trunc or (self.t == self.limit and not term)
            if term or trunc:
                break
        # Phi(terminal) = 0 keeps the best policy unchanged (Ng, Harada & Russell 1999)
        after = p.potential(self.obs) if p.potential and not term else 0.0
        bonus = gamma * after - before if p.potential else 0.0
        return self.look(self.obs), reward, bonus, term, trunc

    def close(self):
        self.env.close()
