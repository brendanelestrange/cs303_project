"""Let the FlyWire fruit-fly connectome learn gymnasium games by reinforcement learning.

Same spiking brain as flappy_fly.py / flappy_readout.py, for any game with a short list of numbers as observation:
game values -> input neuron populations -> 40 ms of simulated brain -> action.

Two ways to learn, both from the game's reward only:
  readout : the connectome is frozen; a linear readout of K neurons' spike counts is trained (PPO, actor + critic).
  stdp    : the connectome itself changes. Output populations vote for the action and dopamine-like
            reward-modulated STDP changes the synapses onto them; dopamine = reward prediction error.
            Learns simple cue -> action games; it did not learn CartPole (see FLY_GYM.md).

List the games:    python fly_gym.py list
Random baseline:   python fly_gym.py random cartpole
Train:             python fly_gym.py train cartpole --save cartpole.npz
Test frozen:       python fly_gym.py test --load cartpole.npz --episodes 100
Watch it play:     python fly_gym.py test --load cartpole.npz --episodes 3 --render
Any other game:    python fly_gym.py train MountainCarContinuous-v0 --save mcc.npz
Without the brain: python fly_gym.py train cartpole --no-brain      (how much does the connectome matter?)
"""
import argparse
import json
import os
import time
from pathlib import Path
from types import SimpleNamespace

import numpy as np

EVAL_SEED = 900_000_000        # evaluation games never overlap training games
VALID_SEED = 600_000_000       # games used to choose between greedy and sampled play
MAX_INPUT_NEURONS = 2400       # games with many inputs get smaller populations instead of using up the brain

GYM_HELPERS = r"""
# One decision: spike counts of the K read-out neurons, then of every output population.
function brain_step!(a, obs, pos, K)
    b = a.brain
    c = zeros(K + length(a.outputs))
    for _ in 1:a.steps
        for (d, x) in enumerate(obs)
            pop = x > 0 ? a.inputs[2d-1] : a.inputs[2d]
            p_spike = a.max_hz * min(abs(x), 1) * b.p.dt / 1000
            for i in pop
                rand() < p_spike && (b.g[i] += a.input_mv)
            end
        end
        for i in step!(b)
            pos[i] > 0 && (c[pos[i]] += 1)
            a.action_of[i] > 0 && (c[K + a.action_of[i]] += 1)
        end
    end
    b.e .*= exp(-a.steps * b.p.dt / b.p.tau_e)
    return c
end

as_pops(flat, pop) = collect.(Iterators.partition(Int.(collect(flat)), pop))
restore_wiring!(a, ins, outs, pop) = (set_wiring!(a, as_pops(ins, pop), as_pops(outs, pop)); nothing)
set_synapses!(a, w) = (a.brain.W.nzval .= w; nothing)
place!(pos, ids) = (for (k, i) in enumerate(ids); pos[Int(i)] = k; end; nothing)
flat(pops) = reduce(vcat, pops)
"""


class FlyBrainPlayer:
    """The connectome with one input pair per game value and one output population per choice."""

    def __init__(self, n_obs, n_out, cfg, wiring=None):
        import flappy_fly as ff
        import flappy_readout as fr
        self.pop = min(cfg["pop"], MAX_INPUT_NEURONS // (2 * n_obs))
        if self.pop < 2:
            raise ValueError(f"{n_obs} inputs are too many for this brain (about {MAX_INPUT_NEURONS // 4} at most)")
        self.jl, self.agent = ff.load_brain(SimpleNamespace(
            load=None, n_obs=n_obs, n_actions=n_out, neuropils=cfg["neuropils"], pop=self.pop,
            action_ms=cfg["action_ms"], seed=cfg["seed"], max_hz=200.0, input_mv=70.0, lr=cfg["stdp_lr"],
            tau_e=500.0, outputs="downstream", select_wiring=0, print_wiring_score=False))
        jl = self.jl
        jl.seval(fr.RES_HELPERS)
        jl.seval(GYM_HELPERS)
        self.agent.brain.learning = False
        self.K = cfg["K"]
        if wiring:        # a saved player: put the exact same neurons back instead of trusting the seed
            jl.restore_wiring_b(self.agent, wiring["inputs"], wiring["outputs"], self.pop)
            ids = wiring["ids"]
            if len(wiring["synapses"]):
                if len(wiring["synapses"]) != len(self.agent.brain.W.nzval):
                    raise ValueError("this player was trained on a different connectome file")
                jl.set_synapses_b(self.agent, wiring["synapses"])
        else:
            ids = jl.reservoir_ids(self.agent, self.K)
        self.ids = np.array(ids).astype(np.int64)
        self.pos = jl.zeros(jl.Int, len(self.agent.brain.v))
        jl.place_b(self.pos, self.ids)

    def describe(self):
        return (f"{len(self.agent.brain.v)} neurons, {len(self.agent.brain.W.nzval)} synapses, "
                f"{self.pop} neurons per input/output population, {self.K} read out")

    def reset(self):
        self.jl.reset_b(self.agent)

    def step(self, f):
        """(spike counts of the read-out neurons, spike counts of the output populations)."""
        c = np.array(self.jl.brain_step_b(self.agent, f, self.pos, self.K), dtype=np.float64)
        return c[:self.K], c[self.K:]

    def wiring(self, learned):
        flat = lambda p: np.array(self.jl.flat(p)).astype(np.int64)
        return dict(inputs=flat(self.agent.inputs), outputs=flat(self.agent.outputs), ids=self.ids,
                    synapses=np.array(self.agent.brain.W.nzval) if learned else np.zeros(0))


class NoBrain:
    """Control: the readout sees what the input populations would get, with no neurons in between."""

    def __init__(self, n_obs, n_out):
        self.K, self.n_out = 2 * n_obs, n_out

    def describe(self):
        return f"no brain: the readout sees the {self.K} input values directly"

    def reset(self):
        pass

    def step(self, f):
        return np.concatenate([np.maximum(f, 0), np.maximum(-f, 0)]), np.zeros(self.n_out)

    def wiring(self, learned):
        return dict(inputs=[], outputs=[], ids=[], synapses=[])


def log_softmax(z):
    z = z - z.max(-1, keepdims=True)
    return z - np.log(np.exp(z).sum(-1, keepdims=True))


class Adam:
    def __init__(self, shape):
        self.m, self.v, self.t = np.zeros(shape), np.zeros(shape), 0

    def step(self, grad, lr):
        self.t += 1
        self.m = 0.9 * self.m + 0.1 * grad
        self.v = 0.999 * self.v + 0.001 * grad ** 2
        return lr * (self.m / (1 - 0.9 ** self.t)) / (np.sqrt(self.v / (1 - 0.999 ** self.t)) + 1e-8)


class Readout:
    """Linear softmax policy (one softmax per head) and linear value estimate on normalised spike counts."""

    def __init__(self, K, heads, mu, sd):
        self.mu, self.sd, self.heads = mu, sd, heads
        ends = np.cumsum(heads)
        self.slices = [slice(e - n, e) for n, e in zip(heads, ends)]
        self.w = np.zeros((int(ends[-1]), K + 1))
        self.v = np.zeros(K + 1)
        self.greedy = True             # frozen play: take the likeliest action, or sample as in training

    def x(self, counts):
        return np.append(np.clip((counts - self.mu) / self.sd, -5, 5), 1.0)

    def value(self, x):
        return x @ self.v

    def logp(self, x):
        z = x @ self.w.T
        return np.concatenate([log_softmax(z[..., s]) for s in self.slices], axis=-1)

    def choose(self, x, rng, greedy=False):
        lp = self.logp(x)
        out = []
        for s in self.slices:
            p = np.exp(lp[s])
            out.append(int(np.argmax(p)) if greedy else int(min(np.searchsorted(np.cumsum(p), rng.random()), len(p) - 1)))
        return out

    def chosen(self, choices):
        """One-hot matrix of the choices made, over all heads."""
        hot = np.zeros((len(choices), self.w.shape[0]))
        for s, c in zip(self.slices, np.asarray(choices).T):
            hot[np.arange(len(choices)), s.start + c] = 1
        return hot


class Voter:
    """STDP players: the output population that fires most above its own running average wins its head."""

    def __init__(self, heads, alpha=0.02):
        ends = np.cumsum(heads)
        self.slices = [slice(e - n, e) for n, e in zip(heads, ends)]
        self.heads, self.alpha, self.base = heads, alpha, np.zeros(int(ends[-1]))

    def choose(self, counts, rng, eps=0.0):
        rel = counts - self.base
        self.base = self.base + self.alpha * (counts - self.base)
        out = []
        for s in self.slices:
            best = np.flatnonzero(rel[s] == rel[s].max())
            out.append(int(rng.integers(s.stop - s.start)) if rng.random() < eps else int(rng.choice(best)))
        return out


def advantages(rew, val, next_val, term, done, gamma, lam):
    """Generalised advantage estimation. next_val[t] is V(s_t+1), kept at time-outs and dropped at real endings."""
    adv, last = np.zeros(len(rew)), 0.0
    for t in reversed(range(len(rew))):
        delta = rew[t] + gamma * next_val[t] * (not term[t]) - val[t]
        last = delta + gamma * lam * (not done[t]) * last
        adv[t] = last
    return adv


def ppo_update(ro, opt_w, opt_v, X, choices, adv, ret, args, lr, rng):
    hot = ro.chosen(choices)
    old = (ro.logp(X) * hot).sum(1)
    adv = (adv - adv.mean()) / (adv.std() + 1e-8)
    n = len(X)
    for _ in range(args.epochs):
        for mb in np.array_split(rng.permutation(n), max(n // args.minibatch, 1)):
            x, h = X[mb], hot[mb]
            lp = ro.logp(x)
            p = np.exp(lp)
            ratio = np.exp((lp * h).sum(1) - old[mb])
            a = adv[mb]
            clipped = ((ratio > 1.2) & (a > 0)) | ((ratio < 0.8) & (a < 0))
            g = np.where(clipped, 0.0, ratio * a)[:, None] * (h - p)
            for s in ro.slices:                              # entropy bonus keeps it exploring
                H = -(p[:, s] * lp[:, s]).sum(1, keepdims=True)
                g[:, s] -= args.entropy * p[:, s] * (lp[:, s] + H)
            ro.w += opt_w.step(g.T @ x / len(mb), lr)
            ro.v += opt_v.step((ret[mb] - x @ ro.v) @ x / len(mb), lr)
        if np.mean(old - (ro.logp(X) * hot).sum(1)) > 0.03:  # policy moved far enough for this batch of experience
            break


def warm_up(game, player, seed, steps=2000):
    """Random play: mean and spread of every read-out neuron."""
    rng = np.random.default_rng(seed)
    rows, ep = [], 0
    while len(rows) < steps:
        f = game.reset(700_000_000 + seed * 10_000 + ep)
        player.reset()
        ep, done = ep + 1, False
        while not done:
            rows.append(player.step(f)[0])
            f, _, _, term, trunc = game.step(game.random_choice(rng))
            done = term or trunc
    rows = np.array(rows)
    return rows.mean(0), rows.std(0) + (0.5 if isinstance(player, FlyBrainPlayer) else 0.1)


class RewardScale:
    """Running spread of the discounted return, so one learning rate suits games with rewards of 1 or 100."""

    def __init__(self, gamma):
        self.gamma, self.G, self.n, self.mean, self.sq = gamma, 0.0, 0, 0.0, 0.0

    def __call__(self, r, done):
        self.G = r + self.gamma * self.G
        self.n += 1
        d = self.G - self.mean
        self.mean += d / self.n
        self.sq += d * (self.G - self.mean)
        if done:
            self.G = 0.0
        return r / max(np.sqrt(self.sq / self.n), 1e-3) if self.n > 1 else r


class Log:
    def __init__(self, every):
        self.every, self.next, self.t0 = every, every, time.time()
        self.returns, self.at = [], []

    def episode(self, total, steps):
        self.returns.append(total)
        self.at.append(steps)

    def maybe_print(self, steps, extra=""):
        if steps < self.next:
            return
        self.next += self.every
        last = self.returns[-50:]
        print(f"step {steps:7d}  games {len(self.returns):5d}  last{len(last)} mean return "
              f"{np.mean(last) if last else float('nan'):8.2f}  [{time.time() - self.t0:.0f}s]{extra}", flush=True)


def train_readout(args, game, player, save):
    rng = np.random.default_rng(args.seed)
    ro = Readout(player.K, game.heads, *warm_up(game, player, args.seed))
    opt_w, opt_v = Adam(ro.w.shape), Adam(ro.v.shape)
    scale, log = RewardScale(args.gamma), Log(max(args.steps // 20, 1))
    ep, total, steps, updates = 0, 0.0, 0, 0
    f = game.reset(args.seed * 1_000_000)
    player.reset()
    x = ro.x(player.step(f)[0])
    while steps < args.steps:
        X, C, R, V, NV, T, D = [], [], [], [], [], [], []
        for _ in range(args.horizon):
            c = ro.choose(x, rng)
            f, r, bonus, term, trunc = game.step(c, args.gamma)
            total += r
            steps += 1
            done = term or trunc
            nx = ro.x(player.step(f)[0])                     # also at the end: a time-out still has a future
            X.append(x); C.append(c); V.append(ro.value(x)); NV.append(ro.value(nx))
            R.append(scale(r + bonus, done)); T.append(term); D.append(done)
            x = nx
            if done:
                log.episode(total, steps)
                ep, total = ep + 1, 0.0
                f = game.reset(args.seed * 1_000_000 + ep)
                player.reset()
                x = ro.x(player.step(f)[0])
        V = np.array(V)
        adv = advantages(R, V, NV, T, D, args.gamma, args.lam)
        lr = args.lr * max(1 - steps / args.steps, 0.05)
        ppo_update(ro, opt_w, opt_v, np.array(X), C, adv, adv + V, args, lr, rng)
        updates += 1
        log.maybe_print(steps)
        if updates % 10 == 0:
            save(dict(ro=ro), log)
    return dict(ro=ro), log


def train_stdp(args, game, player, save):
    """Three-factor learning: spike-timing eligibility x dopamine, dopamine = reward prediction error."""
    rng = np.random.default_rng(args.seed)
    jl, agent = player.jl, player.agent
    critic = Readout(player.K, game.heads, *warm_up(game, player, args.seed))   # only its value estimate is used
    voter = Voter(game.heads)
    scale, log = RewardScale(args.gamma), Log(max(args.steps // 20, 1))
    agent.brain.learning = True
    ep, steps, dopamine_sq, ckpt = 0, 0, 1.0, args.steps // 10
    while steps < args.steps:
        f = game.reset(args.seed * 1_000_000 + ep)
        player.reset()
        counts, out = player.step(f)
        x, total, done = critic.x(counts), 0.0, False
        while not done:
            eps = args.epsilon * max(1 - 2 * steps / args.steps, 0.0)       # explore in the first half only
            c = voter.choose(out, rng, eps)
            f, r, bonus, term, trunc = game.step(c, args.gamma)
            total += r
            steps += 1
            done = term or trunc
            counts, out = player.step(f)
            nx = critic.x(counts)
            delta = scale(r + bonus, done) + args.gamma * critic.value(nx) * (not term) - critic.value(x)
            critic.v += args.critic_lr * delta * x / (x @ x)                # normalised TD(0)
            dopamine_sq += 0.01 * (delta ** 2 - dopamine_sq)
            dopamine = float(np.clip(delta / np.sqrt(dopamine_sq + 1e-8), -3, 3))
            # only synapses onto the chosen action's population. FlyBrain.reward! (every excitatory synapse)
            # learned the reverse of the right answer here on a two-cue bandit, cause not found.
            for s, choice in zip(voter.slices, c):
                jl.reward_chosen_b(agent, dopamine, s.start + choice)
            x = nx
        log.episode(total, steps)
        ep += 1
        log.maybe_print(steps)
        if steps >= ckpt:
            ckpt += args.steps // 10
            save(dict(voter=voter), log)
    agent.brain.learning = False
    return dict(voter=voter), log


def play_frozen(game, player, policy, episodes, seed0=EVAL_SEED):
    """Evaluation: nothing learns, no exploration, the game's own reward. policy=None plays at random."""
    rng = np.random.default_rng(12345)
    returns, lengths = [], []
    if "voter" in (policy or {}):
        v = policy["voter"]
        frozen = Voter(v.heads)
        frozen.base = v.base.copy()
    for g in range(episodes):
        f = game.reset(seed0 + g)
        if player:
            player.reset()
        total, n, done = 0.0, 0, False
        while not done:
            if policy is None:
                c = game.random_choice(rng)
            else:
                counts, out = player.step(f)
                c = frozen.choose(out, rng) if "voter" in policy else policy["ro"].choose(
                    policy["ro"].x(counts), rng, greedy=policy["ro"].greedy)
            f, r, _, term, trunc = game.step(c)
            total += r
            n += 1
            done = term or trunc
        returns.append(total)
        lengths.append(n)
    return np.array(returns), np.array(lengths)


def pick_play_style(game, player, policy, episodes=30):
    """Some policies need their randomness (a greedy lander just hovers). Decide on games kept apart from the test."""
    ro, means = policy["ro"], {}
    for ro.greedy in (True, False):
        means[ro.greedy] = play_frozen(game, player, policy, episodes, VALID_SEED)[0].mean()
    ro.greedy = bool(means[True] >= means[False])
    print(f"validation games: greedy {means[True]:.2f}, sampled {means[False]:.2f} "
          f"-> plays {'greedy' if ro.greedy else 'sampled'}", flush=True)


def report(label, returns, lengths, game):
    print(f"{label}: mean return {returns.mean():.2f} +- {returns.std() / np.sqrt(len(returns)):.2f} (s.e.), "
          f"median {np.median(returns):.2f}, best {returns.max():.2f}, mean length {lengths.mean():.0f} decisions, "
          f"{len(returns)} games")
    threshold = game.env.spec.reward_threshold
    if threshold is not None:
        print(f"gymnasium counts this game as solved at {threshold}: "
              f"{'solved' if returns.mean() >= threshold else 'not solved'}")


def write_player(path, args, game, player, policy, log):
    cfg = {k: getattr(args, k) for k in ("learner", "no_brain", "neuropils", "pop", "K", "action_ms", "seed",
                                         "stdp_lr", "gap")}
    cfg.update(game=game.name, bins=game.bins, heads=game.heads, n_obs=game.n_obs)
    data = dict(config=json.dumps(cfg), centre=game.centre, scale=game.scale, returns=log.returns, at=log.at,
                **{f"wiring_{k}": v for k, v in player.wiring("voter" in policy).items()})
    if "ro" in policy:
        ro = policy["ro"]
        data.update(w=ro.w, v=ro.v, mu=ro.mu, sd=ro.sd, greedy=ro.greedy)
    else:
        data.update(base=policy["voter"].base)
    tmp = Path(str(path) + ".tmp")
    with open(tmp, "wb") as fh:                              # never leave a half-written player behind
        np.savez_compressed(fh, **data)
    os.replace(tmp, path)


def read_player(path, render):
    with np.load(path, allow_pickle=False) as z:
        d = {k: z[k] for k in z.files}
    cfg = json.loads(str(d["config"]))
    game = make_game(cfg["game"], render, cfg["bins"], cfg["gap"])
    if (game.n_obs, game.heads) != (cfg["n_obs"], cfg["heads"]):
        raise ValueError(f"{cfg['game']} now has different inputs or actions than when this player was trained")
    game.centre, game.scale = d["centre"], d["scale"]
    if cfg["no_brain"]:
        player = NoBrain(game.n_obs, sum(game.heads))
    else:
        player = FlyBrainPlayer(game.n_obs, sum(game.heads), cfg,
                                {k[7:]: v for k, v in d.items() if k.startswith("wiring_")})
    if "w" in d:
        ro = Readout(player.K, game.heads, d["mu"], d["sd"])
        ro.w, ro.v, ro.greedy = d["w"], d["v"], bool(d["greedy"])
        return game, player, dict(ro=ro)
    voter = Voter(game.heads)
    voter.base = d["base"]
    return game, player, dict(voter=voter)


def make_game(name, render=False, bins=None, gap=None):
    if not render:
        os.environ["SDL_VIDEODRIVER"] = "dummy"
    from fly_games import Game
    return Game(name, render, bins, dict(pipe_gap=gap) if gap else None)


def parse_args(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)
    sub.add_parser("list", help="the tuned games")

    tr = sub.add_parser("train", help="learn a game")
    tr.add_argument("game", help="a preset (see list) or any gymnasium id")
    tr.add_argument("--learner", choices=["readout", "stdp"], default="readout")
    tr.add_argument("--no-brain", action="store_true", help="control: train the readout on the raw inputs")
    tr.add_argument("--steps", type=int, help="brain decisions to train for (default: the game's preset, else 200000)")
    tr.add_argument("--save", help="write the trained player to this .npz file (also every 10 updates)")
    tr.add_argument("--eval-games", type=int, default=100, help="frozen evaluation games at the end (0 = none)")
    tr.add_argument("--seed", type=int, default=0)
    tr.add_argument("--bins", type=int, help="choices per continuous action dimension (default 5)")
    tr.add_argument("--gap", type=int, help="flappy only: pipe gap in pixels (default 150)")
    tr.add_argument("--neuropils", default="AL_L,LH_L,MB_CA_L", help='comma list, or "all" for the whole brain')
    tr.add_argument("--pop", type=int, default=20, help="neurons per input/output population")
    tr.add_argument("--K", type=int, default=500, help="neurons read out")
    tr.add_argument("--action-ms", type=int, default=40, help="simulated brain time per decision (at 20 ms the "
                    "spikes still mostly describe the previous decision's inputs)")
    tr.add_argument("--gamma", type=float, help="discount per decision (default 0.99)")
    tr.add_argument("--lr", type=float, help="readout learning rate (default 0.003)")
    tr.add_argument("--lam", type=float, default=0.95, help="GAE lambda")
    tr.add_argument("--horizon", type=int, default=2048, help="decisions per update")
    tr.add_argument("--epochs", type=int, default=10)
    tr.add_argument("--minibatch", type=int, default=128)
    tr.add_argument("--entropy", type=float, help="exploration bonus (default 0.003)")
    tr.add_argument("--stdp-lr", type=float, default=0.2, help="stdp: synapse learning rate")
    tr.add_argument("--critic-lr", type=float, default=0.05, help="stdp: learning rate of the reward prediction")
    tr.add_argument("--epsilon", type=float, default=0.2, help="stdp: random actions at the start")

    te = sub.add_parser("test", help="play a saved player, frozen")
    te.add_argument("--load", required=True)
    te.add_argument("--episodes", type=int, default=100)
    te.add_argument("--render", action="store_true", help="show the game window")

    ra = sub.add_parser("random", help="random-action baseline")
    ra.add_argument("game")
    ra.add_argument("--episodes", type=int, default=100)
    ra.add_argument("--bins", type=int)
    ra.add_argument("--gap", type=int)
    args = ap.parse_args(argv)

    if args.cmd == "train":
        from fly_games import PRESETS
        preset = PRESETS.get(args.game.lower())
        for key, value in {"steps": 200_000, "gamma": 0.99, "lr": 0.003, "entropy": 0.003, **(preset.train if preset else {})}.items():
            if getattr(args, key) is None:
                setattr(args, key, value)
        if args.no_brain and args.learner == "stdp":
            ap.error("--no-brain only works with --learner readout (STDP needs synapses)")
        if args.save and not args.save.endswith(".npz"):
            ap.error("--save needs a file name ending in .npz")
    if args.cmd == "test" and not Path(args.load).exists():
        ap.error(f"player file '{args.load}' not found. Train one with: python fly_gym.py train GAME --save {args.load}")
    return args


def main(argv=None):
    args = parse_args(argv)
    try:
        run(args)
    except (ValueError, ImportError) as e:
        raise SystemExit(f"error: {e}")


def run(args):
    if args.cmd == "list":
        from fly_games import PRESETS
        for name, p in PRESETS.items():
            print(f"{name:13s} {p.env_id:18s} {p.note}")
        print("\nAny other gymnasium id works too if its observations are a short list of numbers.")
        return
    if args.cmd == "random":
        game = make_game(args.game, bins=args.bins, gap=args.gap)
        report(f"random {args.game}", *play_frozen(game, None, None, args.episodes), game)
        return game.close()
    if args.cmd == "test":
        game, player, policy = read_player(args.load, args.render)
        print(player.describe())
        report(f"test {game.name}", *play_frozen(game, player, policy, args.episodes), game)
        return game.close()

    game = make_game(args.game, bins=args.bins, gap=args.gap)
    if game.needs_calibration:
        game.calibrate(seed=args.seed)
    n_out = sum(game.heads)
    print(f"{game.preset.env_id}: {game.n_obs} inputs, {len(game.heads)} action head(s) with {game.heads} choices")
    t0 = time.time()
    player = NoBrain(game.n_obs, n_out) if args.no_brain else FlyBrainPlayer(game.n_obs, n_out, vars(args))
    print(f"ready in {time.time() - t0:.0f}s: {player.describe()}", flush=True)

    def save(policy, log):
        if args.save:
            write_player(args.save, args, game, player, policy, log)

    t0 = time.time()
    policy, log = (train_stdp if args.learner == "stdp" else train_readout)(args, game, player, save)
    if "ro" in policy:
        pick_play_style(game, player, policy)
    save(policy, log)
    print(f"training took {time.time() - t0:.0f}s" + (f", saved {args.save}" if args.save else ""), flush=True)
    if args.eval_games:
        report("random", *play_frozen(game, None, None, args.eval_games), game)
        report("trained, frozen", *play_frozen(game, player, policy, args.eval_games), game)
    game.close()


if __name__ == "__main__":
    main()
