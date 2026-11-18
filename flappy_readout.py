"""Option C: the fly connectome as a fixed "reservoir" + a small trained readout.

The FlyWire circuit is simulated exactly as in flappy_fly.py, but its synapses never change
(no STDP). Each decision, the 4 game features drive the input neurons for 20 ms of brain time;
we record the spike counts of the K neurons that receive the most excitatory input from them.
A logistic readout p(flap) = sigmoid(w . counts) - the only thing that is trained - picks the action.

Two ways to train the readout:
  reinforce : game reward only (REINFORCE policy gradient). No knowledge of the right answer.
  imitate   : TEACHER - copies the hand-written rule (flap when below the gap centre and not rising),
              with DAgger rounds. This uses our knowledge of the answer; label it as such.

Train (reward only):  python flappy_readout.py train --mode reinforce --episodes 2000 --save readout_rl.jls
Train (teacher):      python flappy_readout.py train --mode imitate --save readout_teacher.jls
Test frozen:          python flappy_readout.py test --load readout_rl.jls --episodes 200
Watch:                python flappy_readout.py test --load readout_rl.jls --episodes 3 --render
"""
import argparse
import os
import time
from pathlib import Path
from types import SimpleNamespace

import numpy as np

import flappy_fly as ff

RES_HELPERS = r"""
function reservoir_ids(a, K)
    W = a.brain.W; N = size(W, 1)
    s = zeros(N); for p in a.inputs, i in p; s[i] = 1.0; end
    exc = Float64.(a.brain.sign .== 1)
    score = W * (s .* exc)
    score[s .> 0] .= -Inf
    return sortperm(score, rev=true)[1:K]
end

function reservoir_counts!(a, obs, pos, K)
    b = a.brain
    c = zeros(K)
    for _ in 1:a.steps
        for (d, x) in enumerate(obs)
            pop = x > 0 ? a.inputs[2d-1] : a.inputs[2d]
            p_spike = a.max_hz * min(abs(x), 1) * b.p.dt / 1000
            for i in pop
                rand() < p_spike && (b.g[i] += a.input_mv)
            end
        end
        for i in step!(b)
            k = pos[i]
            k > 0 && (c[k] += 1)
        end
    end
    return c
end

save_readout(path, w, mu, sd, meta) = serialize(path, (w=collect(Float64, w), mu=collect(Float64, mu),
                                                     sd=collect(Float64, sd), meta=collect(Float64, meta)))
"""


def hand_rule(f):
    """The hand-written teacher: flap when below the gap centre and not already rising."""
    return int(f[0] > 0.1 and f[1] > -0.2)


class Reservoir:
    def __init__(self, K, seed, neuropils, pop):
        a = SimpleNamespace(load=None, neuropils=neuropils, pop=pop, action_ms=20, seed=seed,
                            max_hz=200.0, input_mv=70.0, lr=0.0, tau_e=500.0, outputs="indegree",
                            select_wiring=0, print_wiring_score=False)
        self.jl, self.agent = ff.load_brain(a)
        jl = self.jl
        jl.seval(RES_HELPERS)
        self.agent.brain.learning = False          # the connectome never changes
        ids = jl.reservoir_ids(self.agent, K)
        self.pos = jl.zeros(jl.Int, len(self.agent.brain.v))
        jl.seval("(pos, ids) -> (pos[ids] .= 1:length(ids); nothing)")(self.pos, ids)
        self.K = K

    def reset(self):
        self.jl.reset_b(self.agent)

    def counts(self, f):
        return np.asarray(self.jl.reservoir_counts_b(self.agent, f, self.pos, self.K), dtype=np.float64)


def sigmoid(z):
    return 1.0 / (1.0 + np.exp(-np.clip(z, -30, 30)))


class Readout:
    def __init__(self, K, mu=None, sd=None, w=None):
        self.mu = np.zeros(K) if mu is None else mu
        self.sd = np.ones(K) if sd is None else sd
        self.w = np.zeros(K + 1) if w is None else w
        if w is None:
            self.w[-1] = -1.4                         # start flapping about 20% of the time

    def x(self, counts):
        return np.append(np.clip((counts - self.mu) / self.sd, -5, 5), 1.0)

    def p(self, x):
        return sigmoid(self.w @ x)


def make_env(gap, render, score_limit):
    import gymnasium
    import flappy_bird_gymnasium  # noqa: F401
    return gymnasium.make("FlappyBird-v0", render_mode="human" if render else None, use_lidar=False,
                          pipe_gap=gap, score_limit=score_limit)


def play(env, res, policy, seed, frame_skip=3, record=None):
    """One game. policy(f, counts) -> action. record(f, counts, action, reward) per decision."""
    obs, info = env.reset(seed=seed)
    res.reset()
    done = False
    while not done:
        f = ff.features(obs)
        c = res.counts(f)
        a = policy(f, c)
        r = 0.0
        for k in range(frame_skip):
            obs, rr, term, trunc, info = env.step(a if k == 0 else 0)
            r += rr
            done = term or trunc
            if done:
                break
        if record:
            record(f, c, a, r)
    return info["score"]


def fit_scaler(env, res, games, seed):
    rows = []
    for g in range(games):
        play(env, res, lambda f, c: hand_rule(f) if np.random.rand() > 0.3 else int(np.random.rand() < 0.2),
             seed + g, record=lambda f, c, a, r: rows.append(c))
    X = np.array(rows)
    return X.mean(0), X.std(0) + 0.5


def save(path, ro, args):
    meta = [args.K, args.seed]
    ro_jl = ro
    RES.jl.save_readout(str(Path(path).resolve()), ro_jl.w, ro_jl.mu, ro_jl.sd, meta)


def load(path):
    d = RES_BOOT.jl.seval(f'deserialize(raw"{Path(path).resolve()}")')
    meta = np.asarray(d.meta)
    return Readout(int(meta[0]), np.asarray(d.mu), np.asarray(d.sd), np.asarray(d.w)), int(meta[0]), int(meta[1])


def evaluate(env, res, ro, games, seed0):
    scores = [play(env, res, lambda f, c: int(ro.p(ro.x(c)) > 0.5), seed0 + g) for g in range(games)]
    return np.array(scores)


def train_reinforce(args, env, res, ro):
    """REINFORCE with Adam on the logistic readout; reward = the game's own reward only."""
    m, v, t = np.zeros_like(ro.w), np.zeros_like(ro.w), 0
    batch, scores, t0 = [], [], time.time()
    for ep in range(args.episodes):
        xs, acts, rews = [], [], []

        def pol(f, c):
            x = ro.x(c)
            a = int(np.random.rand() < ro.p(x))
            xs.append(x); acts.append(a)
            return a

        scores.append(play(env, res, pol, args.seed * 100_000 + ep,
                           record=lambda f, c, a, r: rews.append(r)))
        G, ret = 0.0, []
        for r in reversed(rews):
            G = r + args.gamma * G
            ret.append(G)
        batch.append((np.array(xs), np.array(acts), np.array(ret[::-1])))
        if len(batch) == args.batch:
            allret = np.concatenate([b[2] for b in batch])
            mu_r, sd_r = allret.mean(), allret.std() + 1e-6
            grad = np.zeros_like(ro.w)
            for X, A, R in batch:
                P = sigmoid(X @ ro.w)
                grad += ((A - P) * (R - mu_r) / sd_r) @ X
            grad /= len(allret)
            t += 1
            m = 0.9 * m + 0.1 * grad
            v = 0.999 * v + 0.001 * grad ** 2
            ro.w += args.lr * (m / (1 - 0.9 ** t)) / (np.sqrt(v / (1 - 0.999 ** t)) + 1e-8)
            batch = []
        if (ep + 1) % 50 == 0:
            print(f"ep {ep + 1:5d}  last50 pipes {np.mean(scores[-50:]):.2f} (best {max(scores[-50:])})  "
                  f"[{time.time() - t0:.0f}s]", flush=True)
        if args.save and (ep + 1) % 200 == 0:
            save(args.save, ro, args)
            print(f"ep {ep + 1:5d}  checkpoint saved to {args.save}", flush=True)


def fit_logistic(X, y, ro, iters=600, lr=0.05, l2=1e-3):
    w = ro.w.copy()
    pos_w = (1 - y.mean()) / max(y.mean(), 1e-3)          # balance flap/glide
    sw = np.where(y == 1, pos_w, 1.0)
    m, v = np.zeros_like(w), np.zeros_like(w)
    for t in range(1, iters + 1):
        P = sigmoid(X @ w)
        g = X.T @ (sw * (y - P)) / sw.sum() - l2 * w
        m = 0.9 * m + 0.1 * g
        v = 0.999 * v + 0.001 * g ** 2
        w += lr * (m / (1 - 0.9 ** t)) / (np.sqrt(v / (1 - 0.999 ** t)) + 1e-8)
    ro.w = w
    return np.mean((sigmoid(X @ w) > 0.5) == y)


def train_imitate(args, env, res, ro):
    """TEACHER: behaviour cloning of the hand rule + DAgger (the rule labels states the readout visits)."""
    X, y = [], []
    rec = lambda f, c, a, r: (X.append(ro.x(c)), y.append(hand_rule(f)))
    for g in range(args.teacher_games):
        play(env, res, lambda f, c: hand_rule(f) if np.random.rand() > 0.1 else 1 - hand_rule(f),
             args.seed * 100_000 + g, record=rec)
    for rnd in range(args.dagger + 1):
        acc = fit_logistic(np.array(X), np.array(y, dtype=float), ro)
        sc = [play(env, res, lambda f, c: int(ro.p(ro.x(c)) > 0.5), 7_000_000 + rnd * 1000 + g, record=rec)
              for g in range(args.dagger_games)]
        print(f"round {rnd}: {len(y)} labelled decisions, fit accuracy {acc:.3f}, "
              f"readout plays {np.mean(sc):.2f} pipes/game (best {max(sc)})", flush=True)
        if args.save:
            save(args.save, ro, args)
            print(f"checkpoint saved to {args.save}", flush=True)


RES = RES_BOOT = None


def main():
    global RES, RES_BOOT
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("cmd", choices=["train", "test"])
    ap.add_argument("--mode", choices=["reinforce", "imitate"], default="reinforce")
    ap.add_argument("--episodes", type=int, default=2000, help="train: games (reinforce); test: games to evaluate")
    ap.add_argument("--K", type=int, default=300, help="number of reservoir neurons read out")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--neuropils", default="AL_L,LH_L,MB_CA_L")
    ap.add_argument("--pop", type=int, default=20)
    ap.add_argument("--pipe-gap", type=int, default=100)
    ap.add_argument("--score-limit", type=int, default=100, help="end a game after this many pipes (keeps runs bounded)")
    ap.add_argument("--lr", type=float, default=0.02)
    ap.add_argument("--gamma", type=float, default=0.95)
    ap.add_argument("--batch", type=int, default=10)
    ap.add_argument("--teacher-games", type=int, default=40)
    ap.add_argument("--dagger", type=int, default=4)
    ap.add_argument("--dagger-games", type=int, default=20)
    ap.add_argument("--eval-games", type=int, default=200, help="train: frozen evaluation games at the end")
    ap.add_argument("--save")
    ap.add_argument("--load")
    ap.add_argument("--render", action="store_true")
    args = ap.parse_args()
    if not args.render:
        os.environ["SDL_VIDEODRIVER"] = "dummy"
    np.random.seed(args.seed)

    t0 = time.time()
    if args.cmd == "test":
        if not args.load or not Path(args.load).exists():
            raise SystemExit(f"Readout file '{args.load}' not found. Train one first with: "
                             f"python flappy_readout.py train --save {args.load or 'readout.jls'}")
        # Julia must be running to read the file; build the default reservoir, rebuild if the file differs
        res_tmp = Reservoir(args.K, args.seed, args.neuropils, args.pop)
        RES_BOOT = res_tmp
        ro, K, seed = load(args.load)
        res = res_tmp if (K, seed) == (args.K, args.seed) else Reservoir(K, seed, args.neuropils, args.pop)
        RES = res
        env = make_env(args.pipe_gap, args.render, args.score_limit)
        print(f"reservoir ready in {time.time() - t0:.0f}s ({K} neurons read out, wiring seed {seed})")
        s = evaluate(env, res, ro, args.episodes, 9_000_000)
        print(f"test gap {args.pipe_gap}: mean pipes {s.mean():.2f}, median {np.median(s):.0f}, "
              f"best {s.max()}, games hitting the {args.score_limit}-pipe cap {int((s >= args.score_limit).sum())}/{len(s)}")
        env.close()
        return

    res = RES = RES_BOOT = Reservoir(args.K, args.seed, args.neuropils, args.pop)
    env = make_env(args.pipe_gap, False, args.score_limit)
    print(f"reservoir ready in {time.time() - t0:.0f}s ({args.K} neurons read out; connectome frozen)")
    mu, sd = fit_scaler(env, res, 10, 5_000_000)
    ro = Readout(args.K, mu, sd)
    t0 = time.time()
    if args.mode == "reinforce":
        train_reinforce(args, env, res, ro)
    else:
        train_imitate(args, env, res, ro)
    print(f"training took {time.time() - t0:.0f}s")
    if args.save:
        save(args.save, ro, args)
        print("saved", args.save)
    if args.eval_games:
        s = evaluate(env, res, ro, args.eval_games, 9_000_000)
        print(f"frozen eval gap {args.pipe_gap}, {args.eval_games} games: mean pipes {s.mean():.2f}, "
              f"median {np.median(s):.0f}, best {s.max()}, hit {args.score_limit}-pipe cap "
              f"{int((s >= args.score_limit).sum())}")
    env.close()


if __name__ == "__main__":
    main()
