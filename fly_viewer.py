"""Watch any game and the fly brain firing side by side, live in the browser.

The game frames come straight from the gymnasium environment. The brain is the same simulation that fly_gym.py
(or flappy_fly.py / flappy_readout.py) runs; here every spike of every 0.5 ms step is recorded and streamed to
viewer/index.html, where each neuron sits at its real FlyWire position.

Needs: pip install numpy "gymnasium[classic-control,toy-text,box2d]" flappy-bird-gymnasium pygame juliacall

Every trained player (pick the game in the page):  python fly_viewer.py
One player:                                        python fly_viewer.py --load players/lunarlander.npz
An untrained brain on any gymnasium game:          python fly_viewer.py --game MountainCarContinuous-v0
The same, learning by STDP while you watch:        python fly_viewer.py --game cartpole --learn
Flappy Bird brains from the older scripts:         python fly_viewer.py --load readout_rl_gap150.jls --pipe-gap 150

Neuron positions are read from coordinates_princeton.csv (FlyWire: root_id,position,supervoxel_id) next to
this file, or from --coords (download it from codex.flywire.ai). Without it the viewer still works, with a
schematic layout (and says so).
"""
import argparse
import base64
import csv
import functools
import io
import json
import os
import queue
import signal
import threading
import time
import webbrowser
from dataclasses import replace
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

os.environ.setdefault("SDL_VIDEODRIVER", "dummy")   # frames go to the browser, not to a game window
os.environ.setdefault("PYGAME_HIDE_SUPPORT_PROMPT", "1")

import numpy as np

import flappy_fly as ff
import flappy_readout as fr
import fly_gym as fy

ROOT = Path(__file__).resolve().parent
OTHER, INPUT, ACT = 0, 1, 2      # neuron roles; ACT + g pushes action group g
MAX_GROUPS = 5                   # more choices than this share a colour
SMELL = "AL_L,LH_L,MB_CA_L"      # what the older Flappy Bird brains (.jls) were trained on
REGIONS = {"AL": "antennal lobe", "LH": "lateral horn", "MB_CA": "mushroom body", "LAL": "LAL (premotor)"}
DEFAULT_FPS = 30
EXPLORE = 0.1                    # random actions while learning (--learn)
STEP_SECONDS = 1.6               # "One decision" plays this slowly
END_SECONDS = 0.9

VIEWER_HELPERS = r"""
# Same loop (and same random draws) as brain_step! / reservoir_counts! / act!, but keeps every spike.
function record_decision!(a, obs)
    b = a.brain
    ids, per_step = Int32[], zeros(Int32, a.steps)
    for t in 1:a.steps
        for (d, x) in enumerate(obs)
            pop = x > 0 ? a.inputs[2d-1] : a.inputs[2d]
            p_spike = a.max_hz * min(abs(x), 1) * b.p.dt / 1000
            for i in pop
                rand() < p_spike && (b.g[i] += a.input_mv)
            end
        end
        sp = step!(b)
        append!(ids, sp)
        per_step[t] = length(sp)
    end
    b.e .*= exp(-a.steps * b.p.dt / b.p.tau_e)
    return ids, per_step
end

function strongest_edges(a, targets, n)
    pres, posts = reduce(vcat, a.inputs), collect(Int, targets)
    I, J, V = findnz(a.brain.W[posts, pres])
    o = sortperm(V, rev=true)
    o = o[1:min(n, length(o))]
    return pres[J[o]], posts[I[o]], V[o]
end

# The simulated neuropil where each neuron has most of its synapses (needs one pass over the big csv, so cached).
function neuron_regions(path, ids, neuropils)
    cache = "$path.regions.$(hash((neuropils, length(ids)))).jls"
    isfile(cache) && return deserialize(cache)
    index = Dict(id => k for (k, id) in enumerate(ids))
    syn = [Dict{String,Int}() for _ in ids]
    for line in Iterators.drop(eachline(path), 1)
        p, q, neuropil, n, _ = split(line, ',')
        (neuropils === nothing || neuropil in neuropils) || continue
        for root in (p, q)
            k = get(index, parse(Int, root), 0)
            k > 0 && (syn[k][neuropil] = get(syn[k], neuropil, 0) + parse(Int, n))
        end
    end
    regions = [isempty(d) ? "" : argmax(d) for d in syn]
    serialize(cache, regions)
    return regions
end
"""


def b64(a, dtype):
    return base64.b64encode(np.asarray(a).astype(dtype).tobytes()).decode()


def action_groups(moves):
    """Which colour each choice gets: ([group of every choice, over all heads], [group name])."""
    flat = [name for head in moves for name in head]
    if len(flat) <= MAX_GROUPS:
        return list(range(len(flat))), flat
    if 1 < len(moves) <= MAX_GROUPS:
        return [d for d, head in enumerate(moves) for _ in head], [f"action {d}" for d in range(len(moves))]
    return [0] * len(flat), ["any action"]


def pushed_choice(w, slices):
    """For every read-out neuron: the choice its spikes make likelier (compared with the rest of its head), and by how much."""
    w = np.concatenate([w[s] - w[s].mean(0) for s in slices])
    best = w.argmax(0)
    return best, w[best, np.arange(w.shape[1])]


class Fly:
    """A brain, the game it plays and how its spikes become actions. decide() returns
    ([choice per head], [score of every choice], spiking neuron indices, spikes per 0.5 ms step)."""
    unit = "probability"             # what the scores are: probability, or spikes
    score_label = "Return"
    first_seed = fy.EVAL_SEED        # the same games fly_gym.py test plays
    learning = False

    def setup(self, title, kind, game, jl, agent, neuropils, members):
        """members: for every choice, the neurons that push it (strongest first)."""
        self.title, self.kind, self.game, self.jl, self.agent, self.neuropils = title, kind, game, jl, agent, neuropils
        jl.seval(VIEWER_HELPERS)
        self.moves = game.moves
        self.steps, self.dt = int(agent.steps), float(agent.brain.p.dt)
        self.N = len(agent.brain.v)
        self.sign = np.array(agent.brain.sign)
        self.inputs = [np.array(p) - 1 for p in agent.inputs]
        self.group_of, self.groups = action_groups(self.moves)
        self.members = [[] for _ in self.groups]
        for choice, ids in enumerate(members):
            self.members[self.group_of[choice]] += [int(i) for i in ids]
        self.role = np.zeros(self.N, dtype=np.uint8)
        for g, ids in enumerate(self.members):
            self.role[ids] = ACT + g
        self.role[np.concatenate(self.inputs)] = INPUT

    def reset(self):
        self.jl.reset_b(self.agent)

    def spikes(self, f):
        ids, per_step = self.jl.record_decision_b(self.agent, f)
        return np.array(ids, dtype=np.int64) - 1, np.array(per_step, dtype=np.int64)

    def score(self, total):
        """The number shown as the result of a game."""
        return total

    def learn(self, r, bonus, f, term, done):
        """Called after every game step with the reward and the next input."""

    def neuropil_list(self):
        return "nothing" if self.neuropils == "all" else json.dumps(self.neuropils.split(","))

    def root_ids(self):
        """FlyWire ids, or None if the saved agent was built from other neuropils."""
        c = self.jl.seval(f'load_connectome(raw"{ROOT / "connections_princeton.csv"}"; neuropils={self.neuropil_list()})')
        ids = np.array(c.ids)
        return ids if len(ids) == self.N else None

    def regions(self, ids):
        fn = self.jl.seval(f'ids -> neuron_regions(raw"{ROOT / "connections_princeton.csv"}", '
                           f'Int.(collect(ids)), {self.neuropil_list()})')
        return [str(r) for r in fn(ids)]

    def edges(self, n=1500):
        pre, post, w = self.jl.strongest_edges(self.agent, np.concatenate(self.members).astype(int) + 1, n)
        return np.array(pre) - 1, np.array(post) - 1, np.array(w)


class GymFly(Fly):
    """A player from fly_gym.py (any game), or an untrained brain on that game."""

    def __init__(self, args, path=None):
        if path:
            self.game, self.player, self.policy = fy.read_player(path, "rgb_array")
            if isinstance(self.player, fy.NoBrain):
                raise ValueError(f"{path} was trained with --no-brain, so there is no brain to watch")
            with np.load(path, allow_pickle=False) as z:
                neuropils = json.loads(str(z["config"]))["neuropils"]
            title = Path(path).stem
        else:
            self.game = game = fy.make_game(args.game, "rgb_array")
            if game.needs_calibration:
                game.calibrate(seed=args.seed)
            neuropils = args.neuropils or fy.NEUROPILS
            self.player = fy.FlyBrainPlayer(game.n_obs, sum(game.heads), dict(
                neuropils=neuropils, pop=args.pop, K=args.K or 500, action_ms=40, seed=args.seed, stdp_lr=args.lr))
            self.policy = dict(voter=fy.Voter(game.heads))
            title = f"{args.game} (untrained)"
        game, player = self.game, self.player
        self.rng = np.random.default_rng(12345)
        self.read = player.ids - 1
        if args.pipe_gap and hasattr(game.env.unwrapped, "_pipe_gap"):
            game.env.unwrapped._pipe_gap = args.pipe_gap
        if "ro" in self.policy:
            ro = self.policy["ro"]
            choice, strength = pushed_choice(ro.w[:, :-1], ro.slices)
            order = np.argsort(-strength)
            members = [[self.read[k] for k in order if choice[k] == c] for c in range(ro.w.shape[0])]
            kind = "frozen connectome + trained readout"
        else:
            members = [np.array(p) - 1 for p in player.agent.outputs]
            self.unit = "spikes above its average"
            self.learning = args.learn
            if self.learning:
                critic = fy.Readout(player.K, game.heads, *fy.warm_up(game, player, args.seed))
                self.dopamine, self.last = fy.Dopamine(critic, args.gamma, args.critic_lr), None
                player.agent.brain.learning = True
            kind = "STDP, learning" if self.learning else "STDP, frozen" if path else "untrained connectome"
        self.setup(title, kind, game, player.jl, player.agent, neuropils, members)
        self.action_of = np.array(player.agent.action_of)
        if hasattr(game.env.unwrapped, "_pipe_gap"):
            self.score_label = "Pipes"

    def score(self, total):
        return int(self.game.info["score"]) if self.score_label == "Pipes" else total

    def count(self, ids):
        """The two things FlyBrainPlayer.step returns, from a list of spikes."""
        out = np.bincount(self.action_of[ids], minlength=sum(self.game.heads) + 1)[1:]
        return np.bincount(ids, minlength=self.N)[self.read].astype(float), out.astype(float)

    def reset(self):
        super().reset()
        self.last = None

    def decide(self, f):
        ids, per_step = self.spikes(f)
        counts, out = self.count(ids)
        if "ro" in self.policy:
            ro = self.policy["ro"]
            x = ro.x(counts)
            return ro.choose(x, self.rng, greedy=ro.greedy), np.exp(ro.logp(x)), ids, per_step
        voter = self.policy["voter"]
        scores = out - voter.base
        if self.learning:
            self.reinforce(counts, False, False)
        choices = voter.choose(out, self.rng, EXPLORE if self.learning else 0.0)
        if self.learning:
            self.last = [self.dopamine.critic.x(counts), choices, 0.0]
        return choices, scores, ids, per_step

    def learn(self, r, bonus, f, term, done):
        if not self.learning:
            return
        self.last[2] = r + bonus
        if done:                     # the last state's spikes are needed too (as in train_stdp), but not shown
            self.reinforce(self.count(self.spikes(f)[0])[0], term, True)
            self.last = None

    def reinforce(self, counts, term, done):
        """Dopamine for the previous decision, now that the brain has seen what came of it."""
        if self.last is None:
            return
        x, choices, r = self.last
        d = self.dopamine(x, r, self.dopamine.critic.x(counts), term, done)
        for s, choice in zip(self.policy["voter"].slices, choices):
            self.jl.reward_chosen_b(self.agent, d, s.start + choice)


class FlappyFly(Fly):
    """A Flappy Bird brain saved by flappy_readout.py (a readout) or flappy_fly.py (an STDP agent)."""
    score_label = "Pipes"
    first_seed = 9_000_000           # the same games flappy_readout.py test plays

    def __init__(self, args, path):
        from fly_games import Game
        game = Game("flappy", "rgb_array", make=dict(pipe_gap=args.pipe_gap or 100, score_limit=args.score_limit))
        game.preset = replace(game.preset, repeat=args.frame_skip)
        game.limit = None            # flappy_readout.py plays until the bird dies or reaches --score-limit
        neuropils = args.neuropils or SMELL
        res = fr.Reservoir(args.K or 300, args.seed, neuropils, args.pop)      # also starts Julia
        jl, self.ro, agent = res.jl, None, res.agent
        self.args = args
        saved = jl.deserialize(str(Path(path).resolve()))
        if jl.isa(saved, jl.Agent):
            agent = saved
            self.learning = args.learn
            kind = "STDP, learning" if args.learn else "STDP, frozen"
            members = [np.array(p) - 1 for p in agent.outputs]                 # glide, flap
            self.unit = "spikes" if args.decide == "max" else "spikes above its average"
        else:
            fr.RES_BOOT = res
            self.ro, K, seed = fr.load(path)
            if (K, seed) != (args.K or 300, args.seed):
                res = fr.Reservoir(K, seed, neuropils, args.pop)
            agent, kind = res.agent, "frozen connectome + trained readout"
            self.pos = pos = np.array(res.pos)                                 # 0 = not read out, else 1..K
            ids = np.empty(self.ro.w.size - 1, dtype=int)
            ids[pos[pos > 0] - 1] = np.flatnonzero(pos)
            order = np.argsort(-np.abs(self.ro.w[:-1]))
            members = [[ids[k] for k in order if self.ro.w[k] <= 0], [ids[k] for k in order if self.ro.w[k] > 0]]
        agent.brain.learning = self.learning
        self.rng = np.random.default_rng(args.seed)
        self.base, self.bias = np.zeros(2), float(args.flap_bias)
        self.setup(Path(path).stem, kind, game, jl, agent, neuropils, members)

    def score(self, total):
        return int(self.game.info["score"])

    def decide(self, f):
        ids, per_step = self.spikes(f)
        if self.ro is not None:
            k = self.pos[ids]
            counts = np.bincount(k[k > 0] - 1, minlength=self.ro.w.size - 1).astype(float)
            p = float(self.ro.p(self.ro.x(counts)))
            return [int(p > 0.5)], [1 - p, p], ids, per_step
        fired = np.bincount(ids, minlength=self.N)
        counts = np.array([fired[m].sum() for m in self.members], dtype=float)
        a = self.args
        if a.decide == "max":
            scores, free = counts, True
        else:                                           # same rule as act_rel! (options B and B2)
            scores = counts - self.base
            self.base = self.base + a.base_alpha * scores
            free = self.bias == 0 and a.flap_target <= 0
            scores = scores - [0, self.bias]            # flap has to win by this many spikes
        diff = scores[1] - scores[0]
        action = int(self.rng.integers(2)) if free and diff == 0 else int(diff > 0)
        if a.decide != "max" and a.flap_target > 0:
            self.bias += 0.1 * (action - a.flap_target)
        return [action], scores, ids, per_step

    def learn(self, r, bonus, f, term, done):
        if self.learning:
            self.jl.reward_b(self.agent, r)


def open_fly(args, path):
    """path: a saved player (.npz or .jls), or None for an untrained brain on args.game."""
    return FlappyFly(args, path) if path and Path(path).suffix == ".jls" else GymFly(args, path)


def find_players(args):
    """{name shown in the page: path}. Players without a brain are left out."""
    found = {}
    for p in sorted(Path(args.players).glob("*.npz")):
        try:
            with np.load(p, allow_pickle=False) as z:
                if not json.loads(str(z["config"]))["no_brain"]:
                    found[p.stem] = p
        except (OSError, ValueError, KeyError):
            print(f"{p} is not a player saved by fly_gym.py, skipped")
    return found


@functools.lru_cache(maxsize=1)
def read_positions(path):
    pos = {}
    with open(path, newline="") as fh:
        for row in csv.DictReader(fh):
            pos.setdefault(int(row["root_id"]), row["position"])           # several markers per neuron: first one
    return {k: [float(v) for v in p.strip("[] ").split()] for k, p in pos.items()}


def layout(fly, ids, coords, max_backdrop=45_000):
    """Positions scaled to about [-1, 1]: (neurons, rest-of-brain backdrop, 'flywire' or 'schematic')."""
    rng = np.random.default_rng(0)
    if ids is not None and coords.exists():
        xyz = read_positions(coords)
        brain = np.array(list(xyz.values()))
        lo, hi = np.percentile(brain, [0.5, 99.5], axis=0)
        # x right, y up, z toward the viewer (a rotation of FlyWire space, not a mirror image)
        place = lambda p: (np.asarray(p) - (lo + hi) / 2) / ((hi - lo)[0] / 2) * [1, -1, -1]
        missing = [k for k in ids if int(k) not in xyz]
        if not missing:
            ours = set(int(k) for k in ids)
            rest = np.array([p for k, p in xyz.items() if k not in ours])
            rest = rest[rng.permutation(len(rest))[:max_backdrop]]
            return place([xyz[int(k)] for k in ids]), place(rest), "flywire"
        print(f"{len(missing)} neurons have no position in {coords.name}; using the schematic layout")
    # inputs left, everything else in the middle, one clump per action group on the right
    centre, spread = np.zeros((ACT + len(fly.groups), 2)), np.full(ACT + len(fly.groups), 0.1)
    centre[INPUT], spread[INPUT], spread[OTHER] = (-0.75, 0), 0.12, 0.3
    centre[ACT:, 0], centre[ACT:, 1] = 0.75, np.linspace(0.45, -0.45, len(fly.groups) + 2)[1:-1]
    p = np.column_stack([centre[fly.role], np.zeros(fly.N)]) + rng.normal(size=(fly.N, 3)) * spread[fly.role][:, None]
    return p, np.zeros((0, 3)), "schematic"


class Hub:
    """Connected browsers and the controls they share."""

    def __init__(self, players=()):
        self.lock = threading.Condition()
        self.clients = []
        self.playing, self.steps_due, self.speed = True, 0, 1.0
        self.gap, self.restart = None, False           # gap: only for games with pipes
        self.players, self.current, self.wanted, self.meta = list(players), None, None, b"{}"

    def send(self, msg):
        data = json.dumps(msg)
        with self.lock:
            for q in self.clients:
                if q.full():                       # a stalled tab loses old ticks instead of growing forever
                    try:
                        q.get_nowait()
                    except queue.Empty:
                        pass
                q.put_nowait(data)

    def state(self):
        return dict(type="state", playing=self.playing, speed=self.speed, gap=self.gap)

    def control(self, cmd, value):
        with self.lock:
            if cmd == "play":
                self.playing = bool(value)
            elif cmd == "step":
                self.playing, self.steps_due = False, min(self.steps_due + 1, 3)
            elif cmd == "speed":
                self.speed = min(max(float(value), 0.1), 4.0)
            elif cmd == "gap":
                if self.gap is not None:
                    self.gap = int(min(max(float(value), 80), 220))
            elif cmd == "player":
                if value not in self.players:
                    raise ValueError(value)
                self.wanted = value
            elif cmd == "restart":
                self.restart = True
                if not self.playing:
                    self.steps_due = max(self.steps_due, 1)
            else:
                raise ValueError(cmd)
            self.lock.notify_all()
        self.send(self.state())

    def wait_turn(self):
        """Block until someone is watching and something should happen. Returns "switch", "step" or "play"."""
        with self.lock:
            while not (self.wanted or (self.clients and (self.playing or self.steps_due))):
                self.lock.wait(0.25)               # short timeout so Ctrl+C is noticed
            if self.wanted:
                return "switch"
            if self.playing:
                return "play"
            self.steps_due -= 1
            return "step"

    def pause_for(self, seconds):
        end = time.monotonic() + seconds
        while (left := end - time.monotonic()) > 0 and not self.wanted:
            time.sleep(min(left, 0.05))


def make_handler(hub):
    class Handler(BaseHTTPRequestHandler):
        protocol_version = "HTTP/1.1"

        def log_message(self, *a):
            pass

        def local(self):
            """Only answer pages opened on this computer (blocks DNS-rebinding and cross-site posts)."""
            host = self.headers.get("Host", "").rsplit(":", 1)[0]
            origin = self.headers.get("Origin")
            ok = host in ("localhost", "127.0.0.1") and (
                origin is None or origin.split("://")[-1] == self.headers.get("Host"))
            if not ok:
                self.reply(403, b"forbidden", "text/plain")
            return ok

        def reply(self, code, body, kind):
            self.send_response(code)
            self.send_header("Content-Type", kind)
            self.send_header("Content-Length", str(len(body)))
            self.send_header("Cache-Control", "no-store")
            self.end_headers()
            self.wfile.write(body)

        def do_GET(self):
            if not self.local():
                return
            if self.path == "/":
                self.reply(200, (ROOT / "viewer" / "index.html").read_bytes(), "text/html; charset=utf-8")
            elif self.path == "/meta.json":
                self.reply(200, hub.meta, "application/json")
            elif self.path == "/events":
                self.stream()
            else:
                self.reply(404, b"not found", "text/plain")

        def stream(self):
            self.send_response(200)
            self.send_header("Content-Type", "text/event-stream")
            self.send_header("Cache-Control", "no-store")
            self.send_header("Connection", "close")
            self.end_headers()
            self.close_connection = True
            q = queue.Queue(maxsize=12)
            with hub.lock:
                q.put_nowait(json.dumps(hub.state()))
                hub.clients.append(q)
                hub.lock.notify_all()
            try:
                while True:
                    try:
                        line = "data: " + q.get(timeout=5) + "\n\n"
                    except queue.Empty:
                        line = ": still here\n\n"
                    self.wfile.write(line.encode())
                    self.wfile.flush()
            except OSError:
                pass
            finally:
                with hub.lock:
                    hub.clients.remove(q)

        def do_POST(self):
            if not self.local():
                return
            body = self.rfile.read(int(self.headers.get("Content-Length") or 0))
            try:
                if self.path != "/control":
                    raise ValueError(self.path)
                msg = json.loads(body)
                hub.control(msg["cmd"], msg.get("value"))
            except (ValueError, KeyError, TypeError, AttributeError):
                return self.reply(400, b"bad request", "text/plain")
            self.reply(200, b"{}", "application/json")

    return Handler


class Show:
    """Plays one fly's games one decision at a time (same loop as fly_gym.play_frozen), plus frames."""

    def __init__(self, fly, hub, first_seed=None):
        self.fly, self.game, self.hub = fly, fly.game, hub
        self.seed = fly.first_seed if first_seed is None else first_seed
        self.games, self.scores, self.f = 0, [], None
        env = self.game.env
        self.fps = env.metadata.get("render_fps") or DEFAULT_FPS
        self.draws = env.render_mode == "rgb_array"
        self.frames = []
        self.game.on_frame = lambda: self.frames.append(self.frame())
        # Never faster than the brain time being shown; games that decide every 1/50 s run at half speed
        self.seconds = max(self.game.preset.repeat / self.fps, fly.steps * fly.dt / 1000)

    def frame(self):
        if not self.draws:
            return None
        return png(self.game.env.render())

    def tick(self):
        game, fly, start = self.game, self.fly, None
        if self.f is None:
            if self.hub.gap is not None:
                game.env.unwrapped._pipe_gap = self.hub.gap
            self.f = game.reset(self.seed + self.games)
            fly.reset()
            start, self.total, self.decisions = self.frame(), 0.0, 0
            self.games += 1
        f, self.frames = self.f, []
        choices, scores, ids, per_step = fly.decide(f)
        self.f, reward, bonus, term, trunc = game.step(choices)
        done = bool(term or trunc)
        fly.learn(reward, bonus, self.f, term, done)
        self.total += reward
        self.decisions += 1
        score = fly.score(self.total)
        if done:
            self.scores.append(score)
            self.f = None
        return dict(type="tick", start=start, frames=[p for p in self.frames if p], spikes=b64(ids, "<u4"),
                    per_step=per_step.tolist(), features=[round(float(v), 4) for v in f], choices=choices,
                    scores=[round(float(s), 4) for s in scores], reward=reward, score=score, done=done,
                    timeout=bool(trunc and not term), game=self.games, decision=self.decisions,
                    seconds=self.seconds * (len(self.frames) / game.preset.repeat), best=max(self.scores, default=None),
                    mean=float(np.mean(self.scores)) if self.scores else None)


def png(frame):
    import pygame
    h, w, _ = frame.shape
    out = io.BytesIO()
    pygame.image.save(pygame.image.frombuffer(np.ascontiguousarray(frame).tobytes(), (w, h), "RGB"), out, "f.png")
    return base64.b64encode(out.getvalue()).decode()


def raster_rows(fly, families):
    """Which neurons the raster shows, with a label per block: inputs, action groups, then a sample of every region."""
    each = max(2, min(20, 160 // len(fly.inputs)))
    names = [n for neg, pos in fly.game.senses for n in (pos, neg)]          # inputs[2d] fires for x > 0
    rows = [dict(label=n, role=INPUT, neurons=p[:each].tolist()) for n, p in zip(names, fly.inputs) if n]
    if len(rows) > 10:
        rows = [dict(label=f"{len(rows)} senses", role=INPUT, neurons=sum((r["neurons"] for r in rows), []))]
    each = min(60, 150 // len(fly.groups))
    rows += [dict(label=name, role=ACT + g, neurons=ids[:each])
             for g, (name, ids) in enumerate(zip(fly.groups, fly.members)) if ids]
    others, rng = fly.role == OTHER, np.random.default_rng(0)
    blocks = sorted(set(families) - {""}) if families is not None else []
    if not 1 <= len(blocks) <= 5:
        blocks = [None]
    for b in blocks:
        ids = np.flatnonzero(others if b is None else others & (families == b))
        if len(ids):
            pick = rng.choice(ids, size=min(100 // len(blocks), len(ids)), replace=False)
            rows.append(dict(label="others" if b is None else REGIONS.get(b, b), role=OTHER, neurons=pick.tolist()))
    return rows


def build_meta(args, fly, show, hub):
    ids = fly.root_ids()
    pos, rest, kind = layout(fly, ids, Path(args.coords))
    families = regions = None
    if ids is not None:
        families = np.array([r.removesuffix("_L").removesuffix("_R") for r in fly.regions(ids)])
        if len(set(families)) <= 6:
            regions = [dict(name=REGIONS.get(b, b), neurons=int((families == b).sum()),
                            inputs=int(((families == b) & (fly.role == INPUT)).sum()),
                            outputs=int(((families == b) & (fly.role >= ACT)).sum()))
                       for b in sorted(set(families) - {""})]
    pre, post, w = fly.edges()
    synapses = len(fly.agent.brain.W.nzval)
    game = fly.game
    print(f"{fly.title}: {game.preset.env_id}, {fly.N} neurons, {synapses} synapses, {kind} layout, {fly.kind}", flush=True)
    return json.dumps(dict(
        n=fly.N, synapses=synapses, layout=kind, title=fly.title, kind=fly.kind, env=game.preset.env_id,
        readout=fly.unit == "probability", unit=fly.unit, score_label=fly.score_label, steps=fly.steps, dt=fly.dt,
        seconds=show.seconds, realtime=game.preset.repeat / show.fps, draws=show.draws, senses=game.senses,
        moves=fly.moves, groups=fly.groups, group_of=fly.group_of, regions=regions,
        players=hub.players, current=hub.current,
        pos=b64(pos, "<f4"), rest=b64(rest, "<f4"), role=b64(fly.role, "u1"), sign=b64(fly.sign, "i1"),
        ids=[str(i) for i in ids] if ids is not None else None,      # 64-bit ids: strings, JS would round them
        families=families.tolist() if families is not None else None,
        edges=b64(np.column_stack([pre, post]), "<u4"), edge_w=b64(w, "<f4"), rows=raster_rows(fly, families))).encode()


def start_show(args, hub, name, path):
    fly = open_fly(args, path)
    show = Show(fly, hub, args.game_seed)
    unwrapped = fly.game.env.unwrapped
    hub.current, hub.gap = name, int(unwrapped._pipe_gap) if hasattr(unwrapped, "_pipe_gap") else None
    hub.meta = build_meta(args, fly, show, hub)
    return show


def run(args, hub, show, players):
    while True:
        turn = hub.wait_turn()
        t0 = time.monotonic()
        if turn == "switch":
            with hub.lock:
                name, hub.wanted = hub.wanted, None
            if name != hub.current:
                hub.send(dict(type="loading", name=name))
                try:
                    old, show = show, start_show(args, hub, name, players[name])
                    old.game.close()
                except (ValueError, ImportError, OSError) as e:
                    print(f"could not open {name}: {e}", flush=True)
                    hub.send(dict(type="failed", name=name, text=str(e)))
                    continue
                hub.restart = False
                hub.send(dict(type="reload"))
            continue
        if hub.restart:
            hub.restart, show.f = False, None
        msg = show.tick()
        msg["seconds"] = STEP_SECONDS if turn == "step" else msg["seconds"] / hub.speed
        hub.send(msg)
        wait = msg["seconds"] + (END_SECONDS if msg["done"] and turn == "play" else 0)
        hub.pause_for(wait - (time.monotonic() - t0))


def parse_args(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--load", help="a player saved by fly_gym.py (.npz), or a Flappy Bird brain from flappy_readout.py "
                    "or flappy_fly.py (.jls). Default: the first player in --players")
    ap.add_argument("--game", help="watch an untrained brain play this game (a fly_gym.py preset or any gymnasium id)")
    ap.add_argument("--players", default=str(ROOT / "players"), help="folder of players to offer in the page")
    ap.add_argument("--learn", action="store_true", help="STDP brains only: keep learning from reward while you watch")
    ap.add_argument("--lr", type=float, default=0.2, help="--game --learn: synapse learning rate")
    ap.add_argument("--critic-lr", type=float, default=0.05, help="--learn: learning rate of the reward prediction")
    ap.add_argument("--gamma", type=float, default=0.99, help="--learn: discount per decision")
    ap.add_argument("--coords", default=str(ROOT / "coordinates_princeton.csv"), help="FlyWire neuron positions")
    ap.add_argument("--game-seed", type=int, help="first game's seed (default: the games the test commands play)")
    ap.add_argument("--neuropils", help=f"--game: default {fy.NEUROPILS}. Flappy .jls files: must match training, "
                    f"default {SMELL}. Players (.npz) remember their own")
    ap.add_argument("--pop", type=int, default=20, help="--game and .jls: neurons per input/output population")
    ap.add_argument("--K", type=int, help="--game and .jls: neurons read out (default 500; .jls 300)")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--pipe-gap", type=int, help="Flappy Bird's gap (default: as trained; .jls 100)")
    ap.add_argument("--frame-skip", type=int, default=3, help=".jls only: game frames per brain decision")
    ap.add_argument("--score-limit", type=int, default=100, help=".jls only")
    ap.add_argument("--decide", choices=["max", "baseline"], default="max", help=".jls STDP agents: see flappy_fly.py")
    ap.add_argument("--base-alpha", type=float, default=0.02)
    ap.add_argument("--flap-bias", type=float, default=0.0)
    ap.add_argument("--flap-target", type=float, default=0.0)
    ap.add_argument("--port", type=int, default=8303)
    ap.add_argument("--no-open", action="store_true", help="don't open the browser")
    args = ap.parse_args(argv)
    if args.load and args.game:
        ap.error("use --load (a saved brain) or --game (an untrained one), not both")
    if args.load and not Path(args.load).exists():
        ap.error(f"'{args.load}' not found. Train one with: python fly_gym.py train GAME --save {args.load}")
    if args.load and Path(args.load).suffix not in (".npz", ".jls"):
        ap.error("--load needs a .npz (fly_gym.py) or .jls (flappy_readout.py, flappy_fly.py) file")
    return args


def main():
    args = parse_args()
    np.random.seed(args.seed)
    players = find_players(args)
    if args.load:
        name = next((n for n, p in players.items() if p.resolve() == Path(args.load).resolve()), None)
        if name is None:
            name = Path(args.load).stem if Path(args.load).stem not in players else Path(args.load).name
            players = {name: Path(args.load), **players}
    elif args.game:
        name = f"{args.game} (untrained)"
        players = {name: None, **players}
    elif players:
        name = next(iter(players))
    else:
        raise SystemExit(f"no players in {args.players}. Use --game cartpole, or train one with fly_gym.py")
    hub = Hub(players)
    print("starting Julia and loading the connectome (about 10 s, longer the first time)...", flush=True)
    try:
        show = start_show(args, hub, name, players[name])
    except (ValueError, ImportError) as e:
        raise SystemExit(f"error: {e}")
    # Julia takes over SIGINT and dumps a trace; nothing needs saving here, so just leave
    signal.signal(signal.SIGINT, lambda *_: os._exit(0))
    if args.learn and not show.fly.learning:
        print("--learn ignored: a readout brain's synapses are frozen")
    server = ThreadingHTTPServer(("127.0.0.1", args.port), make_handler(hub))
    server.daemon_threads = True
    threading.Thread(target=server.serve_forever, daemon=True).start()
    url = f"http://localhost:{args.port}/"
    print(f"open {url}   (Ctrl+C to stop)", flush=True)
    if not args.no_open:
        webbrowser.open(url)
    run(args, hub, show, players)                      # Julia has to stay on the main thread


if __name__ == "__main__":
    main()
