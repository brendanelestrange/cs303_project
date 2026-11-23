"""Watch the game and the fly brain firing side by side, live in the browser.

The game frames come straight from the gymnasium environment. The brain is the same simulation that
flappy_fly.py / flappy_readout.py run; here every spike of every 0.5 ms step is recorded and streamed
to viewer/index.html, where each neuron sits at its real FlyWire position.

Needs: pip install numpy gymnasium flappy-bird-gymnasium pygame juliacall

Watch a trained readout:   python fly_viewer.py --load readout_rl_gap150.jls --pipe-gap 150
Watch a trained STDP fly:  python fly_viewer.py --load fly_agent.jls
Watch an untrained fly learn by reward-modulated STDP:  python fly_viewer.py --learn

Neuron positions are read from coordinates_princeton.csv (FlyWire: root_id,position,supervoxel_id) next to
this file, or from --coords (download it from codex.flywire.ai). Without it the viewer still works, with a schematic layout (and says so).
"""
import argparse
import base64
import csv
import io
import json
import os
import queue
import signal
import threading
import time
import webbrowser
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

os.environ.setdefault("SDL_VIDEODRIVER", "dummy")   # frames go to the browser, not to a game window
os.environ.setdefault("PYGAME_HIDE_SUPPORT_PROMPT", "1")

import numpy as np

import flappy_fly as ff
import flappy_readout as fr

ROOT = Path(__file__).resolve().parent
OTHER, INPUT, FLAP, GLIDE = 0, 1, 2, 3
# inputs[2d-1] fire when feature d is > 0, inputs[2d] when it is < 0 (see flappy_fly.features)
INPUT_NAMES = ["below gap", "above gap", "falling", "rising",
               "pipe close", "pipe far", "low on screen", "high on screen"]
GAME_FPS = 30
STEP_SECONDS = 1.6      # "step" plays one decision this slowly
CRASH_SECONDS = 0.9

VIEWER_HELPERS = r"""
# Same loop (and same random draws) as reservoir_counts! / act!, but keeps every spike.
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
"""


def b64(a, dtype):
    return base64.b64encode(np.asarray(a).astype(dtype).tobytes()).decode()


class Fly:
    """The brain plus its decision rule: a trained readout, or the STDP agent's two output groups."""

    def __init__(self, args):
        self.args = args
        res = fr.Reservoir(args.K, args.seed, args.neuropils, args.pop)      # also starts Julia
        self.jl = jl = res.jl
        jl.seval(VIEWER_HELPERS)
        self.ro = None
        self.agent = res.agent
        if args.load:
            saved = jl.deserialize(str(Path(args.load).resolve()))
            if jl.isa(saved, jl.Agent):
                self.agent = saved
            else:
                fr.RES_BOOT = res
                self.ro, K, seed = fr.load(args.load)
                if (K, seed) != (args.K, args.seed):
                    res = fr.Reservoir(K, seed, args.neuropils, args.pop)
                self.agent = res.agent
        elif args.learn:
            jl.set_lr_b(self.agent, args.lr)
        self.learning = bool(args.learn and self.ro is None)
        self.agent.brain.learning = self.learning
        self.steps = int(self.agent.steps)
        self.dt = float(self.agent.brain.p.dt)
        self.N = len(self.agent.brain.v)
        self.sign = np.array(self.agent.brain.sign)
        self.inputs = [np.array(p) - 1 for p in self.agent.inputs]
        self.rng = np.random.default_rng(args.seed)
        self.base, self.bias = np.zeros(2), float(args.flap_bias)

        self.role = np.zeros(self.N, dtype=np.uint8)
        if self.ro is not None:
            pos = np.array(res.pos)                   # 0 = not read out, else 1..K
            self.pos = pos
            ids = np.empty(self.ro.w.size - 1, dtype=int)
            ids[pos[pos > 0] - 1] = np.flatnonzero(pos)
            order = np.argsort(-np.abs(self.ro.w[:-1]))
            self.flap_ids = [int(ids[k]) for k in order if self.ro.w[k] > 0]
            self.glide_ids = [int(ids[k]) for k in order if self.ro.w[k] <= 0]
        else:
            glide, flap = (np.array(p) - 1 for p in self.agent.outputs)
            self.flap_ids, self.glide_ids = flap.tolist(), glide.tolist()
        self.role[self.flap_ids] = FLAP
        self.role[self.glide_ids] = GLIDE
        self.role[np.concatenate(self.inputs)] = INPUT

    def reset(self):
        self.jl.reset_b(self.agent)

    def decide(self, f):
        """One decision: (action, evidence for flap, threshold, spiking neuron indices, spikes per step)."""
        ids, per_step = self.jl.record_decision_b(self.agent, f)
        ids, per_step = np.array(ids, dtype=np.int64) - 1, np.array(per_step, dtype=np.int64)
        if self.ro is not None:
            k = self.pos[ids]
            counts = np.bincount(k[k > 0] - 1, minlength=self.ro.w.size - 1).astype(float)
            p = float(self.ro.p(self.ro.x(counts)))
            return int(p > 0.5), p, 0.5, ids, per_step
        fired = np.bincount(ids, minlength=self.N)
        counts = np.array([fired[self.glide_ids].sum(), fired[self.flap_ids].sum()], dtype=float)
        a = self.args
        if a.decide == "max":
            diff = counts[1] - counts[0]
            action = int(diff > 0) if diff else int(self.rng.integers(2))
            return action, float(diff), 0.0, ids, per_step
        rel = counts - self.base                        # same rule as act_rel! (options B and B2)
        self.base += a.base_alpha * (counts - self.base)
        diff, th = float(rel[1] - rel[0]), self.bias
        if th == 0 and a.flap_target <= 0:
            action = int(diff > 0) if diff else int(self.rng.integers(2))
        else:
            action = int(diff > th)
            if a.flap_target > 0:
                self.bias += 0.1 * (action - a.flap_target)
        return action, diff, th, ids, per_step

    def reward(self, r):
        if self.learning:
            self.jl.reward_b(self.agent, r)

    def edges(self, n=1500):
        pre, post, w = self.jl.strongest_edges(self.agent, np.array(self.flap_ids + self.glide_ids) + 1, n)
        return np.array(pre) - 1, np.array(post) - 1, np.array(w)

    def root_ids(self):
        np_list = "nothing" if self.args.neuropils == "all" else json.dumps(self.args.neuropils.split(","))
        c = self.jl.seval(f'load_connectome(raw"{ROOT / "connections_princeton.csv"}"; neuropils={np_list})')
        return np.array(c.ids)


def layout(fly, coords, max_backdrop=45_000):
    """Positions scaled to about [-1, 1]: (neurons, rest-of-brain backdrop, root ids, 'flywire' or 'schematic')."""
    ids = fly.root_ids()
    ids = ids if len(ids) == fly.N else None       # a saved agent built from other neuropils
    rng = np.random.default_rng(0)
    if ids is not None and coords.exists():
        pos = {}
        with open(coords, newline="") as fh:
            for row in csv.DictReader(fh):
                pos.setdefault(int(row["root_id"]), row["position"])   # several markers per neuron: first one
        xyz = {k: [float(v) for v in p.strip("[] ").split()] for k, p in pos.items()}
        brain = np.array(list(xyz.values()))
        lo, hi = np.percentile(brain, [0.5, 99.5], axis=0)
        # x right, y up, z toward the viewer (a rotation of FlyWire space, not a mirror image)
        place = lambda p: (np.asarray(p) - (lo + hi) / 2) / ((hi - lo)[0] / 2) * [1, -1, -1]
        missing = [k for k in ids if int(k) not in xyz]
        if not missing:
            ours = set(int(k) for k in ids)
            rest = np.array([p for k, p in xyz.items() if k not in ours])
            rest = rest[rng.permutation(len(rest))[:max_backdrop]]
            return place([xyz[int(k)] for k in ids]), place(rest), ids, "flywire"
        print(f"{len(missing)} neurons have no position in {coords.name}; using the schematic layout")
    centre = {INPUT: (-0.75, 0.0), OTHER: (0.0, 0.0), FLAP: (0.75, 0.3), GLIDE: (0.75, -0.3)}
    spread = {INPUT: 0.12, OTHER: 0.3, FLAP: 0.1, GLIDE: 0.1}
    xy = np.array([centre[r] for r in fly.role])
    p = np.column_stack([xy, np.zeros(fly.N)]) + rng.normal(size=(fly.N, 3)) * np.array(
        [spread[r] for r in fly.role])[:, None]
    return p, np.zeros((0, 3)), ids, "schematic"


class Hub:
    """Connected browsers and the controls they share."""

    def __init__(self, gap):
        self.lock = threading.Condition()
        self.clients = []
        self.playing, self.steps_due, self.speed = True, 0, 1.0
        self.gap, self.restart = gap, False

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
                self.gap = int(min(max(float(value), 80), 220))
            elif cmd == "restart":
                self.restart = True
                if not self.playing:
                    self.steps_due = max(self.steps_due, 1)
            else:
                raise ValueError(cmd)
            self.lock.notify_all()
        self.send(self.state())

    def wait_turn(self):
        """Block until someone is watching and the game should move. Returns True for a single step."""
        with self.lock:
            while not (self.clients and (self.playing or self.steps_due)):
                self.lock.wait(0.25)               # short timeout so Ctrl+C is noticed
            if self.playing:
                return False
            self.steps_due -= 1
            return True

    def pause_for(self, seconds):
        end = time.monotonic() + seconds
        while (left := end - time.monotonic()) > 0:
            time.sleep(min(left, 0.05))


def make_handler(hub, meta):
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
                self.reply(200, meta, "application/json")
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


def png(frame):
    import pygame
    h, w, _ = frame.shape
    out = io.BytesIO()
    pygame.image.save(pygame.image.frombuffer(np.ascontiguousarray(frame).tobytes(), (w, h), "RGB"), out, "f.png")
    return base64.b64encode(out.getvalue()).decode()


class Game:
    """Plays games one decision at a time. Same loop as flappy_readout.play, plus frames."""

    def __init__(self, args, fly, hub):
        import gymnasium
        import flappy_bird_gymnasium  # noqa: F401  (registers the env)
        self.env = gymnasium.make("FlappyBird-v0", render_mode="rgb_array", use_lidar=False,
                                  pipe_gap=args.pipe_gap, score_limit=args.score_limit)
        self.args, self.fly, self.hub = args, fly, hub
        self.seed, self.game, self.scores, self.obs = args.game_seed, 0, [], None

    def tick(self, render=True):
        if self.obs is None:
            self.env.unwrapped._pipe_gap = self.hub.gap
            self.obs, _ = self.env.reset(seed=self.seed + self.game)
            self.fly.reset()
            self.game += 1
        f = ff.features(self.obs)
        action, evidence, threshold, ids, per_step = self.fly.decide(f)
        frames, reward, done, info = [], 0.0, False, {}
        for k in range(self.args.frame_skip):
            self.obs, r, term, trunc, info = self.env.step(action if k == 0 else 0)
            reward += r
            done = term or trunc
            if render:
                frames.append(png(self.env.render()))
            if done:
                break
        self.fly.reward(reward)
        if done:
            self.scores.append(int(info["score"]))
            self.obs = None
        return dict(type="tick", frames=frames, spikes=b64(ids, "<u4"), per_step=per_step.tolist(),
                    features=[round(float(v), 4) for v in f], action=action, evidence=evidence,
                    threshold=threshold, reward=reward, score=int(info["score"]), done=done, game=self.game,
                    games=len(self.scores), best=max(self.scores, default=0),
                    mean=float(np.mean(self.scores)) if self.scores else None)

    def run(self):
        hub = self.hub
        while True:
            single = hub.wait_turn()
            t0 = time.monotonic()
            if hub.restart:
                hub.restart, self.obs = False, None
            msg = self.tick()
            speed = hub.speed
            msg["seconds"] = STEP_SECONDS if single else len(msg["frames"]) / GAME_FPS / speed
            hub.send(msg)
            wait = msg["seconds"] + (CRASH_SECONDS if msg["done"] and not single else 0)
            hub.pause_for(wait - (time.monotonic() - t0))


def build_meta(args, fly):
    pos, rest, ids, kind = layout(fly, Path(args.coords))
    pre, post, w = fly.edges()
    others = np.flatnonzero(fly.role == OTHER)
    sample = np.random.default_rng(0).choice(others, size=min(80, len(others)), replace=False)
    rows = [dict(label=n, role=INPUT, neurons=p.tolist()) for n, p in zip(INPUT_NAMES, fly.inputs)]
    rows += [dict(label="push flap", role=FLAP, neurons=fly.flap_ids[:60]),
             dict(label="push glide", role=GLIDE, neurons=fly.glide_ids[:60]),
             dict(label=f"{len(sample)} others", role=OTHER, neurons=sample.tolist())]
    mode = "readout" if fly.ro is not None else "agent"
    name = Path(args.load).name if args.load else "untrained connectome"
    print(f"{fly.N} neurons, {len(fly.agent.brain.W.nzval)} synapses, {kind} layout, {mode}: {name}")
    return json.dumps(dict(
        n=fly.N, synapses=len(fly.agent.brain.W.nzval), layout=kind, mode=mode, name=name, learning=fly.learning,
        steps=fly.steps, dt=fly.dt, frame_skip=args.frame_skip, fps=GAME_FPS,
        pos=b64(pos, "<f4"), rest=b64(rest, "<f4"), role=b64(fly.role, "u1"), sign=b64(fly.sign, "i1"),
        ids=[str(i) for i in ids] if ids is not None else None,      # 64-bit ids: strings, JS would round them
        edges=b64(np.column_stack([pre, post]), "<u4"), edge_w=b64(w, "<f4"), rows=rows)).encode()


def parse_args(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--load", help="a .jls saved by flappy_readout.py or flappy_fly.py (default: untrained brain)")
    ap.add_argument("--learn", action="store_true", help="STDP agents only: keep learning from reward while you watch")
    ap.add_argument("--lr", type=float, default=0.2, help="STDP learning rate for an untrained brain with --learn")
    ap.add_argument("--coords", default=str(ROOT / "coordinates_princeton.csv"), help="FlyWire neuron positions")
    ap.add_argument("--pipe-gap", type=int, default=100)
    ap.add_argument("--frame-skip", type=int, default=3, help="game frames per brain decision")
    ap.add_argument("--score-limit", type=int, default=100)
    ap.add_argument("--game-seed", type=int, default=9_000_000, help="first game's seed (default: same as flappy_readout.py test)")
    ap.add_argument("--neuropils", default="AL_L,LH_L,MB_CA_L", help="must match how the saved brain was trained")
    ap.add_argument("--pop", type=int, default=20)
    ap.add_argument("--K", type=int, default=300)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--decide", choices=["max", "baseline"], default="max", help="STDP agents: see flappy_fly.py")
    ap.add_argument("--base-alpha", type=float, default=0.02)
    ap.add_argument("--flap-bias", type=float, default=0.0)
    ap.add_argument("--flap-target", type=float, default=0.0)
    ap.add_argument("--port", type=int, default=8303)
    ap.add_argument("--no-open", action="store_true", help="don't open the browser")
    args = ap.parse_args(argv)
    if args.load and not Path(args.load).exists():
        ap.error(f"brain file '{args.load}' not found. Train one with: "
                 f"python flappy_readout.py train --mode imitate --save {args.load}")
    return args


def main():
    args = parse_args()
    np.random.seed(args.seed)
    print("starting Julia and loading the connectome (about 10 s, longer the first time)...", flush=True)
    fly = Fly(args)
    # Julia takes over SIGINT and dumps a trace; nothing needs saving here, so just leave
    signal.signal(signal.SIGINT, lambda *_: os._exit(0))
    if args.learn and not fly.learning:
        print("--learn ignored: a readout brain's synapses are frozen")
    hub = Hub(args.pipe_gap)
    game = Game(args, fly, hub)
    server = ThreadingHTTPServer(("127.0.0.1", args.port), make_handler(hub, build_meta(args, fly)))
    server.daemon_threads = True
    threading.Thread(target=server.serve_forever, daemon=True).start()
    url = f"http://localhost:{args.port}/"
    print(f"open {url}   (Ctrl+C to stop)", flush=True)
    if not args.no_open:
        webbrowser.open(url)
    game.run()                   # Julia has to stay on the main thread


if __name__ == "__main__":
    main()
