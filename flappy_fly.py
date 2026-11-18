"""Let the FlyWire fruit-fly connectome (FlyBrain.jl) play Flappy Bird.

Python runs the game (flappy-bird-gymnasium); Julia runs the spiking brain.
Each decision: 4 game features -> input neuron populations -> 20 ms of
simulated brain activity -> the output population that spiked most picks
the action (0 = glide, 1 = flap). The game reward is fed back as dopamine-like
reward-modulated STDP.

Train headless:   python flappy_fly.py --episodes 200 --save fly_agent.jls
Watch it play:    python flappy_fly.py --load fly_agent.jls --episodes 5 --render
Random baseline:  python flappy_fly.py --random --episodes 50
"""
import argparse
import os
import time
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent
BIRD_X = 0.2 * 288 / 288      # bird's x, normalised (env puts it at 20% of width)
PIPE_W = 52 / 288             # pipe width, normalised
BIRD_H = 24 / 512             # bird height, normalised


def features(obs):
    """Turn the env's 12 normalised values into 4 inputs in [-1, 1]."""
    pipes = obs[:9].reshape(3, 3)            # (x, top-gap edge, bottom-gap edge), sorted by x
    ahead = [p for p in pipes if p[0] + PIPE_W > BIRD_X - 0.02]
    x, top, bottom = ahead[0] if ahead else pipes[-1]
    bird_y, vel = obs[9] + BIRD_H / 2, obs[10]
    gap_centre = (top + bottom) / 2 if bottom > top else 0.4
    return np.clip(np.array([
        (bird_y - gap_centre) * 5,   # >0: bird is below the gap centre -> should flap
        vel,                         # >0: falling
        1 - 2 * x,                   # pipe distance: -1 far, +1 close
        (bird_y - 0.4) * 3,          # >0: low on the screen
    ], dtype=np.float64), -1, 1)


# Julia helpers for options A and B (kept here so FlyBrain's own source is untouched).
JULIA_HELPERS = r"""
using SparseArrays
# A: pick output neurons that the input neurons actually reach in the connectome.
function downstream_outputs!(a, pop; hops=1)
    W = a.brain.W                                   # W[post, pre] = synapse count
    N = size(W, 1)
    s = zeros(N); for p in a.inputs, i in p; s[i] = 1.0; end
    exc = Float64.(a.brain.sign .== 1)
    score = W * (s .* exc)                          # excitatory synapses received from inputs
    if hops >= 2
        score .+= 0.2 .* (W * (exc .* score ./ max(maximum(score), 1.0)))
    end
    score[s .> 0] .= -Inf                           # never reuse an input neuron
    n = length(a.outputs) * pop
    chosen = shuffle(sortperm(score, rev=true)[1:n])
    a.outputs = collect.(Iterators.partition(chosen, pop))
    a.action_of .= 0
    for (k, p) in enumerate(a.outputs); a.action_of[p] .= k; end
    return (sum(score[chosen] .> 0), minimum(score[chosen]), maximum(score[chosen]))
end

# Wiring selection. Inputs come in pairs per observation: inputs[2d-1] fire for x > 0, inputs[2d] for x < 0.
# Features: 1 = bird below gap centre, 2 = falling, 3 = pipe close, 4 = low on screen. For 1, 2 and 4 the
# right move when x > 0 is FLAP (output group 2) and when x < 0 is GLIDE (group 1); 3 has no fixed answer.
# Score = how much the connectome's direct signed synapses already push that way.
function wiring_score(a)
    W, s = a.brain.W, a.brain.sign
    net = zeros(length(a.outputs), length(a.inputs))
    for (o, op) in enumerate(a.outputs), (p, ip) in enumerate(a.inputs)
        sub = W[op, ip]
        net[o, p] = sum(sub * Float64.(s[ip]))
    end
    pref = net[2, :] .- net[1, :]             # flap-minus-glide drive of each input population
    return sum(pref[2d-1] - pref[2d] for d in (1, 2, 4))
end

function set_wiring!(a, inputs, outputs)
    a.inputs = inputs; a.outputs = outputs
    a.action_of .= 0
    for (k, p) in enumerate(outputs); a.action_of[p] .= k; end
end

# Try n random assignments of the same input/output neurons; return [(score, seed, swapped)] best first.
function rank_wirings(a, n; swap=true)
    senders, outs = sort(reduce(vcat, a.inputs)), sort(reduce(vcat, a.outputs))
    pin, pout = length(a.inputs[1]), length(a.outputs[1])
    res = Tuple{Float64,Int,Bool}[]
    for seed in 1:n
        Random.seed!(seed)
        ins = collect.(Iterators.partition(shuffle(senders), pin))
        ous = collect.(Iterators.partition(shuffle(outs), pout))
        set_wiring!(a, ins, ous)
        sc = wiring_score(a)
        swapped = swap && sc < 0               # swapping flap/glide groups exactly negates the score
        push!(res, (swapped ? -sc : sc, seed, swapped))
    end
    sort!(res, rev=true)
    return res
end

function apply_wiring!(a, seed, swapped)
    senders, outs = sort(reduce(vcat, a.inputs)), sort(reduce(vcat, a.outputs))
    pin, pout = length(a.inputs[1]), length(a.outputs[1])
    # rebuild from the same sorted pools rank_wirings used
    Random.seed!(seed)
    ins = collect.(Iterators.partition(shuffle(senders), pin))
    ous = collect.(Iterators.partition(shuffle(outs), pout))
    swapped && reverse!(ous)
    set_wiring!(a, ins, ous)
    return wiring_score(a)
end

set_lr!(a, lr) = (a.brain.p = FlyBrain.Params(; (k => getfield(a.brain.p, k) for k in fieldnames(FlyBrain.Params))..., lr=lr); lr)

# E: reward-modulated STDP restricted to synapses onto the output group of the chosen action
# (excitatory presynaptic neurons only, same bounds as FlyBrain.reward!). Upstream synapses stay fixed.
function reward_chosen!(a, r, action)
    b = a.brain
    b.learning || return
    surprise = r - b.baseline
    b.baseline += 0.05 * surprise
    W, I = b.W, b.incoming
    for post in a.outputs[action + 1], m in nzrange(I, post)
        pre = rowvals(I)[m]
        b.sign[pre] == 1 || continue
        k = nonzeros(I)[m]
        W.nzval[k] = clamp(W.nzval[k] + b.p.lr * surprise * b.e[k], 0, b.p.w_max * b.W0[k])
    end
end

# B: compare each output group with its own running baseline, plus epsilon exploration.
# B2 (2 actions only): flap (action 1) must beat glide by th[1] spikes; if th[2] > 0 the
# threshold adapts so that flaps make up about th[2] of all decisions.
function act_rel!(a, obs, base, eps, alpha, th)
    b = a.brain
    counts = zeros(length(a.outputs))
    for _ in 1:a.steps
        for (d, x) in enumerate(obs)
            pop = x > 0 ? a.inputs[2d-1] : a.inputs[2d]
            p_spike = a.max_hz * min(abs(x), 1) * b.p.dt / 1000
            for i in pop
                rand() < p_spike && (b.g[i] += a.input_mv)
            end
        end
        for i in step!(b)
            a.action_of[i] > 0 && (counts[a.action_of[i]] += 1)
        end
    end
    b.e .*= exp(-a.steps * b.p.dt / b.p.tau_e)
    rel = counts .- base
    base .+= alpha .* (counts .- base)
    rand() < eps && return rand(0:length(counts)-1)
    if length(rel) == 2 && (th[1] != 0 || th[2] > 0)
        flap = rel[2] - rel[1] > th[1]
        th[2] > 0 && (th[1] += 0.1 * (flap - th[2]))
        return Int(flap)
    end
    m = maximum(rel)
    return rand(findall(==(m), rel)) - 1
end
"""


def load_brain(args):
    os.environ.setdefault("PYTHON_JULIACALL_HANDLE_SIGNALS", "yes")
    julia = Path(os.environ.get("LOCALAPPDATA", "")) / "Programs" / "Julia-1.13.4" / "bin" / "julia.exe"
    if julia.exists():
        os.environ.setdefault("PYTHON_JULIAPKG_EXE", str(julia))   # use this Julia, don't download another
        # Windows: Julia's DLLs (libpcre2 etc.) must be findable when Python loads libjulia
        os.environ["PATH"] = str(julia.parent) + os.pathsep + os.environ["PATH"]
        os.add_dll_directory(str(julia.parent))
    from juliacall import Main as jl
    jl.seval(f'import Pkg; Pkg.activate(raw"{ROOT / "FlyBrain"}"; io=devnull); using FlyBrain, Serialization, Random')
    jl.seval(f"Random.seed!({args.seed})")
    if args.load:
        agent = jl.seval(f'deserialize(raw"{Path(args.load).resolve()}")')
    else:
        neuropils = "nothing" if args.neuropils == "all" else repr(args.neuropils.split(","))
        neuropils = neuropils.replace("'", '"')
        agent = jl.seval(
            f'make_agent(raw"{ROOT / "connections_princeton.csv"}"; n_obs=4, n_actions=2, '
            f'neuropils={neuropils}, n_per_pop={args.pop}, action_ms={args.action_ms}, '
            f'max_hz={float(args.max_hz)}, input_mv={float(args.input_mv)}, '
            f'params=FlyBrain.Params(lr={float(args.lr)}, tau_e={float(args.tau_e)}))')
    jl.seval(JULIA_HELPERS)
    if args.outputs != "indegree" and not args.load:
        hops = 2 if args.outputs == "downstream2" else 1
        n_pos, lo, hi = jl.downstream_outputs_b(agent, args.pop, hops=hops)
        print(f"outputs rewired downstream of inputs ({hops} hop): {n_pos}/{2 * args.pop} receive input, "
              f"score {lo:.0f}..{hi:.0f}")
    if args.select_wiring and not args.load:
        ranked = jl.rank_wirings(agent, args.select_wiring, swap=not args.no_swap)
        top = [tuple(ranked[i]) for i in range(min(5, len(ranked)))]
        print("wiring candidates (score, seed, swapped), best first:",
              ", ".join(f"({s:.0f}, {sd}, {sw})" for s, sd, sw in top))
        score, wseed, swapped = tuple(ranked[args.wiring_rank - 1])
        score = jl.apply_wiring_b(agent, wseed, swapped)
        print(f"using wiring rank {args.wiring_rank}: seed {wseed}, flap/glide swapped={swapped}, "
              f"bias score {score:.0f}")
        jl.seval(f"Random.seed!({args.seed})")   # simulation noise still follows --seed
    elif not args.load and getattr(args, "print_wiring_score", True):
        print(f"wiring bias score {jl.wiring_score(agent):.0f}")
    return jl, agent


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--episodes", type=int, default=100)
    ap.add_argument("--frame-skip", type=int, default=3, help="game frames per brain decision (flap only on the first)")
    ap.add_argument("--neuropils", default="AL_L,LH_L,MB_CA_L", help='comma list, or "all" for the whole brain')
    ap.add_argument("--pop", type=int, default=20, help="neurons per input/output population")
    ap.add_argument("--action-ms", type=int, default=20, help="simulated brain time per decision")
    ap.add_argument("--max-hz", type=float, default=200.0, help="input neuron firing rate at |feature| = 1")
    ap.add_argument("--input-mv", type=float, default=70.0, help="kick (mV) per input spike")
    ap.add_argument("--lr", type=float, default=0.2, help="reward-modulated STDP learning rate")
    ap.add_argument("--tau-e", type=float, default=500.0, help="eligibility trace time constant (ms of brain time)")
    ap.add_argument("--pipe-gap", type=int, default=100, help="gap between pipes in pixels (bigger = easier)")
    ap.add_argument("--outputs", choices=["indegree", "downstream", "downstream2"], default="indegree",
                    help="option A: how output neurons are chosen (indegree = FlyBrain default)")
    ap.add_argument("--decide", choices=["max", "baseline"], default="max",
                    help="option B: max = most spikes wins; baseline = most spikes above its own running average")
    ap.add_argument("--epsilon", type=float, default=0.0, help="chance of a random action per decision")
    ap.add_argument("--flap-bias", type=float, default=0.0,
                    help="option B2: flap must beat glide by this many spikes (with --decide baseline)")
    ap.add_argument("--flap-target", type=float, default=0.0,
                    help="option B2: adapt the flap bias so this fraction of decisions are flaps (0 = off)")
    ap.add_argument("--learn", choices=["all", "chosen"], default="all",
                    help="option E: all = FlyBrain.reward! (every synapse); chosen = only synapses onto the chosen action's group")
    ap.add_argument("--base-alpha",type=float, default=0.02, help="update rate of each output's running baseline")
    ap.add_argument("--curriculum", type=float, default=0.0,
                    help="if > 0: every --curriculum-window games, shrink the pipe gap by 10 px (down to 100) "
                         "whenever the window's mean pipes per game is at least this value")
    ap.add_argument("--curriculum-window", type=int, default=50)
    ap.add_argument("--select-wiring", type=int, default=0,
                    help="score this many random input/output assignments by built-in flap/glide bias and use one")
    ap.add_argument("--wiring-rank", type=int, default=1, help="which candidate to use (1 = highest score)")
    ap.add_argument("--no-swap", action="store_true", help="don't swap flap/glide groups to fix a wrongly biased wiring")
    ap.add_argument("--stage-lr-decay", type=float, default=1.0,
                    help="multiply the learning rate by this each time a curriculum stage passes (0 = freeze)")
    ap.add_argument("--lr-halflife", type=float, default=0.0,
                    help="if > 0: learning rate halves every this many games")
    ap.add_argument("--shaping", type=float, default=0.0,
                    help="extra reward for moving toward the gap centre (0 = only the game's reward)")
    ap.add_argument("--no-learn",action="store_true", help="freeze the synapses (evaluation)")
    ap.add_argument("--render", action="store_true", help="show the game window")
    ap.add_argument("--random", action="store_true", help="random policy baseline (no brain)")
    ap.add_argument("--save", help="serialize the trained agent to this .jls file")
    ap.add_argument("--load", help="load a previously saved agent")
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    if args.load and not Path(args.load).exists():
        saved = sorted(p.name for p in Path.cwd().glob("fly*.jls")) + sorted(p.name for p in Path.cwd().glob("my*.jls"))
        raise SystemExit(f"Brain file '{args.load}' not found in {Path.cwd()}.\n"
                         f"Train one first with --save {args.load} (it is written every 100 games and at the end).\n"
                         f"Saved brains here: {', '.join(saved) or 'none'}")
    if not args.render:
        os.environ["SDL_VIDEODRIVER"] = "dummy"
    import gymnasium
    import flappy_bird_gymnasium  # noqa: F401  (registers FlappyBird-v0)

    env = gymnasium.make("FlappyBird-v0", render_mode="human" if args.render else None, use_lidar=False,
                         pipe_gap=args.pipe_gap)
    rng = np.random.default_rng(args.seed)
    if not args.random:
        t0 = time.time()
        jl, agent = load_brain(args)
        if args.no_learn:
            agent.brain.learning = False
        print(f"brain ready in {time.time() - t0:.0f}s: {len(agent.brain.v)} neurons, "
              f"{len(agent.brain.W.nzval)} synapses")

    scores, steps_all, gaps = [], [], []
    stage_mult = 1.0
    flaps = decisions = 0
    if not args.random:
        base = jl.zeros(2)   # running baseline spike count per output group (option B)
        th = jl.seval(f"[{float(args.flap_bias)}, {float(args.flap_target)}]")   # option B2
    t0 = time.time()
    for ep in range(args.episodes):
        obs, info = env.reset(seed=args.seed * 100_000 + ep)
        done, steps, total = False, 0, 0.0
        while not done:
            if args.random:
                action = int(rng.random() < 0.15)
            elif args.decide == "baseline":
                action = int(jl.act_rel_b(agent, features(obs), base, args.epsilon, args.base_alpha, th))
            elif args.epsilon and rng.random() < args.epsilon:
                jl.act_b(agent, features(obs))   # still run the brain so its state/eligibility advance
                action = int(rng.integers(2))
            else:
                action = int(jl.act_b(agent, features(obs)))
            flaps += action
            decisions += 1
            reward, dy_before = 0.0, abs(features(obs)[0])
            for k in range(args.frame_skip):
                obs, r, term, trunc, info = env.step(action if k == 0 else 0)
                reward += r
                steps += 1
                done = term or trunc
                if done:
                    break
            if args.shaping and not done:
                # dense signal: reward moving toward the centre of the next gap
                reward += args.shaping * (dy_before - abs(features(obs)[0]))
            total += reward
            if args.random:
                pass
            elif args.learn == "chosen":
                jl.reward_chosen_b(agent, reward, action)
            else:
                jl.reward_b(agent, reward)
        if not args.random:
            jl.reset_b(agent)
        scores.append(info["score"])
        steps_all.append(steps)
        gaps.append(env.unwrapped._pipe_gap)
        w = args.curriculum_window
        if args.curriculum and (ep + 1) % w == 0 and env.unwrapped._pipe_gap > 100 \
                and min(gaps[-w:]) == env.unwrapped._pipe_gap and np.mean(scores[-w:]) >= args.curriculum:
            env.unwrapped._pipe_gap = max(100, env.unwrapped._pipe_gap - 10)
            if not args.random and args.stage_lr_decay != 1.0:
                stage_mult *= args.stage_lr_decay
            print(f"ep {ep + 1:4d}  curriculum: last {w} games {np.mean(scores[-w:]):.2f} pipes "
                  f"-> gap now {env.unwrapped._pipe_gap}"
                  + (f", lr x{stage_mult:.3g}" if args.stage_lr_decay != 1.0 else ""), flush=True)
        if not args.random and (args.stage_lr_decay != 1.0 or args.lr_halflife):
            mult = stage_mult * (0.5 ** ((ep + 1) / args.lr_halflife) if args.lr_halflife else 1.0)
            jl.set_lr_b(agent, args.lr * mult)
            agent.brain.learning = (mult > 0) and not args.no_learn
        if (ep + 1) % 10 == 0 or args.render:
            last = slice(-10, None)
            print(f"ep {ep + 1:4d}  last10: pipes {np.mean(scores[last]):.2f} "
                  f"(best {max(scores[last])})  frames {np.mean(steps_all[last]):.0f}  "
                  f"[{time.time() - t0:.0f}s]", flush=True)
        if args.save and not args.random and (ep + 1) % 100 == 0:
            # checkpoint, so stopping early (Ctrl+C) still leaves a usable brain
            jl.serialize(str(Path(args.save).resolve()), agent)
            print(f"ep {ep + 1:4d}  checkpoint saved to {args.save}", flush=True)

    print(f"done:mean pipes {np.mean(scores):.2f}, last100 {np.mean(scores[-100:]):.2f}, "
          f"best {max(scores)}, mean frames {np.mean(steps_all):.0f}, last100 frames {np.mean(steps_all[-100:]):.0f}, "
          f"flap rate {flaps / max(decisions, 1):.2f}")
    if args.curriculum:
        final = env.unwrapped._pipe_gap
        at_final = [s for s, g in zip(scores, gaps) if g == final]
        print(f"curriculum: final gap {final}, {len(at_final)} games at it, "
              f"mean pipes there {np.mean(at_final[-100:]) if at_final else float('nan'):.2f} (last <=100)")
    if args.save and not args.random:
        jl.serialize(str(Path(args.save).resolve()), agent)
        print("saved", args.save)
    env.close()


if __name__ == "__main__":
    main()
