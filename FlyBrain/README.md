# FlyBrain.jl

FlyWire connectome → spiking neurons → reward-modulated STDP. Standard library only.

| File | Contents |
|---|---|
| `src/connectome.jl` | read the CSV, sum synapses per neuron pair, Dale's law signs, cache |
| `src/brain.jl` | leaky integrate-and-fire neurons, STDP eligibility, `reward!` |
| `src/agent.jl` | observations → input neurons, output neurons → action |

```julia
agent = make_agent("connections_princeton.csv"; n_obs=4, n_actions=2, neuropils=["AL_L", "LH_L", "MB_CA_L"])
action = act!(agent, obs)   # 0-based
reward!(agent, r)
reset!(agent)               # between episodes
```

Run the tests: `julia --project=FlyBrain -e 'using Pkg; Pkg.test()'`
Check learning: `julia --project=FlyBrain FlyBrain/examples/bandit.jl` (accuracy goes from 50% to about 95%)

From Python (juliacall renames `act!` to `act_b`):

```python
from juliacall import Main as jl
jl.seval('import Pkg; Pkg.activate("FlyBrain"); using FlyBrain')
agent = jl.make_agent("connections_princeton.csv", n_obs=4, n_actions=2, neuropils=["AL_L", "LH_L", "MB_CA_L"])
action = jl.act_b(agent, obs)
jl.reward_b(agent, r)
```
