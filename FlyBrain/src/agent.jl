mutable struct Agent
    brain::Brain
    inputs::Vector{Vector{Int}}   # two per observation: positive values, negative values
    outputs::Vector{Vector{Int}}  # one per action
    action_of::Vector{Int}
    steps::Int
    max_hz::Float64
    input_mv::Float64
end

function make_agent(path; n_obs, n_actions, neuropils=nothing, n_per_pop=20,
                    action_ms=20, max_hz=200.0, input_mv=70.0, params=Params())
    neuropils = neuropils === nothing ? nothing : String.(collect(neuropils))
    c = load_connectome(path; neuropils)
    out_degree = vec(sum(c.W .> 0, dims=1))
    in_degree = vec(sum(c.W .> 0, dims=2))

    senders = sortperm(out_degree, rev=true)[1:2n_obs*n_per_pop]
    receivers = filter(!in(Set(senders)), sortperm(in_degree, rev=true))[1:n_actions*n_per_pop]
    inputs = collect.(Iterators.partition(shuffle(senders), n_per_pop))
    outputs = collect.(Iterators.partition(shuffle(receivers), n_per_pop))

    action_of = zeros(Int, length(c.ids))
    for (a, pop) in enumerate(outputs)
        action_of[pop] .= a
    end
    steps = round(Int, action_ms / params.dt)
    Agent(Brain(c, params), inputs, outputs, action_of, steps, max_hz, input_mv)
end

function act!(a::Agent, obs)
    b = a.brain
    counts = zeros(Int, length(a.outputs))
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
    return rand(findall(==(maximum(counts)), counts)) - 1
end

reward!(a::Agent, r) = reward!(a.brain, r)
reset!(a::Agent) = reset!(a.brain)
