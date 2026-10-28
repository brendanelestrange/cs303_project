Base.@kwdef struct Params
    dt::Float64 = 0.5         # ms
    v_rest::Float64 = -52.0   # mV
    v_th::Float64 = -45.0
    tau_m::Float64 = 20.0
    tau_syn::Float64 = 5.0
    t_ref::Float64 = 2.2
    w_syn::Float64 = 0.275    # mV per synapse
    noise_hz::Float64 = 2.0
    noise_mv::Float64 = 70.0
    tau_stdp::Float64 = 20.0
    tau_e::Float64 = 500.0
    a_plus::Float64 = 1.0
    a_minus::Float64 = 1.05
    lr::Float64 = 0.2
    w_max::Float64 = 3.0      # times the original synapse count
end

mutable struct Brain
    p::Params
    W::SparseMatrixCSC{Float64,Int}
    W0::Vector{Float64}
    sign::Vector{Int}
    incoming::SparseMatrixCSC{Int,Int}   # incoming[pre, post] = position of that synapse in W.nzval
    v::Vector{Float64}
    g::Vector{Float64}
    refrac::Vector{Int}
    x::Vector{Float64}
    y::Vector{Float64}
    e::Vector{Float64}
    spikes::Vector{Int}
    baseline::Float64
    learning::Bool
end

function Brain(c::Connectome, p::Params=Params())
    N = length(c.ids)
    post, pre, _ = findnz(c.W)
    incoming = sparse(pre, post, 1:length(pre), N, N)
    Brain(p, copy(c.W), copy(c.W.nzval), c.sign, incoming,
          fill(p.v_rest, N), zeros(N), zeros(Int, N), zeros(N), zeros(N), zeros(nnz(c.W)),
          Int[], 0.0, true)
end

function step!(b::Brain)
    p, W = b.p, b.W
    for j in b.spikes, k in nzrange(W, j)
        b.g[rowvals(W)[k]] += p.w_syn * b.sign[j] * W.nzval[k]
    end

    empty!(b.spikes)
    p_noise = p.noise_hz * p.dt / 1000
    decay = exp(-p.dt / p.tau_syn)
    for i in eachindex(b.v)
        rand() < p_noise && (b.g[i] += p.noise_mv)
        if b.refrac[i] > 0
            b.refrac[i] -= 1
        else
            b.v[i] += p.dt / p.tau_m * (p.v_rest - b.v[i] + b.g[i])
            if b.v[i] >= p.v_th
                b.v[i] = p.v_rest
                b.refrac[i] = round(Int, p.t_ref / p.dt)
                push!(b.spikes, i)
            end
        end
        b.g[i] *= decay
    end

    b.learning && stdp!(b)
    return b.spikes
end

function stdp!(b::Brain)
    p, W, I = b.p, b.W, b.incoming
    b.x .*= exp(-p.dt / p.tau_stdp)
    b.y .*= exp(-p.dt / p.tau_stdp)
    for j in b.spikes, k in nzrange(W, j)
        b.e[k] -= p.a_minus * b.y[rowvals(W)[k]]
    end
    for i in b.spikes, m in nzrange(I, i)
        b.e[nonzeros(I)[m]] += p.a_plus * b.x[rowvals(I)[m]]
    end
    b.x[b.spikes] .+= 1
    b.y[b.spikes] .+= 1
end

function reward!(b::Brain, r)
    b.learning || return
    surprise = r - b.baseline
    b.baseline += 0.05 * surprise
    W = b.W
    for j in axes(W, 2)
        b.sign[j] == 1 || continue
        for k in nzrange(W, j)
            W.nzval[k] = clamp(W.nzval[k] + b.p.lr * surprise * b.e[k], 0, b.p.w_max * b.W0[k])
        end
    end
end

function reset!(b::Brain)
    b.v .= b.p.v_rest
    b.g .= 0; b.refrac .= 0
    b.x .= 0; b.y .= 0; b.e .= 0
    empty!(b.spikes)
end
