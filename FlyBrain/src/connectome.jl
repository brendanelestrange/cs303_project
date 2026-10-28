const KIND = Dict("ACH" => 1, "GABA" => 2, "GLUT" => 2)
const SIGN = (1, -1, 0)

struct Connectome
    ids::Vector{Int}
    W::SparseMatrixCSC{Float64,Int}   # W[post, pre] = number of synapses
    sign::Vector{Int}
end

function load_connectome(path; neuropils=nothing, min_syn=5)
    cache = "$path.$(hash((neuropils, min_syn))).jls"
    isfile(cache) && return deserialize(cache)
    c = read_connectome(path, neuropils, min_syn)
    serialize(cache, c)
    return c
end

function read_connectome(path, neuropils, min_syn)
    index = Dict{Int,Int}()
    ids = Int[]
    votes = Vector{Vector{Int}}()
    function id!(root)
        get!(index, root) do
            push!(ids, root)
            push!(votes, zeros(Int, 3))
            length(ids)
        end
    end

    pre, post, syn = Int[], Int[], Float64[]
    for line in Iterators.drop(eachline(path), 1)
        p, q, neuropil, n, nt = split(line, ',')
        i, j, n = id!(parse(Int, p)), id!(parse(Int, q)), parse(Int, n)
        votes[i][get(KIND, nt, 3)] += n
        if neuropils === nothing || neuropil in neuropils
            push!(pre, i); push!(post, j); push!(syn, n)
        end
    end

    N = length(ids)
    W = sparse(post, pre, syn, N, N)
    W.nzval[W.nzval .< min_syn] .= 0
    dropzeros!(W)

    sign = [SIGN[argmax(v)] for v in votes]   # Dale's law: one transmitter per neuron
    keep = findall(vec(sum(W, dims=1)) .+ vec(sum(W, dims=2)) .> 0)
    return Connectome(ids[keep], W[keep, keep], sign[keep])
end
