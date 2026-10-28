module FlyBrain

using SparseArrays, Random, Serialization

export Connectome, load_connectome, Params, Brain, step!, Agent, make_agent, act!, reward!, reset!

include("connectome.jl")
include("brain.jl")
include("agent.jl")

end
