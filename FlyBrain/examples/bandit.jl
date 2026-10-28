using FlyBrain, Random

Random.seed!(2)
csv = joinpath(@__DIR__, "..", "..", "connections_princeton.csv")
agent = make_agent(csv; n_obs=2, n_actions=2, neuropils=["AL_L", "LH_L", "MB_CA_L"])

correct = 0
for trial in 1:3000
    cue = rand(1:2)
    obs = cue == 1 ? [1.0, 0.0] : [0.0, 1.0]
    action = act!(agent, obs) + 1
    reward!(agent, action == cue ? 1.0 : -1.0)
    reset!(agent)
    global correct += action == cue
    if trial % 300 == 0
        println("trial $trial  accuracy $(correct / 300)")
        global correct = 0
    end
end
