using Test, FlyBrain, SparseArrays

csv = joinpath(mktempdir(), "tiny.csv")
write(csv, """
pre_root_id,post_root_id,neuropil,syn_count,nt_type
101,102,AL_L,6,ACH
101,102,LH_L,4,ACH
102,103,LH_L,20,GABA
103,101,AL_L,3,ACH
104,102,AL_L,9,GLUT
104,103,LH_L,2,ACH
""")

@testset "connectome" begin
    c = load_connectome(csv)
    idx(root) = findfirst(==(root), c.ids)
    @test c.W[idx(102), idx(101)] == 10
    @test nnz(c.W) == 3
    @test c.sign[[idx(101), idx(102), idx(104)]] == [1, -1, -1]

    al = load_connectome(csv; neuropils=["AL_L"])
    @test length(al.ids) == 3
end

@testset "neurons fire and inhibit" begin
    exc = Connectome([1, 2], sparse([2], [1], [200.0], 2, 2), [1, 1])
    inh = Connectome([1, 2], sparse([2], [1], [200.0], 2, 2), [-1, 1])
    for (c, expect_2_fires) in ((exc, true), (inh, false))
        b = Brain(c, Params(noise_hz=0))
        fired = zeros(Int, 2)
        for _ in 1:2000
            b.g[1] += 1.0
            for i in step!(b)
                fired[i] += 1
            end
        end
        @test fired[1] > 0
        @test (fired[2] > 0) == expect_2_fires
    end
end

@testset "stdp" begin
    c = Connectome([1, 2], sparse([2], [1], [1.0], 2, 2), [1, 1])
    function pair(first, second)
        b = Brain(c, Params(noise_hz=0))
        for t in 1:100
            t == 10 && (b.g[first] += 100)
            t == 20 && (b.g[second] += 100)
            step!(b)
        end
        return b
    end
    causal = pair(1, 2)
    @test causal.e[1] > 0
    reward!(causal, 1.0)
    @test causal.W.nzval[1] > 1.0
    @test pair(2, 1).e[1] < 0
end

@testset "agent" begin
    a = make_agent(csv; n_obs=1, n_actions=1, n_per_pop=1)
    @test act!(a, [0.5]) == 0
    reward!(a, 1.0)
    reset!(a)
    @test all(a.brain.v .== a.brain.p.v_rest)
end
