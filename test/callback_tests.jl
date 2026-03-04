using Boscia
using FrankWolfe
using Test
using Random
using StableRNGs


println("\nCallback Tests")

seed = rand(UInt64)
@show seed
rng = StableRNG(seed)

# Testing of the interface function solve

n = 20
diffi = rand(rng, Bool, n) * 0.6 .+ 0.3

@testset "Callback tests" begin
    function f(x)
        return sum(0.5 * (x .- diffi) .^ 2)
    end
    function grad!(storage, x)
        @. storage = x - diffi
    end

    cube_lmo = Boscia.CubeLMO(zeros(n), ones(n), collect(1:n))
    lmo = Boscia.ManagedLMO(cube_lmo, zeros(n), ones(n), collect(1:n), n)

    @testset "BnB callback" begin
        # Stop once 
        eval_nodes = 0
        function bnb_callback(tree, node; worse_than_incumbent=false, node_infeasible=false, lb_update=false)
            eval_nodes += 1
            if eval_nodes > rand(rng, 1:Int(floor(n/2)))
                tree.root.problem.solving_stage = Boscia.USER_STOP
                return false
            end
            return true
        end
        settings = Boscia.create_default_settings()
        settings.branch_and_bound[:bnb_callback] = bnb_callback
        x_bnb, _, result = Boscia.solve(f, grad!, lmo, settings=settings)

        @test result[:status] == Boscia.USER_STOP
        @test eval_nodes <= Int(floor(n/2))
    end

    @testset "Branch callback" begin
        num_branch = 0
        function branch_callback(tree, node, vidx)
            num_branch += 1
            return false, false
        end
        settings = Boscia.create_default_settings()
        settings.branch_and_bound[:branch_callback] = branch_callback
        x_branch, _, result = Boscia.solve(f, grad!, lmo, settings=settings)

        @test num_branch != 0
    end

    @testset "Propagate bounds" begin
        calls = 0
        function propagate_bounds(tree, node)
            calls += 1
        end
        settings = Boscia.create_default_settings()
        settings.tightening[:propagate_bounds] = propagate_bounds
        x_propagate, _, result = Boscia.solve(f, grad!, lmo, settings=settings)

        @test calls != 0
    end
end