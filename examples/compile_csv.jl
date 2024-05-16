using CSV
using DataFrames

"""
Create the set up data like seeds, dimensions etc.
"""
function set_up_data(df, example::String)
    if example in ["mip_lib_22433", "mip_lib_neos5", "mip_lib_ran14x18-disj-8", "mip_lib_pg5_34"]
        num_seeds = 3
        seeds = Vector{Int64}()
        numV = collect(4:8)
        for i in 1:num_seeds
            append!(seeds, fill(i, length(numV)))
        end

        df[!,:seed] = seeds
        df[!,:numV] = repeat(numV, num_seeds)
    elseif example in ["portfolio_integer", "portfolio_mixed"]
        num_seeds = 10
        seeds = Vector{Int64}()
        dimensions = collect(20:5:120)
        for i in 1:num_seeds
            append!(seeds, fill(i, length(dimensions))
        end

        df[!,:seed] = seeds
        df[!,:dimension] = repeat(dimensions, num_seeds)
    elseif example == "poisson_reg"
        num_seeds = 10
        seeds = Vector{Int64}()
        dimensions = collect(50:20:100)
        k = dimensions./2
        Ns = [0.1,1,5,10]
        for i in 1:num_seeds
            append!(seeds, fill(i, length(Ns))
        end

        df[!,:seed] = repeat(seeds, length(dimensions))
        df[!,:dimension] = repeat(dimensions, num_seeds*length(Ns))
        df[!,:Ns] = repeat(Ns, num_seeds*length(dimensions))
    elseif example == "sparse_log_reg"
        num_seeds = 3
        seeds = Vector{Int64}()
        dimensions = collect(5:5:20)
        M = [0.1, 1]
        var_A = [1, 5]
        p = 5 .* dimensions
        for i in 1:num_seeds
            append!(seeds, fill(i, length(M) * length(var_A)))
        end

        df[!,:seed] = repeat(seeds, length(dimensions))
        df[!,:dimension] = repeat(dimensions, num_seeds * length(var_A) * length(M))
        df[!,:varA] = repeat(varA, num_seeds * length(M) * length(dimensions))
        df[!,:k] = repeat(k, num_seeds * length(var_A) * length(M))
        df[!,:M] = repeat(vcat(fill(M[1], length(var_A)), fill(M[2], length(var_A))), num_seeds * length(dimensions))
    elseif example == "sparse_reg"
        num_seeds = 10
        seeds = Vector{Int64}()
        dimensions = collect(15:30)
        p = 5 .* dimensions
        k = ceil.(dimesions ./ 5)
        for i in 1:num_seeds
            append!(seeds, fill(i, length(dimensions))
        end

        df[!,:seed] = seeds
        df[!,:dimension] = repeat(dimensions, num_seeds)
        df[!,:p] = repeat(k, num_seeds)
        df[!,:k] = repeat(p, num_seeds)
    elseif example == "tailed_cardinality_sparse_log_reg"
        num_seeds = 10
        seeds = Vector{Int64}()
        dimensions = collect(5:5:20)
        var_A = [1,5]
        M = [0.1,1]
        p = 5 .* dimensions
        k = ceil.(dimesions ./ 5)
        for i in 1:num_seeds
            append!(seeds, fill(i, length(M) * length(var_A))
        end

        df[!,:seed] = repeat(seeds, length(dimensions))
        df[!,:dimension] = repeat(dimensions, num_seeds * length(M) * length(A))
        df[!,:varA] = repeat(var_A, num_seeds * length(M) * length(dimensions))
        df[!,:M] = repeat(vcat(fill(M[1], length(var_A)), fill(M[2], length(var_A))), num_seeds * length(dimensions))
    elseif example == "tailed_cardinality"
        num_seeds = 10
        seeds = Vector{Int64}()
        dimensions = collect(5:5:20)
        var_A = [1,5]
        M = [0.1,1]
        p = 5 .* dimensions
        k = ceil.(dimesions ./ 5)
        for i in 1:num_seeds
            append!(seeds, fill(i, length(M) * length(var_A))
        end

        df[!,:seed] = repeat(seeds, length(dimensions))
        df[!,:dimension] = repeat(dimensions, num_seeds * length(M) * length(A))
        df[!,:varA] = repeat(var_A, num_seeds * length(M) * length(dimensions))
        df[!,:M] = repeat(vcat(fill(M[1], length(var_A)), fill(M[2], length(var_A))), num_seeds * length(dimensions))
    else
        error("Unknown example!")
    end
    return df
end

function build_non_grouped_csv(option::String; example = "sparse_reg")
    """
    Read out the time, solution etc from the individuals job files.
    """
    function read_data(example::String, solver::string, folder)
        time = []
        solution = []
        termination = []
        num_n_o_c = []
        lower_bound = []
        dual_gap = []

        @show example, solver

        df = DataFrame(CSV.File(joinpath(@__DIR__, folder * "/" * solver * "_" * example * ".csv")))

        time = df[!,:time]
        solution = df[!,:solution]
        termination = [row == "OPTIMAL" || row == "optimal" || row == "tree.lb>primal-dual_gap" || row == "primal>=tree.incumbent" ? 1 : 0 for row in df[!,:termination]]

        if solver == "boscia"
            dual_gap = df[!,:dual_gap]
            lower_bound = df[!,:solution] - df[!,:dual_gap]
            num_n_o_c = df[!,:ncalls]
        elseif solver == "ipopt"
            num_n_o_c = df[!,:num_nodes]
        elseif solver == "scip_oa"
            num_n_o_c = df[:calls]
        end

        @show length(time), length(solution), length(termination), length(num_n_o_c), length(dual_gap), length(lower_bound)

        return time, solution, termination, num_n_o_c, dual_gap, lower_bound
    end

    """
    Compute relative gap with respect to Boscia's lower bound
    """
    function relative_gap(solution, lower_bound)
        rel_gap = []
        for (i,_) in enumerate(solution)
            if min(abs(solution[i]), abs(lower_bound[i])) == 0
                push!(rel_gap, solution[i] - lower_bound[i])
            elseif sign(lower_bound[i]) != sign(solution[i])
                push!(rel_gap, Inf)
            else
                push!(rel_gap, (solution[i] - lower_bound[i])/min(abs(solution[i]), abs(lower_bound[i])))
            end
        end 
        return rel_gap
    end

    function combine_data(df, example, solver, solver_id, minimumTime)
        time, solution, termination, num_n_o_c, dual_gap, lower_bound = read_data(example, solver_id, "final_csvs" )

        df[!,Symbol("time"*solver)] = time
        df[!,Symbol("solution"*solver)] = solution
        df[!,Symbol("termination"*solver)] = termination
        if solver == "Boscia"
            df[!,Symbol("numberNodes"*solver)] = num_n_o_c
            df[!,Symbol("dualGap"*solver)] = dual_gap
            df[!,Symbol("lb"*solver)] = lower_bound

            rel_gap = relative_gap(solution, lower_bound)
        elseif solver in ["Ipopt", "ScipOA"]
            df[!, Symbol("numberNodes"*solver)] = num_n_o_c
        end

        rel_gap = solver == "Boscia" ? relative_gap(solution, lower_bound) : relative_gap(solution, df[!,:lbBoscia])
        df[!,Symbol("relGap"*solver)] = rel_gap

        minimumTime = min.(minimumTime, time)

        return df, minimumTime
    end

    examples = ["mip_lib_22433", "mip_lib_neos5", "mip_lib_pg5_34", "mip_lib_ran14x18", "poisson_reg", "portfolio_integer", "portfolio_mixed", "sparse_log_reg", "sparse_reg", "tailed_cardinality", "tailed_cardinality_sparse_log_reg"]

    if option == "comparison"
        solvers = ["Boscia", "Ipopt", "ScipOA", "Pavito", "Shot"]
    elseif option == "settings"
        solvers = ["Boscia","Boscia_Afw", "Boscia_No_As_No_Ss", "Boscia_No_As", "Boscia_No_Ss", "Boscia_Global_Tightening","Boscia_Local_Tightening", "Boscia_No_Tightening", "Boscia_Strong_Convexity"]
    elseif option == "branching"
        solvers = ["Boscia","Boscia_Strong_Branching", "Boscia_Hybrid_Branching_1", "Boscia_Hybrid_Branching_2", "Boscia_Hybrid_Branching_5", "Boscia_Hybrid_Branching_10", "Boscia_Hybrid_Branching_20"]
    else
        error("Unknown option!")
    end

    for example in examples
        @show example

        # set up data
        df = DataFrame()
        df = set_up_data(df, example)

        minimumTime = fill(Inf, length(df[!,:seed]))

        # read out solver data
        for solver in solvers
            if example in ["tailed_cardinality", "tailed_cardinality_sparse_log_reg"] && solver in ["Ipopt","Pavito","SHOT"]
                continue
            end
            if solver == "Boscia_Strong_Convexity" && !contains(example, "mip_lib")
                continue
            end
            @show solver
            if solver == "Boscia"
                solver1 = "boscia_default"
            elseif solver == "ScipOA"
                solver1 = "scip_oa"
            else
                solver1 = lowercase(solver)
            end

            df, minimumTime = combine_data(df, example, solver, solver1, minimumTime)
        end

        df[!,:minimumTime] = minimumTime

        file_name = joinpath(@__DIR__, "final_csvs/" * example * "_non_grouped.csv")
        CSV.write(file_name, df, append=false)
        println("\n")
    end
end

function build_summary_by_difficulty()
end