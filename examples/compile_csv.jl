using CSV
using DataFrames

"""
Create the set up data like seeds, dimensions etc.
"""
function set_up_data(df, example::String)
    if example in ["miplib_22433", "miplib_neos5", "miplib_ran14x18-disj-8", "miplib_pg5_34"]
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
            append!(seeds, fill(i, length(dimensions)))
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
            append!(seeds, fill(i, length(Ns)))
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
        df[!,:varA] = repeat(var_A, num_seeds * length(M) * length(dimensions))
        df[!,:p] = repeat(p, num_seeds * length(var_A) * length(M))
        df[!,:M] = repeat(vcat(fill(M[1], length(var_A)), fill(M[2], length(var_A))), num_seeds * length(dimensions))
    elseif example == "sparse_reg"
        num_seeds = 10
        seeds = Vector{Int64}()
        dimensions = collect(15:30)
        p = 5 .* dimensions
        k = ceil.(dimensions ./ 5)
        for i in 1:num_seeds
            append!(seeds, fill(i, length(dimensions)))
        end

        df[!,:seed] = seeds
        df[!,:dimension] = repeat(dimensions, num_seeds)
        df[!,:p] = repeat(k, num_seeds)
        df[!,:k] = repeat(p, num_seeds)
    elseif example == "tailed_cardinality_sparse_log_reg"
        num_seeds = 3
        seeds = Vector{Int64}()
        dimensions = collect(5:5:20)
        var_A = [1,5]
        M = [0.1,1]
        p = 5 .* dimensions
        k = ceil.(dimensions ./ 5)
        for i in 1:num_seeds
            append!(seeds, fill(i, length(M) * length(var_A)))
        end

        df[!,:seed] = repeat(seeds, length(dimensions))
        df[!,:dimension] = repeat(dimensions, num_seeds * length(M) * length(var_A))
        df[!,:varA] = repeat(var_A, num_seeds * length(M) * length(dimensions))
        df[!,:M] = repeat(vcat(fill(M[1], length(var_A)), fill(M[2], length(var_A))), num_seeds * length(dimensions))
    elseif example == "tailed_cardinality"
        num_seeds = 10
        seeds = Vector{Int64}()
        dimensions = collect(5:5:20)
        var_A = [1,5]
        M = [0.1,1]
        p = 5 .* dimensions
        k = ceil.(dimensions ./ 5)
        for i in 1:num_seeds
            append!(seeds, fill(i, length(M) * length(var_A)))
        end

        df[!,:seed] = repeat(seeds, length(dimensions))
        df[!,:dimension] = repeat(dimensions, num_seeds * length(M) * length(var_A))
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
    function read_data(example::String, solver::String, folder)
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
        optimality = ["OPTIMAL", "optimal", "Optimal", "Optimal (tolerance reached)", "tree.lb>primal-dual_gap", "primal>=tree.incumbent", "Optimal (tree empty)", "ALMOST_LOCALLY_SOLVED", "LOCALLY_SOLVED"]
        termination = [row in optimality ? 1 : 0 for row in df[!,:termination]]

        # All the problems are feasible.
        # If a solver returns that it isn't, it counts as non-solved.
        infeas_idx = findall(x->x=="INFEASIBLE", df[!,:termination])
        if !isempty(infeas_idx)
            time[infeas_idx] = 1800.0
        end
        
        if contains(solver, "boscia")
            dual_gap = df[!,:dual_gap]
            lower_bound = df[!,:solution] - df[!,:dual_gap]
            num_n_o_c = df[!,:ncalls]
        elseif solver == "ipopt"
            num_n_o_c = df[!,:num_nodes]
            time ./= 1000
        elseif solver == "scip_oa"
            num_n_o_c = df[!,:calls]
        end

        # The MIP LIB instances sometimes overshoot the time limit.
        over_time_idx = findall(x-> x > 1850.0, df[!,:time])
        if !isempty(over_time_idx)
            time[over_time_idx] .= 1800.0
            termination[over_time_idx] .= 0
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
        if occursin("Boscia", solver) #solver == "Boscia"
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

    if option == "comparison"
        solvers = ["Boscia", "Ipopt", "ScipOA", "Pavito", "Shot"]
    elseif option == "settings"
        solvers = ["Boscia","Boscia_Afw", "Boscia_No_As_No_Ss", "Boscia_No_As", "Boscia_No_Ss", "Boscia_Global_Tightening","Boscia_Local_Tightening", "Boscia_No_Tightening", "Boscia_Strong_Convexity"]
    elseif option == "branching"
        solvers = ["Boscia","Boscia_Strong_Branching", "Boscia_Hybrid_Branching_1", "Boscia_Hybrid_Branching_2", "Boscia_Hybrid_Branching_5", "Boscia_Hybrid_Branching_10", "Boscia_Hybrid_Branching_20"]
    else
        error("Unknown option!")
    end

    @show example

    # set up data
    df = DataFrame()
    df = set_up_data(df, example)

    @show size(df)

    minimumTime = fill(Inf, length(df[!,:seed]))

    # read out solver data
    for solver in solvers
        if example in ["tailed_cardinality", "tailed_cardinality_sparse_log_reg"] && solver in ["Ipopt","Pavito","Shot"]
            continue
        end
        if solver == "Boscia_Strong_Convexity" && !contains(example, "miplib")
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

    file_name = joinpath(@__DIR__, "final_csvs/" * example * "_" * option * "_non_grouped.csv")
    CSV.write(file_name, df, append=false)
    println("\n")
end

function build_summary_by_difficulty(option::String; example="sparse_reg")
    function geo_mean(group)
        prod = 1.0
        n = 0
        if isempty(group)
            return -1
        end
        for element in group
            # @show element
            if element != Inf
                prod = prod * abs(element)
                n += 1
            end
        end
        if n == 0
            return Inf
        end
        return prod^(1/n)
    end

    function geom_shifted_mean(xs; shift=big"1.0")
        a = length(xs)  
        n= 0
        prod = 1.0  
        if a != 0 
            for xi in xs
                if xi != Inf 
                    prod = prod*(xi+shift)  
                    n += 1
                end
            end
            return Float64(prod^(1/n) - shift)
        end
        return Inf
    end

    function custom_mean(group)
        sum = 0.0
        n = 0
        dash = false

        if isempty(group)
            return -1
        end
        for element in group
            if element == "-"
                dash = true
                continue
            end
            if element != Inf 
                if typeof(element) == String7 || typeof(element) == String3
                    element = parse(Float64, element)
                end
                sum += element
                n += 1
            end
        end
        if n == 0
            return dash ? "-" : Inf
        end
        return sum/n
    end

    function summarize(example, timeslots, solver, option)
        num_instances = []
        term = []
        term_rel = []
        time = []
        num_nodes = []
        rel_gap_nt = []

        @show solver

        df_ng = DataFrame(CSV.File(joinpath(@__DIR__, "final_csvs/" * example * "_" * option * "_non_grouped.csv")))

        termination = findall(x -> x==1, df_ng[!,Symbol("termination"*solver)])

        for timeslot in timeslots
            instances = findall(x -> x>timeslot, df_ng[!,:minimumTime])
            push!(num_instances, length(instances))

            term_in_time = intersect(instances, termination)
            push!(term, length(term_in_time))

            push!(term_rel, length(term_in_time)/length(instances)*100)
            push!(time, geom_shifted_mean(df_ng[instances, Symbol("time"*solver)]))
            if contains(solver, "Boscia") || solver in ["Ipopt", "ScipOA"]
                push!(num_nodes, custom_mean(df_ng[instances, Symbol("numberNodes"*solver)]))
            end

           # notSolved = intersect(instances, notAllSolved)
            if  isempty(instances) #isempty(notSolved)
                push!(rel_gap_nt, NaN)
            else
                push!(rel_gap_nt, custom_mean(df_ng[instances,Symbol("relGap"*solver)]))
            end
                
        end

        # rounding
        non_inf = findall(isfinite, rel_gap_nt)
        rel_gap_nt[non_inf] = round.(rel_gap_nt[non_inf], digits=2) 
        non_inf = findall(isfinite, time)
        time[non_inf] = round.(time[non_inf], digits=2)
        non_inf = findall(isfinite, num_nodes)
        num_nodes[non_inf] = convert.(Int64,round.(num_nodes[non_inf]))
        non_inf = findall(isfinite, term_rel)
        term_rel[non_inf] = convert.(Int64, round.(term_rel[non_inf]))
        term_rel[non_inf] = string.(term_rel[non_inf]) .* " %"

        println("\n")

        return num_instances, term, term_rel, time, num_nodes, rel_gap_nt
    end

    time_slots = [0, 10, 300, 600, 1200]

    if option == "comparison"
        solvers = ["Boscia", "Ipopt", "ScipOA", "Pavito", "Shot"]
    elseif option == "settings"
        solvers = ["Boscia","Boscia_Afw", "Boscia_No_As_No_Ss", "Boscia_No_As", "Boscia_No_Ss", "Boscia_Global_Tightening","Boscia_Local_Tightening", "Boscia_No_Tightening","Boscia_Strong_Convexity"]
    elseif option == "branching"
        solvers = ["Boscia","Boscia_Strong_Branching", "Boscia_Hybrid_Branching_1", "Boscia_Hybrid_Branching_2", "Boscia_Hybrid_Branching_5", "Boscia_Hybrid_Branching_10","Boscia_Hybrid_Branching_20"]
    else
        error("Unknown option!")
    end

    df = DataFrame()
    @show size(df)

    for (i, solver) in enumerate(solvers)
        if example in ["tailed_cardinality", "tailed_cardinality_sparse_log_reg"] && solver in ["Ipopt","Pavito","Shot"]
            continue
        end
        if solver == "Boscia_Strong_Convexity" && !contains(example, "miplib")
            continue
        end
        num_instances, num_terminated, rel_terminated, m_time, m_nodes_cuts, rel_gap_nt = summarize(example, time_slots, solver, option) 
    
        if i == 1
            df[!,:minTime] = time_slots
            df[!,:numInstances] = num_instances
            @show length(df[!,:numInstances])
        end
        @show length(num_terminated)
        df[!,Symbol(solver*"Term")] = num_terminated
        df[!,Symbol(solver*"TermRel")] = rel_terminated
        df[!,Symbol(solver*"Time")] = m_time
        df[!,Symbol(solver*"RelGapNT")] = rel_gap_nt

        if contains(solver, "Boscia") || solver in ["Ipopt","ScipOA"]
            df[!,Symbol(solver*"NodesOCuts")] = m_nodes_cuts
        end
    end
        
    file_name = joinpath(@__DIR__, "final_csvs/" * example * "_" * option * "_summary_by_difficulty.csv")
    CSV.write(file_name, df, append=false)
    println("\n")
end

examples = ["miplib_22433", "miplib_neos5", "miplib_pg5_34", "miplib_ran14x18-disj-8", "poisson_reg", "portfolio_integer", "portfolio_mixed", "sparse_log_reg", "sparse_reg", "tailed_cardinality", "tailed_cardinality_sparse_log_reg"]

examples = ["miplib_22433", "miplib_neos5", "miplib_pg5_34", "miplib_ran14x18-disj-8", "portfolio_mixed", "portfolio_integer", "sparse_log_reg", "sparse_reg", "tailed_cardinality", "tailed_cardinality_sparse_log_reg"]

examples = ["miplib_22433", "miplib_neos5", "miplib_pg5_34", "miplib_ran14x18-disj-8", ]

#examples = ["poisson_reg"]

for example in examples

    # comparison
    build_non_grouped_csv("comparison", example=example)
    build_summary_by_difficulty("comparison", example=example)

    # settings 
    build_non_grouped_csv("settings", example=example)
    build_summary_by_difficulty("settings", example=example)

    # branching
    build_non_grouped_csv("branching", example=example)
    build_summary_by_difficulty("branching", example=example)
end