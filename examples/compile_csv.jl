using CSV
using DataFrames

function build_non_grouped_csv(option::String; example = "sparse_reg")
    """
    Read out the set up data, i.e. seed, m , n etc.
    """
    function set_up_data(criterion, type; setting=false, long_run=false, comparison=false)
        prob = criterion in ["AF","DF","GTIF"] ? "fusion" : "opt"
        df_condi = if criterion in ["GTI","GTIF"]
            DataFrame(CSV.File(joinpath(@__DIR__, "csv/GTI_" * prob * "_" * type * "_data.csv")))
        elseif setting
            DataFrame(CSV.File(joinpath(@__DIR__, "csv/settings_" * prob * "_" * type * "_data.csv")))
        elseif long_run
            DataFrame(CSV.File(joinpath(@__DIR__, "csv/long_run_" * prob * "_" * type * "_data.csv")))
        elseif comparison
            DataFrame(CSV.File(joinpath(@__DIR__, "csv/" * prob * "_" * type * "_data.csv")))
        else
            DataFrame(CSV.File(joinpath(@__DIR__, "csv/" * prob * "_" * type * "_data.csv")))
        end

        seeds = df_condi[!,:seed]
        dims = df_condi[!,:numberOfExperiments]
        num_para = df_condi[!, :numberOfParameters]
        allowed_Ex = df_condi[!,:numberOfAllowedEx]
        fracs = df_condi[!,:frac]
        eigmax = df_condi[!,:eigmax]
        eigmin = df_condi[!,:eigmin]
        ratio = df_condi[!,:ratio]

        return seeds, dims, num_para, allowed_Ex, fracs, eigmax, eigmin, ratio
    end

    """
    Read out the time, solution etc from the individuals job files.
    """
    function read_data(criterion, solver, dimensions, folder, type)
        time = []
        solution = []
        termination = []
        num_n_o_c = []
        lower_bound = []
        dual_gap = []
        for m in dimensions
            @show m
            df_dim = DataFrame(CSV.File(joinpath(@__DIR__, "csv/"* folder *"/" * solver *"_" * criterion * "_" * string(m) * "_" * type * "_optimality.csv")))

            dim_termination = [row == "OPTIMAL" || row == "optimal" || row == "tree.lb>primal-dual_gap" || row == "primal>=tree.incumbent" ? 1 : 0 for row in df_dim[!,:termination]]

            time = vcat(time, df_dim[!,:time])
            if  solver == "custombb" && criterion in ["A","AF"]  #criterion in ["A","D","AF","DF"] &&
                solution = vcat(solution, df_dim[!,:solution])
            elseif solver == "custombb" && criterion in ["D","DF"]
                solution = vcat(solution, df_dim[!,:solution_scaled])
            elseif solver == "pajarito"
                solution = vcat(solution, df_dim[!,:solution]*1/m)
            elseif solver == "socp"
                for i in 1:length(df_dim[!,:scaled_solution])
                    if df_dim[!,:feasible] == "true"
                        solution = push!(solution, df_dim[i,:scaled_solution])
                    else
                        solution = push!(solution, Inf)
                    end
                end
                #solution = vcat(solution, df_dim[!,:scaled_solution])
            else
                solution = vcat(solution, df_dim[!,:solution])
            end
            termination = vcat(termination, dim_termination)
            if solver == "scip"
                num_n_o_c = vcat(num_n_o_c, df_dim[!,:calls])
            elseif solver == "boscia"
                num_n_o_c = vcat(num_n_o_c, df_dim[!,:num_nodes])
            elseif solver == "custombb"
                num_n_o_c = vcat(num_n_o_c, df_dim[!,:number_nodes])
            elseif solver == "socp"
                num_n_o_c = vcat(num_n_o_c, df_dim[!,:numberCuts])
            elseif solver == "pajarito"
                num_n_o_c = vcat(num_n_o_c, df_dim[!,:numberIterations])
            end
            if solver == "boscia"
                dual_gap = vcat(dual_gap, df_dim[!,:dual_gap])
                lower_bound = vcat(lower_bound, df_dim[!, :solution] - df_dim[!, :dual_gap]) 
            end
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

    if option == "comparison"
        criteria = ["A", "AF", "DF", "D"]
        solvers = ["Boscia","CustomBB","Pajarito","SOCP", "SCIP"]
        #solvers = ["Boscia","CustomBB","Pajarito", "SCIP"]
        dimensions = [50,60,80,100,120]

        for criterion in criteria
            @show criterion
            seeds, dims, num_para, allowed_Ex, fracs, eigmax, eigmin, ratio = set_up_data(criterion, type, setting = option == "setting", long_run = option == "long_run", comparison = option == "comparison")

            df = DataFrame()

            df[!,:seed] = seeds
            df[!,:numberOfExperiments] = dims
            df[!,:numberOfParameters] = num_para
            df[!,:numberOfAllowedEx] = allowed_Ex
            df[!,:frac] = fracs
            df[!,:eigmax] = eigmax
            df[!,:eigmin] = eigmin
            df[!,:ratio] = ratio 

            @show size(df)

            for solver in solvers
                if criterion in ["A","D"] && solver == "SCIP"
                    continue
                end
                @show solver
                time, solution, termination, num_n_o_c, dual_gap, lower_bound = read_data(criterion, lowercase(solver), dimensions, solver, type)

                df[!,Symbol("time"*solver)] = time
                df[!,Symbol("solution"*solver)] = solution
                df[!,Symbol("termination"*solver)] = termination
                df[!,Symbol("numberNodes"*solver)] = num_n_o_c
                if solver == "Boscia"
                    df[!,Symbol("dualGap"*solver)] = dual_gap
                    df[!,Symbol("lb"*solver)] = lower_bound

                    rel_gap = relative_gap(solution, lower_bound)
                else
                    df[!,Symbol("dualGap"*solver)] = solution - df[!,:lbBoscia]
                    rel_gap = relative_gap(solution, df[!,:lbBoscia])
                end
                df[!,Symbol("relGap"*solver)] = rel_gap

                @show size(df)
            end

            if criterion in ["AF","DF"]
                df[!,:minimumTime] = min.(df[!,:timeBoscia], df[!,:timeSCIP], df[!,:timeCustomBB], df[!,:timePajarito], df[!,:timeSOCP])
                df[!,:allSolved] = df[!,:terminationBoscia] + df[!,:terminationCustomBB] + df[!,:terminationSCIP] + df[!,:terminationPajarito] + df[!,:terminationSOCP].== 5

                #df[!,:minimumTime] = min.(df[!,:timeBoscia], df[!,:timeSCIP], df[!,:timeCustomBB], df[!,:timePajarito])
                #df[!,:allSolved] = df[!,:terminationBoscia] + df[!,:terminationCustomBB] + df[!,:terminationSCIP] + df[!,:terminationPajarito] .== 4
            else
                df[!,:minimumTime] = min.(df[!,:timeBoscia], df[!,:timeCustomBB], df[!,:timePajarito], df[!,:timeSOCP])
                df[!,:allSolved] = df[!,:terminationBoscia] + df[!,:terminationCustomBB] + df[!,:terminationPajarito] + df[!,:terminationSOCP] .== 4

                #df[!,:minimumTime] = min.(df[!,:timeBoscia], df[!,:timeCustomBB], df[!,:timePajarito])
                #df[!,:allSolved] = df[!,:terminationBoscia] + df[!,:terminationCustomBB] + df[!,:terminationPajarito] .== 3
            end

            file_name = joinpath(@__DIR__, "csv/Results/" * criterion * "_optimality_" * type * "_non_grouped.csv")
            CSV.write(file_name, df, append=false)
            println("\n")
        end
    elseif option == "setting"
    elseif option == "branching"
    else
        error("Unknown option!")
    end
end

function build_summary_by_difficulty()
end