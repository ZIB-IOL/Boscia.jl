import FrankWolfe: fast_dot, compute_extreme_point, muladd_memory_mode, get_active_set_iterate

"""
InfeasibleFrankWolfeNode functions

    InfeasibleFrankWolfeNode <: AbstractFrankWolfeNode

A node in the branch-and-bound tree storing information for a Frank-Wolfe subproblem.

`std` stores the id, lower and upper bound of the node.
`valid_active` vector of booleans indicating which vertices in the global active set are valid for the node.
`lmo` is the minimization oracle capturing the feasible region.
"""
mutable struct InfeasibleFrankWolfeNode{IB<:IntegerBounds} <: AbstractFrankWolfeNode
    std::Bonobo.BnBNodeInfo
    valid_active::Vector{Bool}
    local_bounds::IB
end

"""
InfeasibleFrankWolfeNode: Create the information of the new branching nodes 
based on their parent and the index of the branching variable
"""
function Bonobo.get_branching_nodes_info(
    tree::Bonobo.BnBTree,
    node::InfeasibleFrankWolfeNode,
    vidx::Int,
)
    # get solution
    x = Bonobo.get_relaxed_values(tree, node)

    # add new bounds to the feasible region left and right
    # copy bounds from parent
    varbounds_left = copy(node.local_bounds)
    varbounds_right = copy(node.local_bounds)

    if haskey(varbounds_left.upper_bounds, vidx)
        delete!(varbounds_left.upper_bounds, vidx)
    end
    if haskey(varbounds_right.lower_bounds, vidx)
        delete!(varbounds_right.lower_bounds, vidx)
    end
    push!(varbounds_left.upper_bounds, (vidx => MOI.LessThan(floor(x[vidx]))))
    push!(varbounds_right.lower_bounds, (vidx => MOI.GreaterThan(ceil(x[vidx]))))

    #valid_active is set at evaluation time
    node_info_left = (valid_active=Bool[], local_bounds=varbounds_left)
    node_info_right = (valid_active=Bool[], local_bounds=varbounds_right)

    return [node_info_left, node_info_right]

end


"""
Build up valid_active; is called whenever the global active_set changes
"""
function populate_valid_active!(
    active_set::FrankWolfe.ActiveSet,
    node::InfeasibleFrankWolfeNode,
    lmo::FrankWolfe.LinearMinimizationOracle,
)
    empty!(node.valid_active)
    for i in eachindex(active_set)
        push!(node.valid_active, is_linear_feasible(lmo, active_set.atoms[i]))
    end
end

function Bonobo.get_relaxed_values(tree::Bonobo.BnBTree, node::InfeasibleFrankWolfeNode)
    return copy(FrankWolfe.get_active_set_iterate(tree.root.problem.active_set))
end



"""
Blended pairwise CG coping with infeasible vertices in the active set.
Infeasible vertices are those for which the passed `filter_function(v)` is false.
They can be used only as away directions, not as forward ones. The LMO is always assumed to return feasible vertices.
"""
function infeasible_blended_pairwise(
    f,
    grad!,
    lmo,
    x0,
    node::InfeasibleFrankWolfeNode;
    line_search::FrankWolfe.LineSearchMethod=Adaptive(),
    epsilon=1e-7,
    max_iteration=10000,
    print_iter=1000,
    trajectory=false,
    verbose=false,
    memory_mode::FrankWolfe.MemoryEmphasis=InplaceEmphasis(),
    gradient=nothing,
    callback=nothing,
    timeout=Inf,
    print_callback=print_callback,
    renorm_interval=1000,
    lazy=false,
    linesearch_workspace=nothing,
    lazy_tolerance=2.0,
    filter_function=(args...) -> true,
    eager_filter=true,
    coldStart=false,
)
    # add the first vertex to active set from initialization
    active_set = ActiveSet([(1.0, x0)])

    return infeasible_blended_pairwise(
        f,
        grad!,
        lmo,
        active_set,
        node,
        line_search=line_search,
        epsilon=epsilon,
        max_iteration=max_iteration,
        print_iter=print_iter,
        trajectory=trajectory,
        verbose=verbose,
        memory_mode=memory_mode,
        gradient=gradient,
        callback=callback,
        timeout=timeout,
        print_callback=print_callback,
        renorm_interval=renorm_interval,
        lazy=lazy,
        linesearch_workspace=linesearch_workspace,
        lazy_tolerance=lazy_tolerance,
        filter_function=filter_function,
        eager_filter=eager_filter,
        coldStart=coldStart,
    )
end

function infeasible_blended_pairwise(
    f,
    grad!,
    lmo,
    active_set::FrankWolfe.ActiveSet,
    node::InfeasibleFrankWolfeNode;
    line_search::FrankWolfe.LineSearchMethod=FrankWolfe.Adaptive(),
    epsilon=1e-7,
    max_iteration=10000,
    print_iter=1000,
    trajectory=false,
    verbose=false,
    memory_mode::FrankWolfe.MemoryEmphasis=FrankWolfe.InplaceEmphasis(),
    gradient=nothing,
    callback=nothing,
    timeout=Inf,
    print_callback=FrankWolfe.print_callback,
    renorm_interval=1000,
    lazy=false,
    linesearch_workspace=nothing,
    lazy_tolerance=2.0,
    filter_function=(args...) -> true,
    eager_filter=true, # removes infeasible points from the get go
    coldStart=false, # if the active set is completey infeasible, it will be cleaned up and restarted
)
    # format string for output of the algorithm
    format_string = "%6s %13s %14e %14e %14e %14e %14e %14i\n"

    # if true, all infeasible vertices will be deleted from the active set
    if eager_filter
        if count(node.valid_active) > 0
            for i in Iterators.reverse(eachindex(node.valid_active))
                if !node.valid_active[i]
                    deleteat!(active_set, i)
                    deleteat!(node.valid_active, i)
                end
            end
            FrankWolfe.active_set_renormalize!(active_set)
        else
            v = compute_extreme_point(lmo, randn(length(active_set.atoms[1])))
            FrankWolfe.empty!(active_set)
            empty!(node.valid_active)
            FrankWolfe.push!(active_set, (1.0, v))
            push!(node.valid_active, true)
        end
        FrankWolfe.compute_active_set_iterate!(active_set)
    end

    t = 0
    primal = Inf
    x = get_active_set_iterate(active_set)
    feasible_x = eager_filter ? x : calculate_feasible_x(active_set, node, coldStart)
    tt = FrankWolfe.regular
    traj_data = []
    if trajectory && callback === nothing
        callback = FrankWolfe.trajectory_callback(traj_data)
    end
    time_start = time_ns()

    d = similar(x)

    if verbose
        println("\nInfeasible Blended Pairwise Conditional Gradient Algorithm.")
        NumType = eltype(x)
        println(
            "MEMORY_MODE: $memory_mode STEPSIZE: $line_search EPSILON: $epsilon MAXITERATION: $max_iteration TYPE: $NumType",
        )
        grad_type = typeof(gradient)
        println("GRADIENTTYPE: $grad_type LAZY: $lazy lazy_tolerance: $lazy_tolerance")
        if memory_mode isa FrankWolfe.InplaceEmphasis
            @info("In memory_mode memory iterates are written back into x0!")
        end
        headers =
            ("Type", "Iteration", "Primal", "Dual", "Dual Gap", "Time", "It/sec", "#ActiveSet")
        print_callback(headers, format_string, print_header=true)
    end

    # likely not needed anymore as now the iterates are provided directly via the active set
    if gradient === nothing
        gradient = similar(x)
    end

    grad!(gradient, x)
    v = compute_extreme_point(lmo, gradient)
    # if !lazy, phi is maintained as the global dual gap
    phi = max(0, fast_dot(feasible_x, gradient) - fast_dot(v, gradient))
    local_gap = zero(phi)
    gamma = 1.0

    if linesearch_workspace === nothing
        linesearch_workspace = FrankWolfe.build_linesearch_workspace(line_search, x, gradient)
    end

    while t <= max_iteration && (phi >= max(epsilon, eps()) || !filter_function(lmo, x))

        # managing time limit
        time_at_loop = time_ns()
        if t == 0
            time_start = time_at_loop
        end
        # time is measured at beginning of loop for consistency throughout all algorithms
        tot_time = (time_at_loop - time_start) / 1e9

        if timeout < Inf
            if tot_time ≥ timeout
                if verbose
                    @info "Time limit reached"
                end
                break
            end
        end

        #####################

        # compute current iterate from active set
        x = get_active_set_iterate(active_set)
        grad!(gradient, x)
        feasible_x = eager_filter ? x : calculate_feasible_x(active_set, node, coldStart)

        _, v_local, v_local_loc, v_val, a_val, a, a_loc, _, _ =
            active_set_argminmax_filter(active_set, gradient, node, lmo, filter_function)

        local_gap = fast_dot(gradient, a) - fast_dot(gradient, v_local)
        # if not finite, there is no feasible vertex to move towards,
        # local_gap = -Inf to be sure to pick the FW vertex
        if !isfinite(v_val)
            local_gap -= Inf
        end
        if !lazy
            v = compute_extreme_point(lmo, gradient)
            dual_gap = fast_dot(gradient, feasible_x) - fast_dot(gradient, v)
            phi = dual_gap
        end
        # minor modification from original paper for improved sparsity
        # (proof follows with minor modification when estimating the step)
        if local_gap ≥ phi / lazy_tolerance
            if !is_linear_feasible(lmo, a)
                deleteat!(active_set, a_loc)
                FrankWolfe.active_set_renormalize!(active_set)
                FrankWolfe.compute_active_set_iterate!(active_set)
            else
                d = muladd_memory_mode(memory_mode, d, a, v_local)
                vertex_taken = v_local
                gamma_max = a_val
                gamma = FrankWolfe.perform_line_search(
                    line_search,
                    t,
                    f,
                    grad!,
                    gradient,
                    x,
                    d,
                    gamma_max,
                    linesearch_workspace,
                    memory_mode,
                )
                # reached maximum of lambda -> dropping away vertex
                if gamma ≈ gamma_max
                    tt = FrankWolfe.drop
                    active_set.weights[v_local_loc] += gamma
                    deleteat!(active_set, a_loc)
                    @assert FrankWolfe.active_set_validate(active_set)
                else # transfer weight from away to local FW
                    tt = FrankWolfe.pairwise
                    active_set.weights[a_loc] -= gamma
                    active_set.weights[v_local_loc] += gamma
                end
                populate_valid_active!(active_set, node, lmo)
                FrankWolfe.active_set_update_iterate_pairwise!(active_set, gamma, v_local, a)
            end
        else # add to active set
            if lazy # otherwise, v computed above already
                v = compute_extreme_point(lmo, gradient)
            end
            vertex_taken = v
            dual_gap = fast_dot(gradient, feasible_x) - fast_dot(gradient, v)
            if (!lazy || dual_gap ≥ phi / lazy_tolerance)
                tt = FrankWolfe.regular
                d = FrankWolfe.muladd_memory_mode(memory_mode, d, x, v)

                gamma = FrankWolfe.perform_line_search(
                    line_search,
                    t,
                    f,
                    grad!,
                    gradient,
                    x,
                    d,
                    one(eltype(x)),
                    linesearch_workspace,
                    memory_mode,
                )
                # dropping active set and restarting from singleton
                if gamma ≈ 1.0
                    FrankWolfe.active_set_initialize!(active_set, v)
                else
                    renorm = mod(t, renorm_interval) == 0
                    FrankWolfe.active_set_update!(active_set, gamma, v)
                end
                @assert FrankWolfe.active_set_validate(active_set)
                populate_valid_active!(active_set, node, lmo)
            else # dual step
                tt = FrankWolfe.dualstep
                # set to computed dual_gap for consistency between the lazy and non-lazy run.
                # that is ok as we scale with the K = 2.0 default anyways
                phi = dual_gap
            end
        end
        if (
            ((mod(t, print_iter) == 0 || tt == FrankWolfe.dualstep) == 0 && verbose) ||
            callback !== nothing ||
            !(
                line_search isa FrankWolfe.Agnostic ||
                line_search isa FrankWolfe.Nonconvex ||
                line_search isa FrankWolfe.FixedStep
            )
        )
            primal = f(x)
        end
        if callback !== nothing
            state = (
                t=t,
                primal=primal,
                dual=primal - phi,
                dual_gap=phi,
                time=tot_time,
                x=x,
                v=vertex_taken,
                gamma=gamma,
                active_set=active_set,
                gradient=gradient,
            )
            callback(state)
        end

        if verbose && (mod(t, print_iter) == 0 || tt == FrankWolfe.dualstep)
            if t == 0
                tt = FrankWolfe.initial
            end
            rep = (
                st[Symbol(tt)],
                string(t),
                Float64(primal),
                Float64(primal - dual_gap),
                Float64(dual_gap),
                tot_time,
                t / tot_time,
                length(active_set),
            )
            print_callback(rep, format_string)
            flush(stdout)
        end
        t += 1
    end

    # recompute everything once more for final verfication / do not record to trajectory though for now!
    # this is important as some variants do not recompute f(x) and the dual_gap regularly but only when reporting
    # hence the final computation.
    # do also cleanup of active_set due to many operations on the same set

    if verbose
        x = get_active_set_iterate(active_set)
        grad!(gradient, x)
        v = compute_extreme_point(lmo, gradient)
        primal = f(x)
        phi = fast_dot(x, gradient) - fast_dot(v, gradient)
        tt = FrankWolfe.last
        rep = (
            FrankWolfe.st[Symbol(tt)],
            string(t - 1),
            Float64(primal),
            Float64(primal - phi),
            Float64(phi),
            (time_ns() - time_start) / 1.0e9,
            t / ((time_ns() - time_start) / 1.0e9),
            length(active_set),
        )
        print_callback(rep, format_string)
        flush(stdout)
    end
    FrankWolfe.active_set_renormalize!(active_set)
    FrankWolfe.active_set_cleanup!(active_set)
    populate_valid_active!(active_set, node, lmo)
    x = FrankWolfe.get_active_set_iterate(active_set)
    grad!(gradient, x)
    v = compute_extreme_point(lmo, gradient)
    primal = f(x)
    dual_gap = fast_dot(x, gradient) - fast_dot(v, gradient)
    if verbose
        tt = FrankWolfe.pp
        rep = (
            st[Symbol(tt)],
            string(t - 1),
            Float64(primal),
            Float64(primal - dual_gap),
            Float64(dual_gap),
            (time_ns() - time_start) / 1.0e9,
            t / ((time_ns() - time_start) / 1.0e9),
            length(active_set),
        )
        print_callback(rep, format_string)
        print_callback(nothing, format_string, print_footer=true)
        flush(stdout)
    end

    return x, v, primal, dual_gap, traj_data, active_set
end

function active_set_argminmax_filter(
    active_set::FrankWolfe.ActiveSet,
    direction,
    node::InfeasibleFrankWolfeNode,
    lmo::FrankWolfe.LinearMinimizationOracle,
    filter_function::Function;
    Φ=0.5,
)
    val = Inf
    valM = -Inf
    idx = -1
    idxM = -1
    for i in eachindex(active_set)
        temp_val = fast_dot(active_set.atoms[i], direction)
        if !node.valid_active[i]
            @debug "Hitting infeasible vertex"
        end
        if temp_val < val && node.valid_active[i]
            val = temp_val
            idx = i
        end
        if valM < temp_val && active_set.weights[i] != 0.0 # don't step away from "deleted" vertices
            valM = temp_val
            x = zeros(length(active_set.atoms[1]))
            s = zero(eltype(x))
            # Are there any feasible vertices
            idxM = i
        end
    end
    return (active_set[idx]..., idx, val, active_set[idxM]..., idxM, valM, valM - val ≥ Φ)
end

function calculate_feasible_x(
    active_set::FrankWolfe.ActiveSet,
    node::InfeasibleFrankWolfeNode,
    coldStart::Bool,
)
    x = zeros(length(active_set.atoms[1]))
    s = zero(eltype(x))
    # Are there any feasible vertices
    if count(node.valid_active) > 0
        for i in eachindex(active_set)
            if node.valid_active[i]
                x .+= active_set.weights[i] * active_set.atoms[i]
                s += active_set.weights[i]
            end
        end
        x ./= s
    else
        v = compute_extreme_point(node.lmo, randn(length(active_set.atoms[1])))
        λ = 0.0
        # If coldStart tre, completely restart the active set
        if coldStart
            FrankWolfe.empty!(active_set)
            empty!(node.valid_active)
            λ = 1.0
        else # if not, add v and asign greatest currently appearing weight
            λ = maximum(active_set.weights)
        end
        FrankWolfe.push!(active_set, (λ, v))
        push!(node.valid_active, true)
        FrankWolfe.active_set_renormalize!(active_set)
        FrankWolfe.compute_active_set_iterate!(active_set)
        x .= v
    end
    return x
end
