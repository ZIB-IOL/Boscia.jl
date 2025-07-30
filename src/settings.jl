"""
    settings_bnb(mode::Mode;...)

Set the settings for the branch-and-bound algorithm.

Requires:

- `mode` the mode of the algorithm. See the `Boscia.Mode` enum for the available modes. If no mode is provided, the default mode is used.

Returns:

- `Dict` of settings for the branch-and-bound algorithm.

Available settings:

- `traverse_strategy` encodes how to choose the next node for evaluation. By default the node with the best lower bound is picked.
- `branching_strategy` fixes the branching strategy. By default, weuse `MOST_INFEASIBLE`, i.e. we branch on the entry which is the farthest away from being an integer.
- `verbose` if `true`, logs and solution statistics are printed. Per default, this is `false`.
- `node_limit` maximum number of nodes to be evaluated. Per default, there is no limit.
- `time_limit` algorithm will stop if the time limit is reached. Depending on the problem it is possible that no feasible solution has been found yet. On default, there is no time limit.
- `print_iter` encodes after how many processed nodes the current node and solution status is printed. The logs are always printed if a new integral solution has been found. Per default, `print_iter` is set to `100``.
- `bnb_callback` optional callback function that is called after every node evaluation. It will be called before the Boscia internal callback handling the printing of the logs. It receives the tree, the node and the following keyword arguments: `worse_than_incumbent=false`, `node_infeasible=false`, `lb_update=false`.
- `no_pruning` if `true`, no pruning of nodes is performed. Per default, nodes are pruned if they have a lower bound which is worse than the best known solution. Per default, this is `true` for the `HEURISTIC` mode and `false` for the `OPTIMAL` mode.
- `ignore_lower_bound` if `true`, the lower bound obtain by Frank-Wolfe is ignored and in the logs, only Inf will be printed. Per default, this is `true` for the `HEURISTIC` mode and `false` for the `OPTIMAL` mode.
- `start_solution` an initial solution can be provided if known. It will be used as the initial incumbent.
- `use_shadow_set` the shadow set is the set of discarded vertices which is inherited by the children nodes. It is used to avoid recomputing of vertices in case the BLMO is expensive. In case of a cheap BLMO, performance might improve by disabling this option. Per default, this is `true`.
"""
function settings_bnb(;
    mode::Mode=Boscia.DEFAULT_MODE,
    traverse_strategy=Bonobo.BestFirstSearch(),
    branching_strategy=Bonobo.MOST_INFEASIBLE(),
    verbose=false,
    node_limit=Inf,
    time_limit=Inf,
    print_iter=100,
    bnb_callback=nothing,
    no_pruning=mode == HEURISTIC_MODE ? true : false,
    ignore_lower_bound=mode == HEURISTIC_MODE ? true : false,
    start_solution=nothing,
    use_shadow_set=true,
)
    return Dict(
        :traverse_strategy => traverse_strategy,
        :branching_strategy => branching_strategy,
        :verbose => verbose,
        :node_limit => node_limit,
        :time_limit => time_limit,
        :print_iter => print_iter,
        :bnb_callback => bnb_callback,
        :no_pruning => no_pruning,
        :ignore_lower_bound => ignore_lower_bound,
        :start_solution => start_solution,
        :use_shadow_set => use_shadow_set,
    )
end

"""
    settings_frank_wolfe(mode::Mode;...)

Options for the Frank-Wolfe algorithm used as node solver.

Requires:

- `mode` the mode of the algorithm. See the `Boscia.Mode` enum for the available modes. If no mode is provided, the default mode is used.

Returns:

- `Dict` of settings for the Frank-Wolfe algorithm.

Available settings:

- `variant` the Frank-Wolfe variant to be used to solve the node problem. Options currently available are `AwayFrankWolfe`, `BlendedConditionalGradient`, `BlendedPairwiseConditionalGradient`, `DecompositionInvariantConditionalGradient` and `StandardFrankWolfe`. Per default, this is set to `BlendedPairwiseConditionalGradient`.
- `line_search` specifies the line search method used in the FrankWolfe variant. Default is the `FrankWolfe.Secant` line search. For other available types, check the FrankWolfe.jl package.
- `max_fw_iter` maximum number of iterations in a Frank-Wolfe run. Per default, this is set to `10000`.
- `fw_timeout` time limit for the Frank-Wolfe runs. Per default, there is no time limit. It is preferred to set the iteration limit but this can be used as a fallback and/or if the BLMO call is time consuming.
- `min_fw_iterations` the minimum number of Frank-Wolfe iterations performed in the node evaluation. Per default, this is set to `5`.
- `fw_verbose` if `true`, the Frank-Wolfe logs are printed at each node. Mostly meant for debugging. Per default, this is `false`.
- `lazy` flag specifies whether the lazification of the Frank-Wolfe variant should be used. Per default `true`. Note that it has no effect on standard Frank-Wolfe.
- `lazy_tolerance` decides how much progress is deemed enough to not have to call the LMO. Only used if the `lazy` flag is activated. Per default, this is set to `2`.
"""
function settings_frank_wolfe(;
    mode::Mode=Boscia.DEFAULT_MODE,
    variant=BlendedPairwiseConditionalGradient(),
    line_search=FrankWolfe.Secant(),
    max_fw_iter=10000,
    fw_timeout=Inf,
    min_fw_iterations=5,
    fw_verbose=false,
    lazy=true,
    lazy_tolerance=2,
)
    return Dict(
        :variant => variant,
        :line_search => line_search,
        :max_fw_iter => max_fw_iter,
        :fw_timeout => fw_timeout,
        :min_fw_iterations => min_fw_iterations,
        :fw_verbose => fw_verbose,
        :lazy => lazy,
        :lazy_tolerance => lazy_tolerance,
    )
end

"""
    settings_tolerances(mode::Mode;...)

Set the tolerances for the Frank-Wolfe algorithm. These are tolerances both for the Branch-and-Bound tree as well as for the Frank-Wolfe variant used as node solver.

Requires:

- `mode` the mode of the algorithm. See the `Boscia.Mode` enum for the available modes. If no mode is provided, the default mode is used.

Returns:

- `Dict` of tolerances for the Frank-Wolfe algorithm.

Available settings:

- `fw_epsilon` the solving precision of Frank-Wolfe at the root node.
- `dual_gap` absolute dual gap. If the difference between the incumbent and the lower bound reaches this value, the algorithm stops. Per default, this is set to `1e-6`.
- `rel_dual_gap` relative dual gap. If the difference between the incumbent and the lower bound reaches this value, the algorithm stops. Per default, this is set to `1e-2`.
- `dual_gap_decay_factor` the FrankWolfe tolerance at a given level `i` in the tree is given by `fw_epsilon * dual_gap_decay_factor^i` until we reach the `min_node_fw_epsilon`. Per default, this is set to `0.8`.
- `min_number_lower` if not `Inf`, evaluation of a node is stopped if at least `min_number_lower` open nodes have a better lower bound. Per default, this is set to `Inf`.
- `min_node_fw_epsilon` smallest fw epsilon tolerance, see also `dual_gap_decay_factor`. Per default, this is set to `1e-6`.
"""
function settings_tolerances(;
    mode::Mode=Boscia.DEFAULT_MODE,
    fw_epsilon=1e-2,
    dual_gap=1e-6,
    rel_dual_gap=1.0e-2,
    dual_gap_decay_factor=0.8,
    min_number_lower=Inf,
    min_node_fw_epsilon=1e-6,
)
    return Dict(
        :fw_epsilon => fw_epsilon,
        :dual_gap => dual_gap,
        :rel_dual_gap => rel_dual_gap,
        :dual_gap_decay_factor => dual_gap_decay_factor,
        :min_number_lower => min_number_lower,
        :min_node_fw_epsilon => min_node_fw_epsilon,
    )
end

"""
    settings_postprocessing(mode::Mode;...)

Set the settings for the postprocessing.

Requires:

- `mode` the mode of the algorithm. See the `Boscia.Mode` enum for the available modes. If no mode is provided, the default mode is used.

Returns:

- `Dict` of settings for the postprocessing.

Available settings:

- `use_postsolve` if `true`, runs the specified Frank-Wolfe variant on the problem with the integral variables fixed to the solution, i.e. it only optimizes over the continuous variables. This might improve the solution if one has many continuous variables. Per default, this is `true`.
- `max_iteration_post` maximum number of iterations in the Frank-Wolfe run during postsolve. Per default, this is set to `10000`.
"""
function settings_postprocessing(; mode::Mode=Boscia.DEFAULT_MODE, use_postsolve=true, max_iteration_post=10000)
    return Dict(:use_postsolve => use_postsolve, :max_iteration_post => max_iteration_post)
end

"""
    settings_heuristic(mode::Mode;...)

Set the settings for the heuristics.

Requires:

- `mode` the mode of the algorithm. See the `Boscia.Mode` enum for the available modes. If no mode is provided, the default mode is used.

Returns:

- `Dict` of settings for the heuristics.

Available settings:

- `custom_heuristics` list of custom heuristics from the user. Heuristics can be created via the `Boscia.Heuristic` constructor. It requires a function, a probability and an identifier (symbol). Note that the heuristics defined in Boscia themselves don't have to be added here and can be set via the probability parameters below.
- `post_heuristics_callback` callback function called whenever a new solution is found and added to the tree. 
- `prob_rounding` the probability for calling the simple rounding heuristic. Since the feasibility has to be checked, it might be expensive to do this for every node. Per default, this is activated for every node.
- `follow_gradient_prob` the probability for calling the follow-the-gradient heuristic. Per default, this is `0.0`.
- `follow_gradient_steps` the number of steps for the follow-the-gradient heuristic. Per default, this is `10`.
- `rounding_lmo_01_prob` the probability for calling the rounding-LMO-01 heuristic. Per default, this is `0.0`.
- `probability_rounding_prob` the probability for calling the probability-rounding heuristic. Per default, this is `0.0`.
- `hyperplane_aware_rounding_prob` the probability for calling the hyperplane-aware-rounding heuristic. Per default, this is `0.0`.
- `add_all_solutions` if `true`, all solutions found by the heuristics, Frank-Wolfe or the BLMO are added to the tree. Per default, this is `true` for the `HEURISTIC` mode and `false` for the `OPTIMAL` mode.
"""
function settings_heuristic(;
    mode::Mode=Boscia.DEFAULT_MODE,
    custom_heuristics=[Heuristic()],
    post_heuristics_callback=nothing,
    rounding_prob=1.0,
    follow_gradient_prob=0.0,
    follow_gradient_steps=10,
    rounding_lmo_01_prob=0.0,
    probability_rounding_prob=0.0,
    hyperplane_aware_rounding_prob=0.0,
    add_all_solutions=mode == HEURISTIC_MODE ? true : false,
)
    round_heu = Heuristic(rounding_heuristic, rounding_prob, :rounding)
    follow_grad_heu = Heuristic(
        (tree, tlmo, x) -> follow_gradient_heuristic(tree, tlmo, x, follow_gradient_steps),
        follow_gradient_prob,
        :follow_gradient,
    )
    rounding_lmo_01_heu =
        Heuristic(rounding_lmo_01_heuristic, rounding_lmo_01_prob, :rounding_lmo_01)
    probability_rounding_heu =
        Heuristic(probability_rounding, probability_rounding_prob, :probability_rounding)
    hyperplane_aware_rounding_heu = Heuristic(
        rounding_hyperplane_heuristic,
        hyperplane_aware_rounding_prob,
        :hyperplane_aware_rounding,
    )

    heuristics = vcat(
        [
            round_heu,
            follow_grad_heu,
            rounding_lmo_01_heu,
            probability_rounding_heu,
            hyperplane_aware_rounding_heu,
        ],
        custom_heuristics,
    )

    return Dict(
        :heuristics => heuristics,
        :post_heuristics_callback => post_heuristics_callback,
        :add_all_solutions => add_all_solutions,
    )
end

"""
    settings_tightening(mode::Mode;...)

Set the tightening parameters.

Requires:

- `mode` the mode of the algorithm. See the `Boscia.Mode` enum for the available modes. If no mode is provided, the default mode is used.

Returns:

- `Dict` of settings for the tightening.

Available settings:

- `dual_tightening` flag to decide  whether to use dual tightening techniques at node level. Note that this only porvides valid tightenings if your function is convex! Per default, this is `true`.
- `global_dual_tightening` flag to decide whether to generate dual tightenings from new solutions that are gloablly valid. Per default, this is `true`.
- `strong_convexity` strong convexity parameter of the objective `f`, used for tightening the dual bound at every node. Per default, this is set to `0.0`.
- `sharpness_constant` - the constant `M > 0` for `(θ, M)`-sharpness. `f` is `(θ, M)`-sharpness: `f` satisfies `min_{x^* ∈ X^*} || x - x^* || ≤ M (f(x) - f^(x^*))^θ` where `X^*` is the set of minimizer of `f`. Note that tightenings using sharpness are only valid if the problem has a unique minimizer, i.e. `f` is stricly convex! Per default, this is set to `0.0`.
- `sharpness_exponent` - the exponent `θ ∈ [0, 1/2]` for `(θ, M)`-sharpness. Per default, this is set to `Inf`.
- `propagate_bounds` optional function that allows the user to propagate and tighten bounds depending on the node. Receives the tree and the node as input.
"""
function settings_tightening(;
    mode::Mode=Boscia.DEFAULT_MODE,
    dual_tightening=true,
    global_dual_tightening=true,
    strong_convexity=0.0,
    sharpness_constant=0.0,
    sharpness_exponent=Inf,
    propagate_bounds=nothing,
)
    return Dict(
        :dual_tightening => dual_tightening,
        :global_dual_tightening => global_dual_tightening,
        :strong_convexity => strong_convexity,
        :sharpness_constant => sharpness_constant,
        :sharpness_exponent => sharpness_exponent,
        :propagate_bounds => propagate_bounds,
    )
end

"""
    settings_domain(mode::Mode;...)

To set settings for a non-trivial domain, i.e. if not all points of the feasible region are domain feasible.

Requires:

- `mode` the mode of the algorithm. See the `Boscia.Mode` enum for the available modes. If no mode is provided, the default mode is used.

Returns:

- `Dict` of settings for the domain.

Available settings:

- `domain_oracle` given a point `x`: returns `true` if `x` is in the domain of `f`, else false. Per default, it always returns `true`. In case of the non-trivial domain oracle, the initial point has to be domain feasible for `f` and can be set via the `active_set``. Additionally, the user has to provide a function `domain_point`, see below. Also, depending on the line search method, you might have to provide the domain oracle to it, too. The default line search Secant, for example, requires the domain oracle.
- `find_domain_point` given the current node bounds return a domain feasible point respecting the bounds. If no such point can be found, return `nothing`. Only necessary for a non-trivial domain oracle.
- `active_set` can be used to specify a starting point. By default, the direction (1,..,n) where n is the size of the problem is used to find a start vertex. This has to be of the type `FrankWolfe.ActiveSet`. Beware that the active set may only contain actual vertices of the feasible region.
"""
function settings_domain(;
    mode::Mode=Boscia.DEFAULT_MODE,
    domain_oracle=_trivial_domain,
    find_domain_point=_trivial_domain_point,
    active_set::Union{Nothing,FrankWolfe.ActiveSet}=nothing,
)
    return Dict(
        :domain_oracle => domain_oracle,
        :find_domain_point => find_domain_point,
        :active_set => active_set,
    )
end
