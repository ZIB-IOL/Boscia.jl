"""
Branch-and-Bound settings.
"""
function BnBSettings(mode::Mode;
    traverse_strategy=Bonobo.BestFirstSearch(),
    branching_strategy=Bonobo.MOST_INFEASIBLE(),
    verbose=false,
    node_limit=Inf,
    time_limit=Inf,
    print_iter=100,
    bnb_callback=nothing,
    no_pruning=mode == HEURISTIC ? true : false,
    ignore_lower_bound= mode == HEURISTIC ? true : false,
    start_solution=nothing,
    use_shadow_set=true
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
Frank-Wolfe settings.
"""
function FWSettings(mode::Mode;
    variant=BPCG(),
    line_search=FrankWolfe.Secant(),
    max_fw_iter=10000,
    fw_timeout=Inf,
    min_fw_iterions=5,
    fw_verbose=false,
    lazy=true,
    lazy_tolerance=2
)
    return Dict(
        :variant => variant,
        :line_search => line_search,
        :max_fw_iter => max_fw_iter,
        :fw_timeout => fw_timeout,
        :min_fw_iterions => min_fw_iterions,
        :fw_verbose => fw_verbose,
        :lazy => lazy,
        :lazy_tolerance => lazy_tolerance,
    )
end

"""
Tolerances.
"""
function tolerances(mode::Mode;
    fw_epsilon=1e-2,
    dual_gap=1e-6,
    rel_dual_gap=1.0e-2,
    dual_gap_decay_factor=0.8,
    min_number_lower=Inf,
    min_node_fw_epsilon=1e-6
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
Postprocessing settings.
"""
function postprocessing(mode::Mode; 
    use_postsolve=true,
    max_iteration_post=10000,
)
    return Dict(
        :use_postsolve => use_postsolve,
        :max_iteration_post => max_iteration_post,
    )
end

"""
Heuristic settings.
"""
function heuristic_settings(mode::Mode;
    custom_heuristics=[Heuristic()],
    post_heuristics_callback=nothing,
    rounding_prob=1.0,
    add_all_solutions=mode == HEURISTIC ? false : true
)
    return Dict(
        :custom_heuristics => custom_heuristics,
        :post_heuristics_callback => post_heuristics_callback,
        :rounding_prob => rounding_prob,
        :add_all_solutions => add_all_solutions,
    )
end

"""
Tightening settings.
"""
function tightening(mode::Mode; 
    dual_tightening=true,
    global_dual_tightening=true,
    strong_convexity=0.0,
    sharpness_constant=0.0,
    sharpness_exponent=Inf,
    propagate_bounds=nothing
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
Non-trivial domain settings.
"""
function domain_settings(mode::Mode;
    domain_oracle=_trivial_domain,
    find_domain_point=_trivial_domain_point,
    active_set::Union{Nothing,FrankWolfe.ActiveSet}=nothing
)
    return Dict(
        :domain_oracle => domain_oracle,
        :find_domain_point => find_domain_point,
        :active_set => active_set,
    )
end