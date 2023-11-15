## TO DO 

## Have a general interface for heuristics 
## so that the user can link in custom heuristics.
"""
    Boscia Heuristic

Interface for heuristics in Boscia.    
`h` is the heuristic function receiving as input ..
`prob` is the probability with which it will be called.        
"""
# Would 'Heuristic' also suffice? Or might we run into Identifer conflicts with other packages?
struct BosciaHeuristic
    h::Function
    prob::Float64
end

BosciaHeuristic() = BosciaHeuristic(x -> nothing, 0.0)

"""
Chooses heuristic by rolling a dice.
"""
function pick_heuristic(heuristic_list)
end
