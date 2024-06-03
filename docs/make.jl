using Pkg
pkg"activate .. "


using Boscia
using Documenter
using SparseArrays
using LinearAlgebra

using Literate, Test

# Activate the project environment
project_path = joinpath(dirname(@__DIR__))  # Adjust this path if necessary
Pkg.activate(project_path)

# Ensure all dependencies are installed
Pkg.instantiate()

EXAMPLE_DIR = joinpath(dirname(@__DIR__), "examples")
DOCS_EXAMPLE_DIR = joinpath(@__DIR__, "src", "examples")
DOCS_REFERENCE_DIR = joinpath(@__DIR__, "src", "reference")

function file_list(dir, extension)
    return filter(file -> endswith(file, extension), sort(readdir(dir)))
end

# includes plot_utils to the example file before running it
function include_utils(content)
    return """
    import Boscia ; include(joinpath(dirname(pathof(Boscia)), "../examples/plot_utils.jl")) # hide
    """ * content
end

function literate_directory(jl_dir, md_dir)
    for filename in file_list(md_dir, ".md")
        filepath = joinpath(md_dir, filename)
        rm(filepath)
    end
    for filename in file_list(jl_dir, ".jl")
        filepath = joinpath(jl_dir, filename)
        # `include` the file to test it before `#src` lines are removed. It is
        # in a testset to isolate local variables between files.
        if startswith(filename, "docs")
            Literate.markdown(
                filepath, md_dir;
                documenter=true, flavor=Literate.DocumenterFlavor(), preprocess=include_utils,
            )
        end
    end
    return nothing
end

literate_directory(EXAMPLE_DIR, DOCS_EXAMPLE_DIR)
cp(joinpath(EXAMPLE_DIR, "plot_utils.jl"), joinpath(DOCS_EXAMPLE_DIR, "plot_utils.jl") , force = true)

ENV["GKSwstype"] = "100"

generated_path = joinpath(@__DIR__, "src")
base_url = "https://github.com/saurabhintoml/Boscia.jl/"
isdir(generated_path) || mkdir(generated_path)

open(joinpath(generated_path, "contributing.md"), "w") do io
    # Point to source license file
    println(
        io,
        """
        ```@meta
        EditURL = "$(base_url)CONTRIBUTING.md"
        ```
        """,
    )
    # Write the contents out below the meta block
    for line in eachline(joinpath(dirname(@__DIR__), "CONTRIBUTING.md"))
        println(io, line)
    end
end

open(joinpath(generated_path, "basics.md"), "w") do io
    # Point to source license file
    println(
        io,
        """
        ```@meta
        EditURL = "$(base_url)BASICS.md"
        ```
        """,
    )
    # Write the contents out below the meta block
    for line in eachline(joinpath(dirname(@__DIR__), "BASICS.md"))
        println(io, line)
    end
end

open(joinpath(generated_path, "Examples.md"), "w") do io
    # Point to source license file
    println(
        io,
        """
        ```@meta
        EditURL = "$(base_url)EXAMPLES.md"
        ```
        """,
    )
    # Write the contents out below the meta block
    for line in eachline(joinpath(dirname(@__DIR__), "EXAMPLES.md"))
        println(io, line)
    end
end

open(joinpath(generated_path, "index.md"), "w") do io
    # Point to source license file
    println(
        io,
        """
        ```@meta
        EditURL = "$(base_url)README.md"
        ```
        """,
    )
    # Write the contents out below the meta block
    for line in eachline(joinpath(dirname(@__DIR__), "README.md"))
        println(io, line)
    end
end

makedocs(;
    modules=[Boscia],
    authors="Saurabh Srivastava",
    repo = "https://saurabhintoml.github.io/Boscia.jl/blob/{commit}{path}#L{line}" ,
    sitename="Boscia.jl",
    format=Documenter.HTML(;
        prettyurls = get(ENV , "CI" , nothing) == "true" ,
        repolink ="https://saurabhintoml.github.io/Boscia.jl",
        edit_link="master",
        assets=String[],
    ),
    pages=[
        "Home" => "index.md",
        "How Does it Work" => "basics.md" ,
       
     
        "Examples" => "examples.md",
        "API reference" =>
            [joinpath("reference", f) for f in file_list(DOCS_REFERENCE_DIR, ".md")],
        "Contributing" => "contributing.md",
    ],
)

deploydocs(;
    repo="github.com/saurabhintoml/Boscia.jl",
    push_preview = true , 
    devbranch = "master" )
    
