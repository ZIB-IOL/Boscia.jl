# Add the parent directory to LOAD_PATH so we can import Boscia
# This doesn't modify Project.toml, keeping it clean like FrankWolfe.jl
pushfirst!(LOAD_PATH, joinpath(@__DIR__, ".."))

using Documenter, Boscia
using SparseArrays
using LinearAlgebra

using Literate

EXAMPLE_DIR = joinpath(dirname(@__DIR__), "examples")
DOCS_EXAMPLE_DIR = joinpath(@__DIR__, "src", "examples")
DOCS_REFERENCE_DIR = joinpath(@__DIR__, "src", "reference")

function file_list(dir, extension)
    return filter(file -> endswith(file, extension), sort(readdir(dir)))
end

function literate_directory(jl_dir, md_dir)
    # Remove old markdown files first
    for filename in file_list(md_dir, ".md")
        filepath = joinpath(md_dir, filename)
        rm(filepath)
    end
    
    # Process all .jl files that start with "docs"
    for filename in file_list(jl_dir, ".jl")
        if startswith(filename, "docs")
            filepath = joinpath(jl_dir, filename)
            Literate.markdown(
                filepath,
                md_dir;
                documenter=true,
                flavor=Literate.DocumenterFlavor(),
            )
        end
    end
    return nothing
end

# Generate markdown files from example .jl files
literate_directory(EXAMPLE_DIR, DOCS_EXAMPLE_DIR)

working_dir = @__DIR__

# Function to copy contents of README.md to index.md
function copy_readme_to_index()
    readme_path = joinpath(working_dir, "..", "README.md")
    index_path = joinpath(working_dir, "src", "index.md")
    readme_content = read(readme_path, String)
    return write(index_path, readme_content)
end

# Call the function to update index.md
copy_readme_to_index()

# Generate documentation
makedocs(
    sitename="Boscia.jl",
    modules=[Boscia],
    format=Documenter.HTML(repolink="https://github.com/ZIB-IOL/Boscia.jl.git"),
    pages=[
        "Home" => "index.md",
        "How does it work?" => "basics.md",
        "Examples" => [
            "Network Design Problem" => "examples/docs-01-network-design.md",
            "Graph Isomorphism Problem" => "examples/docs-02-graph-isomorphism.md",
            "Optimal Design of Experiments" => "examples/docs-03-optimal-design.md",
        ],
        "API Reference" => [
            #"reference/0_reference.md",
            "reference/1_algorithms.md",
            "reference/2_blmo_build.md",
            "reference/custom.md",
            "reference/fw_variant.md",
            "reference/utilities.md",
        ],
    ],
    warnonly=true,
)

# Deploy documentation
deploydocs(
    repo="github.com/ZIB-IOL/Boscia.jl.git",
    devbranch="main",
    devurl="dev",
    target="build",
    branch="gh-pages",
    versions=["stable" => "v^", "v#.#"],
    push_preview=true,
)
