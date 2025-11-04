using Documenter
using Boscia

working_dir = @__DIR__

# Function to copy contents of README.md to index.md
function copy_readme_to_index()
    readme_path = joinpath(working_dir, "..", "README.md")  # Adjusted path
    index_path = joinpath(working_dir, "src", "index.md")
    readme_content = read(readme_path, String)
    return write(index_path, readme_content)
end

# Call the functions to update index.md and 1_algorithms.md
copy_readme_to_index()

# Generate documentation
makedocs(
    sitename="Boscia.jl",
    modules=[Boscia],
    format=Documenter.HTML(repolink="https://github.com/ZIB-IOL/Boscia.jl.git"),
    pages=[
        "Home" => "index.md",
        "How does it work?" => "basics.md",
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
