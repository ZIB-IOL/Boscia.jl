using Boscia
using Documenter

DocMeta.setdocmeta!(Boscia, :DocTestSetup, :(using Boscia); recursive=true)

makedocs(;
    modules=[Boscia],
    authors="Saurabh Srivastava",
    sitename="Boscia.jl",
    format=Documenter.HTML(;
        canonical="https://saurabhintoml.github.io/Boscia.jl",
        edit_link="master",
        assets=String[],
    ),
    pages=[
        "Home" => "index.md",
    ],
)

deploydocs(;
    repo="github.com/saurabhintoml/Boscia.jl",
    devbranch="master",
)
