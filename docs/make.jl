using TVMFFI
using Documenter

DocMeta.setdocmeta!(TVMFFI, :DocTestSetup, :(using TVMFFI);
    recursive = true)

makedocs(;
    modules = [TVMFFI],
    authors = "Gabriel Wu <wuzihua@pku.edu.cn> and contributors",
    repo = "https://github.com/lucifer1004/TVMFFI.jl/blob/{commit}{path}#{line}",
    sitename = "TVMFFI.jl",
    checkdocs = :exports,
    format = Documenter.HTML(;
        prettyurls = get(ENV, "CI", "false") == "true",
        canonical = "https://lucifer1004.github.io/TVMFFI.jl",
        edit_link = "main",
        assets = String[]),
    pages = [
        "Home" => "index.md",
        "API Reference" => "api.md"
    ])

deploydocs(;
    repo = "github.com/lucifer1004/TVMFFI.jl",
    devbranch = "main")
