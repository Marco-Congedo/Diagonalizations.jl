push!(LOAD_PATH,"../src/")
using Documenter, Diagonalizations

makedocs(
   sitename="Diagonalizations",
   authors="Marco Congedo",
   modules=[Diagonalizations],
   pages =
   [
      "index.md",
      "Main Module" => "Diagonalizations.md",
      "Filters" => Any[
         "One dataset (m=1)" => Any[
            "PCA" => "pca.md",
            "Whitening" => "whitening.md",
            "CSP" => "csp.md",
            "CSTP" => "cstp.md",
            "AJD" => "ajd.md",
         ],
         "Two datasets (m=2)" => Any[
            "MCA" => "mca.md",
            "CCA" => "cca.md",
         ],
         "Several datasets (m>2)" => Any[
            "gMCA" => "gmca.md",
            "gCCA" => "gcca.md",
            "mAJD" => "majd.md",
         ],
      "Tools" => "tools.md",   
      ],
   ]
)

deploydocs(
   # root
   # target = "build", # add this folder to .gitignore!
   repo = "github.com/Marco-Congedo/Diagonalizations.jl.git",
   # branch = "gh-pages",
   # osname = "linux",
   # deps = Deps.pip("pygments", "mkdocs"),
   # devbranch = "dev",
   # devurl = "dev",
   # versions = ["stable" => "v^", "v#.#", devurl => devurl],
)
