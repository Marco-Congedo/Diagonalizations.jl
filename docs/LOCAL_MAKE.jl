# Build the documenattion locally
# This si useful to check the documentation before deploying

push!(LOAD_PATH,"../src/")
push!(LOAD_PATH, @__DIR__)
using Documenter, DocumenterTools, DocumenterCitations, DocumenterInterLinks
using Diagonalizations

makedocs(
   sitename="PosDefManifold",
   format = Documenter.HTML((prettyurls = false)), # ELIMINATE pretty URL for deploying
   authors="Marco Congedo, CNRS, Grenoble, France",
   modules=[Diagonalizations],
   pages =
   [
      "index.md",
      "Diagonalizations" => "Diagonalizations.md",
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
      ],
      "Tools" => "tools.md",
      "Algorithms" => "algorithms.md",
   ]
)

