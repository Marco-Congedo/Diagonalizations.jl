#   Main Module of the Diagonalization.jl Package for Julia language
#   v 0.1.0 - last update 1st of January 2020

#   MIT License
#   Copyright (c) 2019,
#   Marco Congedo, CNRS, Grenoble, France:
#   https://sites.google.com/site/marcocongedo/home

# TODO
# - svd full or not?
# - do not overwrite demeaned data

module Diagonalizations

using LinearAlgebra, Statistics, StatsBase, CovarianceEstimation,
      Base.Threads, PosDefManifold

# Special instructions and variables
BLAS.set_num_threads(Sys.CPU_THREADS)


import Base:size, length, eltype, ==, ≈, ≠, ≉

export
   SCM, LShr, LShrLW, NShrLW, # constants for CovarianceEstimator types
   LinearFilters,
   LinearFilter, LF,          # Structs and its alias
   # methods for the above Structs
   size, cut, length, eltype, ==, ≈, ≠, ≉,
   eig,                       # utilities
   nonDiagonality, nonD,
   spForm,
   genDataMatrix,
   pca,                       # linear filters
   whitening,
   csp,
   cstp,
   mca,
   cca,

   ajd,
   gmca,
   gcca,
   majd

   #OJoB,

# Consts
const 📌            = "Diagonalizations.jl"
const titleFont     = "\x1b[35m"
const separatorFont = "\x1b[95m"
const defaultFont   = "\x1b[0m"
const greyFont      = "\x1b[90m"
const SCM=SimpleCovariance()
const LShr=LinearShrinkage
const LShrLW=LShr(ConstantCorrelation())
const NShrLW=AnalyticalNonlinearShrinkage()


# Types
○          = nothing
SorH       = Union{Symmetric, Hermitian}
VecSorH    = Union{Array{Symmetric, 1}, Array{Hermitian, 1}}
SorHo      = Union{SorH, Nothing}
Vec        = Vector{T} where T<:Union{Real, Complex}
Veco       = Union{Vector{T}, Nothing} where T<:Union{Real, Complex}
Mat        = Matrix{T} where T<:Union{Real, Complex}
Mato       = Union{Matrix{T}, Nothing} where T<:Union{Real, Complex}
VecMat     = Vector{Matrix{T}} where T<:Union{Real, Complex}
VecMato    = Union{Vector{Matrix{T}}, Nothing} where T<:Union{Real, Complex}
VecVecMat  = Vector{Vector{Matrix{T}}} where T<:Union{Real, Complex}
Diagonalo  = Union{Diagonal, Nothing}
Float64o   = Union{Float64, Nothing}
Stringo    = Union{String, Nothing}
Tmean      = Union{Int64, AbstractVector, AbstractMatrix, Nothing}
Into       = Union{Int64, Nothing}
Tw         = Union{StatsBase.AbstractWeights, Nothing}
TeVar      = Union{Float64, Int64}
TeVaro     = Union{TeVar, Nothing}


abstract type LinearFilters end

# ===============================================================#
# Structure for all kinds of linear filter
struct LinearFilter <: LinearFilters
   F     :: AbstractArray         # filter
   iF    :: AbstractArray         # left-inverse of the filter
   D     :: Diagonalo   # Diag. Matrix of ev associated with the filter
   eVar  :: Float64o    # explained variance (0, 1]
   ev    :: Veco        # vector of all eigenvalues
   arev  :: Veco        # vector of all accumulated regularized eigenvalues
   name  :: String      # name of the filter

   # Universal constructor of (simple) filters
   function LinearFilter(F::Mat, iF::Mat, D::Diagonalo=○, eVar::Float64o=○,
               ev::Veco=○, arev::Veco=○, name::Stringo=○, check::Bool=true)

     ℹ = 📌*": Simple Linear Filter construction: "
     size(F) ≠ reverse(size(iF))         && throw(DimensionMismatch(ℹ*"
       the size of matrices `F`($(size(F))) and `iF`($(size(iF))) must be one the reverse of the other"))
     (check && iF*F≉ I)                  && throw(ArgumentError(ℹ*"
       matrix `iF` is not a left-inverse of matrix `F`"))
     (D≠○ && size(F, 2) ≠ size(D, 2))    && throw(DimensionMismatch(ℹ*"
       the dimensions of `D`($(size(D, 2))) must be equal to the number of columns of matrix `F`($(size(F, 2)))"))
     (ev≠○ && length(ev)>size(F, 1)) && throw(DimensionMismatch(ℹ*"
       the length of `ev`($(length(ev))) cannot exceed the number of rows of matrix `F`($(size(F, 1)))"))
     (arev≠○ && size(F, 1) < length(arev)) && throw(DimensionMismatch(ℹ*"
       the length of `arev`($(length(arev))) cannot exceed the number of rows of matrix `F`($(size(F, 1)))"))

     new(F, iF, D, eVar, ev, arev, name===○ ? "Linear Filter" : name)
   end

   # minimal constructor given only the filter matrix
   LinearFilter(F::Mat, name::Stringo=○) =
      LinearFilter(F, pinv(F), ○, ○, ○, ○, name===○ ? "Linear Filter" : name, false)


   # Universal constructor of joint linear filters
   function LinearFilter(F::VecMat, iF::VecMat,
                         D::Diagonalo, eVar::Float64o=○, ev::Veco=○,
                         arev::Veco=○, name::Stringo=○, check::Bool=true)

     ℹ = 📌*": Joint Linear Filter construction: "
     length(F) ≠ length(iF) && throw(DimensionMismatch(ℹ*"
       The number of matrices in `F`($(length(F))) and `iF`($(length(iF))) must be the same "))
     for i=1:length(F)
       size(F[i]) == reverse(size(iF[i])) && continue
       size(F[i]) ≠ reverse(size(iF[i]))  && throw(DimensionMismatch(ℹ*"
       the size of matrices `F[$i]`($(size(F[i]))) and `iF[$i]`($(size(iF[i]))) must be one the reverse of the other"))
       # BREAK!
     end
     if check for i=1:length(F) iF*F≉ I   && throw(DimensionMismatch(ℹ*"
       matrix `iF[$i]` is not a left-inverse of matrix `F[$i]`"))
     end end
     n2=minimum(size(f, 2) for f ∈ F)
     (D≠○ && n2 ≠ size(D, 2)) && throw(DimensionMismatch(ℹ*"
       the dimension of `D`($(size(D, 2))) must be equal to the minimum number of columns of the matrices in `F`($n2)"))
     n1=minimum(size(f, 1) for f ∈ F)
     (ev≠○ && length(ev)>n1) && throw(DimensionMismatch(ℹ*"
       the length of `ev`($(length(ev))) cannot exceed the minimum number of rows of the matrices in `F`($n1)"))
     (arev≠○ && length(arev)>n1) && throw(DimensionMismatch(ℹ*"
       the length of `arev` ($(length(arev))) cannot exceed the minimum number of rows of the matrices in `F`($n1)"))
     new(F, iF, D, eVar, ev, arev, name===○ ? "Joint Linear Filter" : name)
   end

   # minimal constructor given only the filters matrix
   LinearFilter(F::VecMat, name::Stringo=○) =
      LinearFilter(F, [pinv(X) for X ∈ F],
                   ○, ○, ○, ○, name===○ ? "Linear Filter" : name, false)
end # Struct LinearFilter

LF=LinearFilter # alias

size(f::LF) = f.F isa Matrix ? size(f.F) : (size(f.F[i]) for i=1:length(f.F))

length(f::LF) = f.F isa Matrix ? 1 : length(f.F)

eltype(f::LF) = f.F isa Matrix ? eltype(f.F) : eltype(f.F[1])



function ==(f::LF, g::LF)
   ((f.F isa Matrix) ≠ (g.F isa Matrix)) && return false
   length(f.F) ≠ length(g.F) && return false
   arevOK=(f.arev≠○ && g.arev≠○) ? f.arev≈g.arev   : true
   evOK  =(f.ev≠○ && g.ev≠○)     ? f.ev≈g.ev       : true
   eVarOK=(f.eVar≠○ && g.eVar≠○) ? f.eVar≈g.eVar   : true
   DOK   =(f.D≠○ && g.D≠○)       ? f.D≈g.D         : true
   if f.F isa Matrix
      FOK   = (spForm(f.iF*g.F)+spForm(g.iF*f.F))<0.01
   else
      FOK   = (mean(spForm(f.iF[i]*g.F[i]) for i=1:length(g.F)) +
               mean(spForm(g.iF[i]*f.F[i]) for i=1:length(f.F))) <0.01
   end
   f.name==g.name && arevOK && evOK && eVarOK && DOK && FOK
end
≈(f::LF, g::LF)= ==(f, g)

≠(f::LF, g::LF)= !==(f, g)
≉(f::LF, g::LF)= !≈(f, g)



# Given a filter, create a reduced filter with dimension p
#### TOUT DOUX: allow passing an eVar value
function cut(f::LinearFilter, p::Int64)
   p=clamp(p, 1, length(f.ev))
   h=1:p
   if f.F isa Matrix
      LF(f.F[:, h], f.iF[h, :], f.D===○ ? ○ : Diagonal(diag(f.D)[h]),
         f.eVar===○ ? ○ : f.eVar[p], f.ev, f.arev, f.name*"($p)", false)
   else
      LF([X[:, h] for X ∈ f.F], [X[h, :] for X ∈ f.iF],
          f.D===○ ? ○ : Diagonal(diag(f.D)[h]), f.eVar===○ ? ○ : f.eVar[p],
          f.ev, f.arev, f.name*"($p)", false)
   end
end
# ===============================================================#


include("tools.jl")
include("pca.jl")
include("csp.jl")
include("cca.jl")
include("../src/optim/JoB.jl")
include("ajd.jl")
include("gcca.jl")



# override the Base.size method
## size(f::LinearFilters) = size(f.p) ## TODO field p in the LF struct
# ++++++++++++++++++++  Show override  +++++++++++++++++++ # (REPL output)
function Base.show(io::IO, ::MIME{Symbol("text/plain")}, f::LinearFilter)
  if f.F isa Matrix
     f.name===○ ? println(io, titleFont, "\n⋱ Linear Filter") :
                  println(io, titleFont, "\n⋱ "*f.name)
     println(io, separatorFont, "⤔   ⤔    ⤔     ⤔      ⤔")
     println(io, separatorFont,".F    ", defaultFont, "matrix{$(eltype(f.F))}$((size(f.F)))")
     println(io, separatorFont,".iF   ", defaultFont, "matrix{$(eltype(f.iF))}$(size(f.iF))")
  else
     f.name===○ ? println(io, titleFont, "\n⋱ Joint Linear Filter") :
                  println(io, titleFont, "\n⋱ "*f.name)
     println(io, separatorFont, "⤕   ⤕    ⤕     ⤕      ⤕")
     println(io, separatorFont,".F    ", defaultFont, "$(length(f.F))-vector of matrices{$(eltype(f.F[1]))}$((size(f.F[1])))")
     println(io, separatorFont,".iF   ", defaultFont, "$(length(f.iF))-vector of matrices{$(eltype(f.iF[1]))}$((size(f.iF[1])))")
  end
  f.D===○ ?    nothing : println(io, separatorFont,".D    ", defaultFont, "diagonal{$(eltype(f.D))}($(size(f.D)))")
  f.eVar===○ ? nothing : println(io, separatorFont,".eVar ", defaultFont, "explained variance=$(round(f.eVar, digits=3))")
  f.ev===○ ?   nothing : println(io, separatorFont,".ev   ", defaultFont, "vector{$(eltype(f.ev))}($(length(f.ev)))")
  f.arev===○ ? nothing : println(io, separatorFont,".arev ", defaultFont, "vector{$(eltype(f.arev))}($(length(f.arev)))")
  println(io, greyFont,"LEGEND:")
  println(io, greyFont,"F=filter, i=inverse, ev=all eigenvalues")
  println(io, greyFont,"D=Diagonal eigenvalues associated with F")
  println(io, greyFont,"ar=accumulated regularized ev")
end

println("\n⭐ "," Welcome to the", titleFont," ",📌," ",defaultFont,"package", " ⭐\n")
@info " "
println(" Your Machine `",gethostname(),"` (",Sys.MACHINE, ")")
println(" runs on kernel ",Sys.KERNEL," with word size ",Sys.WORD_SIZE,".")
println(" CPU  Threads: ", Sys.CPU_THREADS)
println(" Base.Threads: ", "$(Threads.nthreads())")
println(" BLAS Threads: ", "$(Sys.CPU_THREADS)", "\n")

end # module
