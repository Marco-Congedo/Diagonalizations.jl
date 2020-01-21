using LinearAlgebra, PosDefManifold, BenchmarkTools

BenchmarkTools.DEFAULT_PARAMETERS.samples = 100
BenchmarkTools.DEFAULT_PARAMETERS.seconds = 600
BenchmarkTools.DEFAULT_PARAMETERS.evals = 10
# overhead: The estimated loop overhead per evaluation in nanoseconds,
# which is automatically subtracted from every sample time measurement.
#The default value is BenchmarkTools.DEFAULT_PARAMETERS.overhead = 0.
#BenchmarkTools.estimate_overhead can be called to determine this value
#empirically (which can then be set as the default value, if you want).
BenchmarkTools.DEFAULT_PARAMETERS.gctrial = true
BenchmarkTools.DEFAULT_PARAMETERS.gcsample = true


n, k = 3, 10
C=PosDefManifold.randP(n, k; eigvalsSNR=10, SNR=2, commuting=false)
@btime(sort=false)
