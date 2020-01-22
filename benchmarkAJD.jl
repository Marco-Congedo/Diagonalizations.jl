using LinearAlgebra, Statistics, PosDefManifold, BenchmarkTools

#=
BenchmarkTools.DEFAULT_PARAMETERS.samples = 100
BenchmarkTools.DEFAULT_PARAMETERS.seconds = 30
BenchmarkTools.DEFAULT_PARAMETERS.evals = 1
# overhead: The estimated loop overhead per evaluation in nanoseconds,
# which is automatically subtracted from every sample time measurement.
#The default value is BenchmarkTools.DEFAULT_PARAMETERS.overhead = 0.
#BenchmarkTools.estimate_overhead can be called to determine this value
#empirically (which can then be set as the default value, if you want).
BenchmarkTools.DEFAULT_PARAMETERS.gctrial = true
BenchmarkTools.DEFAULT_PARAMETERS.gcsample = true
=#

function Bench(n, k, evsnr, snr)
    A=Vector{Float64}(undef, 4)
    println(" ------------------------------------------------- ")
    println("Benckmark: n=$n, k=$k; evSNR=$evsnr, snr=$snr")
    A[1] = @belapsed(ajd(C; algorithm=:QNLogLike, sort=false))*1000
    A[2] = @belapsed(ajd(C; algorithm=:LogLike, sort=false))*1000
    A[3] = @belapsed(ajd(C; algorithm=:NoJoB, sort=false))*1000
    A[4] = @belapsed(ajd(C; algorithm=:GAJD, sort=false))*1000
    println(" ")
    println("A: ", A)
    println(" ")
end

n, k, evsnr, snr = 4, 4, 10, 100
C=PosDefManifold.randP(n, k; eigvalsSNR=evsnr, SNR=snr, commuting=false)
Bench(n, k, evsnr, snr)
n, k, evsnr, snr = 4, 40, 10, 100
C=PosDefManifold.randP(n, k; eigvalsSNR=evsnr, SNR=snr, commuting=false)
Bench(n, k, evsnr, snr)
n, k, evsnr, snr = 4, 80, 10, 100
C=PosDefManifold.randP(n, k; eigvalsSNR=evsnr, SNR=snr, commuting=false)
Bench(n, k, evsnr, snr)
n, k, evsnr, snr = 4, 160, 10, 100
C=PosDefManifold.randP(n, k; eigvalsSNR=evsnr, SNR=snr, commuting=false)
Bench(n, k, evsnr, snr)

n, k, evsnr, snr = 40, 4, 10, 100
C=PosDefManifold.randP(n, k; eigvalsSNR=evsnr, SNR=snr, commuting=false)
Bench(n, k, evsnr, snr)
n, k, evsnr, snr = 40, 40, 10, 100
C=PosDefManifold.randP(n, k; eigvalsSNR=evsnr, SNR=snr, commuting=false)
Bench(n, k, evsnr, snr)
n, k, evsnr, snr = 40, 80, 10, 100
C=PosDefManifold.randP(n, k; eigvalsSNR=evsnr, SNR=snr, commuting=false)
Bench(n, k, evsnr, snr)
n, k, evsnr, snr = 40, 160, 10, 100
C=PosDefManifold.randP(n, k; eigvalsSNR=evsnr, SNR=snr, commuting=false)
Bench(n, k, evsnr, snr)

n, k, evsnr, snr = 80, 4, 10, 100
C=PosDefManifold.randP(n, k; eigvalsSNR=evsnr, SNR=snr, commuting=false)
Bench(n, k, evsnr, snr)
n, k, evsnr, snr = 80, 40, 10, 100
C=PosDefManifold.randP(n, k; eigvalsSNR=evsnr, SNR=snr, commuting=false)
Bench(n, k, evsnr, snr)
n, k, evsnr, snr = 80, 80, 10, 100
C=PosDefManifold.randP(n, k; eigvalsSNR=evsnr, SNR=snr, commuting=false)
Bench(n, k, evsnr, snr)
n, k, evsnr, snr = 80, 160, 10, 100
C=PosDefManifold.randP(n, k; eigvalsSNR=evsnr, SNR=snr, commuting=false)
Bench(n, k, evsnr, snr)
