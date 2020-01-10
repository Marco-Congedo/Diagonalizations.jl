#   Unit "JoB.jl" of the Diagonalization.jl Package for Julia language
#
#   MIT License
#   Copyright (c) 2020,
#   Marco Congedo°, Ronald Phlypo, CNRS, UGA, Grenoble-INP, France
#   ° https://sites.google.com/site/marcocongedo/home

# ? CONTENTS :

# """
# """

using LinearAlgebra, PosDefManifold

function qnLogLike(C;
                   B0           = nothing,
                   max_iter     = 10000,
                   tol          = 0.,
                   lambda_min   = 1e-4, #1e-4
                   max_ls_tries = 10,
                   verbose      = false)

    transform_set(M, D) = [Hermitian(M*d*M') for d ∈ D]

    loss(B, D) = -(logabsdet(B)[1]) + 0.5 * sum(mean(log, [Diagonal(d) for d∈D]))

    function linesearch(D, B, direction, current_loss, n_ls_tries)
       step = 1.
       current_loss===nothing ? current_loss = loss(B, D) : nothing

       success = false
       new_loss=0.
       for n=1:n_ls_tries
           M = (step * direction) + I
           new_D = transform_set(M, D)
           new_B = M * B
           new_loss = loss(new_B, new_D)
           if new_loss < current_loss
               success = true
               break
           end
           step /= 2.
       end

       return success, new_D, new_B, new_loss, step * direction
    end

    # function qnLogLike
    type=eltype(C[1])
    tol==0. ? tolerance = √eps(real(type)) : tolerance = tol
    n, p = length(C), size(C[1], 1)
    B0===nothing ? B = invsqrt(mean(C)) : B=B0
    new_D=similar(C)
    new_B=similar(B)

    D = transform_set(B, C)
    current_loss = nothing

    if verbose
        println("Running quasi-Newton for joint diagonalization")
    end

    for t=1:max_iter
        diagonals = [Diagonal(d) for d∈D]

        # Gradient
        G = mean(d/dg for (d, dg) ∈ zip(D, diagonals)) - I
        g_norm = norm(G)
        if g_norm < tolerance
            break
        end

        # Hessian coefficients
        H = mean(diag(dg)'./diag(dg) for dg ∈ diagonals)

        # Quasi-Newton's direction
        det = (H .* H') .- 1.
        replace!(x -> x<lambda_min ? lambda_min : x, det)  # Regularize
        #direction = @. -(G * H' - G') / det
        direction =  -(G .* H' - G') ./ det
        # direction =  -G # use gradient only

        # Line search
        success, new_D, new_B, new_loss, direction =
            linesearch(D, B, direction, current_loss, max_ls_tries)
        D = new_D
        B = new_B
        current_loss = new_loss

        # Monitoring
        if verbose
            println(t, ", ", current_loss, ", ", g_norm)
            println("success ", success)
        end
    end # for t

    return B
end


# test
using PosDefManifold, LinearAlgebra

k=5

Cset=randP(k, 5; eigvalsSNR=Inf, commuting=true)
B=qnLogLike(Cset; B0=nothing, verbose=true)


ac=ajd(Cset; algorithm=:NoJoB, simple=true, verbose=true)


Dset=[Diagonal(C) for C in Cset]
A=randn(k, k)
Cset=HermitianVector([Hermitian(A*D*A') for D in Dset])

B=qnLogLike(Cset; B0=pinv(A)+randn(k, k)*0.01, verbose=true)


Cset=randP(10, 20; eigvalsSNR=10, SNR=2, commuting=false)
B=qnLogLike(Cset; B0=nothing, verbose=true)

using Plots

CMax=maximum(maximum(abs.(C)) for C ∈ Cset);
 h1 = heatmap(Cset[1], clim=(-CMax, CMax), title="C1", yflip=true, c=:bluesreds);
 h2 = heatmap(Cset[2], clim=(-CMax, CMax), title="C2", yflip=true, c=:bluesreds);
 h3 = heatmap(Cset[3], clim=(-CMax, CMax), title="C3", yflip=true, c=:bluesreds);
 h4 = heatmap(Cset[4], clim=(-CMax, CMax), title="C4", yflip=true, c=:bluesreds);
 📈=plot(h1, h2, h3, h4, size=(700,400))
# savefig(📈, homedir()*"\\Documents\\Code\\julia\\Diagonalizations\\docs\\src\\assets\\FigAJD1.png")

Dset=[B*C*B' for C ∈ Cset];
# Dset=[ac.F'*C*ac.F for C ∈ Cset];
 DMax=maximum(maximum(abs.(D)) for D ∈ Dset);
 h5 = heatmap(Dset[1], clim=(-DMax, DMax), title="F'*C1*F", yflip=true, c=:bluesreds);
 h6 = heatmap(Dset[2], clim=(-DMax, DMax), title="F'*C2*F", yflip=true, c=:bluesreds);
 h7 = heatmap(Dset[3], clim=(-DMax, DMax), title="F'*C3*F", yflip=true, c=:bluesreds);
 h8 = heatmap(Dset[4], clim=(-DMax, DMax), title="F'*C4*F", yflip=true, c=:bluesreds);
 📉=plot(h5, h6, h7, h8, size=(700,400))
