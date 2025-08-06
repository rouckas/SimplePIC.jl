function differentiate_dirichlet!(E::Vector{Float64}, U::Vector{Float64}, dx::Float64)
    @views E[2:end-1] .= (U[3:end].-U[1:end-2])./(2*dx)
    E[1] = -(-3*U[1] + 4*U[2] - U[3])/(2*dx)
    E[end] = -(3*U[end] - 4*U[end-1] + U[end-2])/(2*dx)
end

function differentiate!(E::Vector{T}, U::Vector{Float64}, geo::Cartesian1D, BC::BCDirichlet1D) where T
    @views E[2:end-1] .= map(u -> SVector(u, 0., 0.), U[3:end].-U[1:end-2])./(2*geo.dx)
    E[1] = SVector((-3*U[1] + 4*U[2] - U[3])/(2*geo.dx), 0, 0)
    E[end] = SVector((3*U[end] - 4*U[end-1] + U[end-2])/(2*geo.dx), 0, 0)
end

function differentiate!(E::Vector{SVector{1, Float64}}, U::Vector{Float64}, geo::Cartesian1D, BC::BCPeriodic1D)
    @views E[2:end-1] .= map(u -> SVector(u), U[3:end].-U[1:end-2])./(2*geo.dx)
    E[1] = SVector((U[2]-U[end-1])/(2*geo.dx)) # XXX this is not general, only for 1D
    E[end] = SVector(E[1])
end

function differentiate!(E::Vector{Float64}, U::Vector{Float64}, geo::Cartesian1D, BC::BCPeriodic1D)
    @views E[2:end-1] .= (U[3:end].-U[1:end-2])./(2*geo.dx)
    E[1] = (U[2]-U[end-1])/(2*geo.dx)
    E[end] = E[1]
end

using SparseArrays
using LinearAlgebra

# set the default solvers
solve_init(geo::Cartesian1D, BC::BCPeriodic1D) = solve_init_fft(geo)
solve_init(geo::Cartesian1D, BC::BCDirichlet1D) = solve_init_sparse_dirichlet(geo)

solve!(U::Vector{Float64}, rho::Vector{Float64}, geo::Cartesian1D, BC::BCDirichlet1D, epsilon_0::Float64, init) = solve_sparse_dirichlet!(U, rho, geo, BC, epsilon_0, init)
solve!(U::Vector{Float64}, rho::Vector{Float64}, geo::Cartesian1D, BC::BCPeriodic1D, epsilon_0::Float64, init) = solve_fft!(U, rho, geo, BC, epsilon_0, init)

function solve!(U::Vector{Float64}, rho::Vector{Float64}, geo::GeoType, BC::BCType, epsilon_0::Float64) where GeoType <: AbstractGeometry where BCType <: BoundaryCondition
    init = solve_init(geo, BC)
    solve!(U, rho, geo, BC, epsilon_0, init)
end

###########################
#      SPARSE SOLVERS     #
###########################

function solve_init_sparse_dirichlet(geo::Cartesian1D)
    d2x = spdiagm(-1 => ones(geo.nx-3), 0 => fill(-2, geo.nx-2), 1=> ones(geo.nx-3))
    lu!(d2x)
end

function laplace(geo::CylindricalR)
    # Birdsall & Langdon pp 333/336
    # potential fixed on outer boundary
    h = geo.dr
    r = LinRange(geo.rmin, geo.rmax, geo.nr) # 
    left = Array{Float64}(undef, geo.nr-1)
    center = Array{Float64}(undef, geo.nr)
    right = Array{Float64}(undef, geo.nr-1)
    center[1] = -4/h^2
    right[1] = 4/h^2
    for i = 2:(geo.nr-1)
        left[i-1] = 1/h^2 - 1/(2*r[i]*h)
        center[i] = -2/h^2
        right[i] = 1/h^2 + 1/(2*r[i]*h)
    end
    left[geo.nr-1] = 0
    center[geo.nr] = 1

    # idr2 = 1.0/geo.dr^2
    # @. i2rdr = 1.0/(2*r*geo.dr)
    # @. left = (idr2 - i2rdr)
    # center = fill(-2 .* idr2, geo.nx-2)
    # @. right = (idr2 + i2rdr)
    spdiagm(-1 => left, 0 => center, 1=> right)
end

function solve_init_sparse_dirichlet(geo::CylindricalR)
    d2x = laplace(geo)
    println("MMMMMMM")
    display(Matrix(d2x))
    lu!(d2x)
end

function solve_init_sparse_periodic(geo::Cartesian1D)
    d2x = spdiagm(-1 => ones(geo.nx-2), 0 => fill(-2, geo.nx-1), 1=> ones(geo.nx-2), geo.nx-2 => [1])
    d2x[geo.nx-1, geo.nx-2] = 0
    d2x[geo.nx-1, geo.nx-1] = 1

    lu!(d2x)
end

function solve_sparse_dirichlet!(U::Vector{Float64}, rho::Vector{Float64}, geo::Cartesian1D, BC::BCDirichlet1D, epsilon_0::Float64, lu)
    U[1] = BC.x[1]
    U[end] = BC.x[2]
    rhs = rho[2:end-1]*(-geo.dx^2/epsilon_0)
    rhs[1] -= U[1]
    rhs[end] -= U[end]
    U[2:end-1] = lu\rhs
    U
end

function solve_sparse_dirichlet!(U::Vector{Float64}, rho::Vector{Float64}, geo::CylindricalR, BC::BCDirichlet1D, epsilon_0::Float64, lu)
    rhs = rho .* (-1.0/epsilon_0)
    rhs[end] = BC.x[2]
    @show rhs
    U[:] = lu\rhs
    U
end


function solve_sparse_periodic!(U::Vector{Float64}, rho::Vector{Float64}, geo::Cartesian1D, BC::BCPeriodic1D, epsilon_0::Float64, lu)
    rhs = rho[1:end-1]*(-geo.dx^2/epsilon_0)
    rhs[end] = 0
    U[1:end-1] = lu\rhs
    U[end] = U[1]
    U
end

###########################
#       DST SOLVERS       #
###########################

function solve_init_dst(geo::Cartesian1D)
    dstplan = FFTW.plan_r2r(zeros(geo.nx-2) , FFTW.RODFT00)
    k = 1:geo.nx-2
    kappa = 1. ./ (2 .* cos.(k .* (pi/(geo.nx-1))) .- 2).* ((-geo.dx^2) / (2*(geo.nx-1)))
    (dstplan, kappa)
end

function solve_dst!(U::Vector{Float64}, rho::Vector{Float64}, geo::Cartesian1D, BC::BCPeriodic1D, epsilon_0::Float64)
    rho_k = FFTW.r2r(rho[2:geo.nx-1] , FFTW.RODFT00) 
    k = 1:geo.nx-2 # https://juliamath.github.io/FFTW.jl/latest/fft.html
    #lambda_k = 2 .* cos.(k .* (pi/(nx-1))) .- 2
    U_k = rho_k ./ (2 .* cos.(k .* (pi/(geo.nx-1))) .- 2).* (-geo.dx^2/epsilon_0)
    U[2:geo.nx-1] .= FFTW.r2r(U_k, FFTW.RODFT00) ./ (2*(geo.nx-1))
end

function solve_dst!(U::Vector{Float64}, rho::Vector{Float64}, geo::Cartesian1D, BC::BCPeriodic1D, epsilon_0::Float64, plan::T) where T <:FFTW.r2rFFTWPlan
    rho_k = plan*rho[2:geo.nx-1] 
    k = 1:geo.nx-2# https://juliamath.github.io/FFTW.jl/latest/fft.html
    #lambda_k = 2 .* cos.(k .* (pi/(nx-1))) .- 2
    U_k = rho_k ./ (2 .* cos.(k .* (pi/(geo.nx-1))) .- 2).* (-geo.dx^2/epsilon_0)
    U[2:geo.nx-1] .= plan*U_k ./ (2*(geo.nx-1))
end

function solve_dst!(U::Vector{Float64}, rho::Vector{Float64}, geo::Cartesian1D, BC::BCPeriodic1D, epsilon_0::Float64, plan::Tuple{Any, Any})
    dstplan, kappa = plan
    rho_k = dstplan*rho[2:geo.nx-1] 
    #k = 1:nx-2# https://juliamath.github.io/FFTW.jl/latest/fft.html
    #lambda_k = 2 .* cos.(k .* (pi/(nx-1))) .- 2
    U_k = rho_k .* kappa ./ epsilon_0
    U[2:geo.nx-1] .= dstplan*U_k
end


###########################
#       FFT SOLVERS       #
###########################

function solve_fft!(U::Vector{Float64}, rho::Vector{Float64}, geo::Cartesian1D, BC::BCPeriodic1D, epsilon_0::Float64)
    # Birdsall Langdon pp 18
    rho_k = fft(@view rho[1:end-1])
    kx = fftfreq(geo.nx-1)*(geo.nx-1).* (2*pi/geo.xmax)
    lambda_k = .-(kx .* sinc.(kx .* (geo.dx / (2*pi)))) .^ 2
    U_k = rho_k ./ lambda_k .* (-1/epsilon_0)
    U_k[1] = 0.
    U[1:end-1] .= real.(ifft(U_k))
    U[end] = U[1]
    return
end

function solve_init_fft(geo::Cartesian1D)
    fftplan = plan_fft(zeros(geo.nx-1))
    ifftplan = plan_ifft(zeros(geo.nx-1))
    kx = fftfreq(geo.nx-1)*(geo.nx-1).* (2*pi/geo.xmax)
    K2 = 1 ./ (kx .* sinc.(kx .* (geo.dx / (2*pi)))) .^ 2 

    (fftplan, ifftplan, K2)
end

function solve_fft!(U::Vector{Float64}, rho::Vector{Float64}, geo::Cartesian1D, BC::BCPeriodic1D, epsilon_0::Float64, plan::Tuple{Any, Any, Any})
    # Birdsall Langdon pp 18
    fftplan, ifftplan, K2 = plan
    
    #rho_k = fft(@view rho[1:end-1])
    U_k = fftplan * (@view rho[1:end-1])

    #kx = fftfreq(nx-1)*(nx-1).* (2*pi/xmax)
    #lambda_k = .-(kx .* sinc.(kx .* (dx / (2*pi)))) .^ 2
    #U_k = rho_k ./ lambda_k .* (-1/epsilon_0)
    U_k .*= K2 ./ epsilon_0
    U_k[1] = 0.
    U[1:end-1] .= real.(ifftplan*U_k)
    U[end] = U[1]
    return
end

