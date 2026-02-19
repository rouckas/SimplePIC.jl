module SimplePIC

using MonteCarloCollisions

using Plots
using FFTW
using Random
using Distributions
using StatsBase
using StaticArrays



export m_e, q_e, q_0, amu, k_B, epsilon_0
export NeutralEnsemble, ParticleEnsemble, Particle1d3v, Particle3d3v, Particle0d3v, Particle1d3vE, Particle1d1vE
export Cartesian1D, CylindricalR, BCDirichlet1D, BCPeriodic1D
export Neutrals, Particles, AbstractParticle, Interaction, Interactions, add_interaction!
export load_interaction_lxcat, load_interactions_lxcat, svmax_find!, init_rates!, make_interactions
export init_time, init_monoenergetic, init_thermal
export advance, advance!, energy, scatter

export epsilon_0, e
export PIC, Diagnostic, RhoProbe, EnergyProbe, NxProbe, NvxProbe, EProbe, PSxProbe, UProbe, sample!
export sample, poisson_solve, interpolate, advance, init_leapfrog, solve_init, solve_init_fft, particle_bc

export random_maxwell_v, random_maxwell_vcomponent, random_maxwell_vflux



abstract type AbstractField end
abstract type Field1D <: AbstractField end

struct Field1DCart{T} <: Field1D
    data::Array{T, 1}
    nx::Int64
    xmax::Float64
    dx::Float64
end

abstract type AbstractGeometry end

struct Cartesian1D <: AbstractGeometry
    nx::Int64
    xmax::Float64
    dx::Float64
    dV::Float64
end

struct CylindricalR <: AbstractGeometry
    nr::Int64
    rmin::Float64
    rmax::Float64
    dr::Float64
    dtheta::Float64
    dz::Float64
end

abstract type BoundaryCondition end

struct BCDirichlet1D <: BoundaryCondition
    x::SVector{2, Float64}
end

struct BCPeriodic1D <: BoundaryCondition
end


function remove(particles::ParticleEnsemble, i::Int64)
    n0 = length(particles.coords)
    # i = n0   : just pop
    # i < n0   : c[n0-1] = pop
    p = pop!(particles.coords)
    if i <= length(particles.coords)
        particles.coords[i] = p
    end
end

function insert(particles::ParticleEnsemble, p::AbstractParticle)
    push!(particles.coords, p)
end



mutable struct PIC{dim, ParticleType, GeometryType, BCType, vdim} 
    #where ParticleType <: AbstractParticle where FieldType <: AbstractField where BCType <: BoundaryCondition
    particles::Vector{ParticleEnsemble{ParticleType}}
    interactions::Vector{Interactions{ParticleEnsemble{ParticleType}}}
    nx::Int64
    xmax::Float64
    dx::Float64
    x::Vector{Float64}
    xbins::Vector{Float64}
    dV::Float64
    geo::GeometryType
    BC::BCType
    rho::Array{Float64, dim}
    U::Array{Float64, dim}
    E::Array{SVector{vdim, Float64}, dim}
    rhobg::Float64
    epsilon_0::Float64
end

function PIC(particles::Vector{ParticleEnsemble{ParticleType}}, interactions::Vector{Interactions{ParticleEnsemble{ParticleType}}},
    geo::AbstractGeometry, BC::BCType, epsilon_0::Float64, vdim::Int64) where ParticleType where BCType <: BoundaryCondition
    PIC{1, ParticleType, Cartesian1D, BCType, vdim}(particles,
        interactions,
        geo.nx, 
        geo.xmax, 
        geo.dx, 
        LinRange(0, geo.xmax, geo.nx), 
        [i-0.5 for i in 0:geo.nx]*geo.xmax/(geo.nx-1),
        geo.dV,
        geo,
        BC,
        zeros(geo.nx),
        zeros(geo.nx),
        zeros(SVector{vdim, Float64}, geo.nx),
        0.,
        epsilon_0)
end

include("Sampling.jl")
include("Poisson.jl")


@inline function interpolate_linear(x::SVector{1, Float64}, dx::Float64, y::Vector{Float64})
    SVector(interpolate_linear(x[1], dx::Float64, y::Vector{Float64}))
end

@inline function interpolate_linear(x::Float64, dx::Float64, y::Vector{T}) where T
    xi = x/dx
    xl = trunc(Int, xi)
    w = xi - xl
    y[xl+1]*(1-w) + y[xl+2]*w
end

function interpolate_linear(particles::ParticleEnsemble, dx::Float64, E::Vector{T}) where T
    for p in particles.coords
        p.E = interpolate_linear(p.r[1], dx, E)
    end
end

function interpolate_linear(particles::ParticleEnsemble, dx::Float64, E::Vector{Float64})
    for p in particles.coords
        p.E = SVector(interpolate_linear(p.r[1], dx, E), 0, 0)
    end
end


function advance(particles::ParticleEnsemble, interactions::Interactions, dt::Float64)
    qmdt = particles.q/particles.m*dt
    Pcoll = -expm1( -interactions.rate*dt)
    #println(particles.name, " Pcoll ", Pcoll)
    for p in particles.coords
        p.v += p.E * qmdt
        p.r += SVector(p.v[1]*dt)

        if rand() < Pcoll
                p.v = MonteCarloCollisions.scatter(p.v, particles.m , interactions)
        end
    end
end

function advance(particles::ParticleEnsemble, dt::Float64)
    qmdt = particles.q/particles.m*dt
    for p in particles.coords
        p.v += p.E * qmdt # XXX translate E::Float64 -> SVector{3, Float64} here
        p.r += p.v[[1]] * dt
    end
end


function advance_v(particles::ParticleEnsemble, dt::Float64)
    qmdt = particles.q/particles.m*dt
    for p in particles.coords
        p.v += p.E * qmdt
    end
end

@inline function particle_bc_periodic(x::Float64, xmax::Float64)
    if x >= xmax
        return x%xmax
    elseif x < 0
        x = x%xmax + xmax
        return x < xmax ? x : 0 # to avoid underflow for x=-epsilon; x%xmax + xmax == xmax in float arigthmetic
    else
        return x
    end
end

function particle_bc(particles::ParticleEnsemble, xmax::Float64, BC::BCPeriodic1D)
    for p in particles.coords
        p.r = particle_bc_periodic.(p.r, xmax)
    end
end

function particle_bc(particles::ParticleEnsemble, xmax::Float64, BC::BCDirichlet1D)
    #to_remove = findall(@. particles.x <= 0 || particles.x >= xmax)
    to_remove = Int64[]
    for (i, p) in enumerate(particles.coords)
        if p.r[1] <= 0 || p.r[1] >= xmax
            push!(to_remove, i)
        end
    end
    #=
    if length(to_remove) > 0
        println("removing ", to_remove)
    end
    =#
    for i in reverse(to_remove)
        remove(particles, i)
    end
end

#particle_bc(particles::ParticleEnsemble, xmax::Float64, ) = particle_bc_dirichlet(particles::ParticleEnsemble, xmax::Float64)

function sample(pic::PIC)
    pic.rho .= pic.rhobg
    for p in pic.particles
        pic.rho .+= sample_linear(p, pic.nx, pic.dx).*(p.q/pic.dV)
    end

    if typeof(pic.BC) <: BCPeriodic1D
        pic.rho[1] += pic.rho[end] - pic.rhobg
        pic.rho[end] = pic.rho[1]
    end
end

function poisson_solve(pic::PIC)
    solve!(pic.U, pic.rho, pic.geo, pic.BC, pic.epsilon_0)
    differentiate!(pic.E, pic.U, pic.geo, pic.BC)
    pic.E .*= -1
end

function poisson_solve(pic::PIC, init)
    solve!(pic.U, pic.rho, pic.geo, pic.BC, pic.epsilon_0, init)
    differentiate!(pic.E, pic.U, pic.geo, pic.BC)
    pic.E .*= -1
end

#poisson_solve(pic::PIC) = poisson_solve(pic)
#poisson_solve(pic::PIC, lu) = poisson_solve(pic::PIC, lu)

function interpolate(pic::PIC)
    for p in pic.particles
        interpolate_linear(p, pic.dx, pic.E)
    end
end

function particle_bc(pic::PIC)
    for p in pic.particles
        particle_bc(p, pic.xmax, pic.BC)
    end
end    

function advance(pic::PIC, dt::Float64)
    for (p, inter) in zip(pic.particles, pic.interactions)
        advance(p, inter, dt)
        particle_bc(p, pic.xmax, pic.BC)
    end
end

function init_leapfrog(pic::PIC, dt::Float64)
    sample(pic)
    poisson_solve(pic)
    interpolate(pic)
    for p in pic.particles
        advance_v(p, -dt/2)
        particle_bc(p, pic.xmax, pic.BC)
    end
end

include("Diagnostic.jl")
include("Constants.jl")
end