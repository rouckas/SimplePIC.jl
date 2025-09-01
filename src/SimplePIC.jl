module DirichletPIC

include("MonteCarloCollisions.jl")
using .MonteCarloCollisions

import Interpolations as itp
import LinearAlgebra as linalg
using Plots
using FFTW
using Random
using Distributions
using StatsBase
using StaticArrays

const epsilon_0 = 8.8541878128e-12
const e = 1.60217662e-19
const m_e = 9.10938356e-31
const m_p = 1.6726219e-27

export m_e, q_e, amu, k_B
export NeutralEnsemble, ParticleEnsemble, Particle1d3v, Particle3d3v, Particle0d3v, Particle1d3vE, Particle1d1vE
export Cartesian1D, CylindricalR, BCDirichlet1D, BCPeriodic1D
export Neutrals, Particles, AbstractParticle, Interaction, Interactions, add_interaction!
export load_interaction_lxcat, load_interactions_lxcat, svmax_find!, init_rates!, make_interactions
export init_time, init_monoenergetic, init_thermal
export advance, advance!, energy, scatter
export Poisson

export epsilon_0, e
export PIC, Diagnostic, RhoProbe, EnergyProbe, NxProbe, NvxProbe, EProbe, PSxProbe, UProbe, TProbe, BProbe, QLProbe, QRProbe, JProbe, QProbe, sample!
export sample, poisson_solve, interpolate, advance, advance_v, init_leapfrog, solve_init, solve_init_fft, particle_bc
export inject
export random_maxwell_vflux, random_maxwell_v, random
export Circuit, interpolate_current, advance_external, maxwell_solve, advance_current, advance_v_all


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

mutable struct Circuit
    R::Float64
    L::Float64
    C::Float64
    Q::Float64
    Js::Vector{Float64}
    Uext::Function
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

function inject(particles::ParticleEnsemble, nparticles::Int, T::Float64, xmax::Float64, t::Float64, dt::Float64)
    for _ in 1:nparticles
        vth = sqrt(2*k_B*T/particles.m)
        v = SVector{3, Float64}(-random_maxwell_vflux(vth), random_maxwell_vcomponent(vth), random_maxwell_vcomponent(vth))
        x = SVector{1, Float64}(xmax+dt*v[1]*rand())
        E = SVector{3, Float64}(0, 0, 0)
        B = SVector{3, Float64}(0, 0, 0)
        tau = Inf
        p = Particle1d3vE(x, v, E, B, t, tau)
        push!(particles.coords, p)
    end
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
    E::Array{MVector{vdim, Float64}, dim}
    rhobg::Float64
    epsilon_0::Float64
    QL::Float64
    QR::Float64
    J_minus::Array{SVector{vdim, Float64}, dim}
    J_plus::Array{SVector{vdim, Float64}, dim}
    J::Array{SVector{vdim, Float64}, dim}
    B::Array{MVector{vdim, Float64}, dim}
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
        zeros(MVector{vdim, Float64}, geo.nx),
        0.,
        epsilon_0, 0.0, 0.0, zeros(SVector{vdim, Float64}, geo.nx), zeros(SVector{vdim, Float64}, geo.nx), zeros(SVector{vdim, Float64}, geo.nx), zeros(MVector{vdim, Float64}, geo.nx))
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

#=
function interpolate_current(pic::PIC, dt::Float64)
    J = [SVector{3, Float64}(0.0, 0.0, 0.0) for i in 1:pic.nx]
    for parts in pic.particles
        for p in parts.coords
            xi1 = p.r[1]/pic.dx
            xl1 = trunc(Int, xi1)
            xi2 = (p.r[1]+p.v[1]*dt)/pic.dx
            xl2 = trunc(Int, xi2)
            w1 = xi1 - xl1
            w2 = xi2 - xl2
            J[xl1+1] = J[xl1+1] + p.q*p.v*1/2*((1-w1)+(1-w2))
            J[xl2+1] = J[xl2+1] + p.q*p.v*1/2*(1 - ((1-w1)+(1-w2)))
        end
    end
    return J
end
=#

function interpolate_current(particles::ParticleEnsemble, dx::Float64, nx::Int64)
    J = zeros(SVector{3, Float64}, nx)
        for p in particles.coords
            xi = p.r[1]/dx
            xl = floor(Int, xi)
            w = xi - xl
            J[xl+1] = J[xl+1] + particles.q*p.v*1/2*(1-w)
            J[xl+2] = J[xl+2] + particles.q*p.v*1/2*w
        end
    return J
end

function advance_position(particles::ParticleEnsemble, interactions::Interactions, dt::Float64)
    Pcoll = -expm1( -interactions.rate*dt)
    #println(particles.name, " Pcoll ", Pcoll)
    for p in particles.coords
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

#=
function advance_v(particles::ParticleEnsemble, dt::Float64)
    qmdt = particles.q/particles.m*dt/2
    for p in particles.coords
        t = p.B*qmdt
        s = 2*t/(1+t.^2)
        vminus = p.v + p.E*qmdt
        vprime = vminus + linalg.cross(vminus, t)
        vplus = vminus + linalg.cross(vprime, s)
        p.v = vplus + p.E*qmdt
    end
end
=#

function advance_v(particles::ParticleEnsemble, dt::Float64)
    qmdt = particles.q / particles.m * dt / 2
    for p in particles.coords
        # half acceleration from E
        vminus = p.v + p.E * qmdt
        
        t = p.B * qmdt
        s = 2 * t / (1 + dot(t, t))
        
        vprime = vminus + cross(vminus, t)
        vplus  = vminus + cross(vprime, s)
        
        p.v = vplus + p.E * qmdt
    end
end

function advance_v_all(pic::PIC, dt::Float64)
    for p in pic.particles
        advance_v(p, dt)
    end
end

@inline function particle_bc_periodic(x::Float64, xmax::Float64)
    if x >= xmax
        return x%xmax
    elseif x < 0
        x = x%xmax + xmax
        return x < xmax ? x : 0 # to avoid underflow for x=-epsilon; x%xmax + xmax == xmax in float arithmetic
    else
        return x
    end
end

function particle_bc(particles::ParticleEnsemble, xmax::Float64, BC::BCPeriodic1D)
    for p in particles.coords
        p.r = particle_bc_periodic.(p.r, xmax)
    end
end

function particle_bc(particles::ParticleEnsemble, pic::PIC, xmax::Float64, BC::BCDirichlet1D)
    #to_remove = findall(@. particles.x <= 0 || particles.x >= xmax)
    to_remove = Int64[]
    for (i, p) in enumerate(particles.coords)
        #=if p.r[1] <= 0 || p.r[1] >= xmax
            push!(to_remove, i)
        end =#
        if p.r[1] <= 0
            push!(to_remove, i)
            pic.QL += particles.q
        end
        if p.r[1] >= xmax
            push!(to_remove, i)
            pic.QR += particles.q
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

function maxwell_solve(pic::PIC, dt::Float64)
    E = deepcopy(pic.E)
    B = deepcopy(pic.B)
    for j in 2:length(pic.E)-1
        E[j+1][2] = E[j][2] - (pic.J[j+1][2] + 2*pic.J[j][2] + pic.J[j-1][2])*dt/4
        B[j+1][3] = B[j][3] - (pic.J[j+1][2] - pic.J[j-1][2])*dt/4
    end
    pic.E = E
    pic.B = B
end

#poisson_solve(pic::PIC) = poisson_solve(pic)
#poisson_solve(pic::PIC, lu) = poisson_solve(pic::PIC, lu)

function interpolate(pic::PIC)
    for p in pic.particles
        interpolate_linear(p, pic.dx, pic.E)
        interpolate_linear(p, pic.dx, pic.B)
    end
end

function particle_bc(pic::PIC)
    for p in pic.particles
        particle_bc(p, pic, pic.xmax, pic.BC)
    end
    #return deltaleft, deltaright
end    

function advance_current(pic::PIC, dt::Float64)
    for (parts, inter) in zip(pic.particles, pic.interactions)
        pic.J_minus = interpolate_current(parts, pic.dx, pic.nx)
        advance_position(parts, inter, dt)
        particle_bc(parts, pic, pic.xmax, pic.BC)
        pic.J_plus = interpolate_current(parts, pic.dx, pic.nx)
        pic.J .= (pic.J_minus .+ pic.J_plus) ./ 2
    end
end

#=
function advance(pic::PIC, dt::Float64, circuit::Circuit, t::Float64)
    for (p, inter) in zip(pic.particles, pic.interactions)
        advance(p, inter, dt)
        particle_bc(p, pic, pic.xmax, pic.BC)
    end
    #return deltaleft, deltaright
end
=#

function advance(pic::PIC, dt::Float64)
    for (p, inter) in zip(pic.particles, pic.interactions)
        advance(p, inter, dt)
        particle_bc(p, pic, pic.xmax, pic.BC)
    end
    #return deltaleft, deltaright
end

function advance_external(pic::PIC, circuit::Circuit, t::Float64, dt::Float64)
    A = pic.dV/pic.xmax
    Cprime = epsilon_0*A/pic.xmax
    U = epsilon_0*(pic.QL - pic.QR)/(2*A)
    if length(circuit.Js) < 2
        push!(circuit.Js, 0.0)
        push!(circuit.Js, 0.0)
    end
    I = (U - circuit.Uext(t) - circuit.Q/circuit.C + circuit.L/dt*(3*circuit.Js[end] - circuit.Js[end - 1]) + circuit.R/2*circuit.Js[end - 1])/(3/2*circuit.R + 2*circuit.L/dt + dt*(1/Cprime + 1/circuit.C))
    push!(circuit.Js, I)
    circuit.Q += I*dt
    pic.QL-=I*dt
    pic.QR+=I*dt
    U = epsilon_0*(pic.QL - pic.QR)/(2*A)
    pic.BC = BCDirichlet1D([U, 0])
end

function init_leapfrog(pic::PIC, dt::Float64)
    sample(pic)
    poisson_solve(pic)
    maxwell_solve(pic, dt)
    interpolate(pic)
    for p in pic.particles
        advance_v(p, -dt/2)
        particle_bc(p, pic, pic.xmax, pic.BC)
    end
    #return deltaleft, deltaright
end


include("Diagnostic.jl")

end