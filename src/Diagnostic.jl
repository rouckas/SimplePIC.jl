
abstract type AbstractProbe end

#***** CHARGE DENSITY PROBE *****

struct RhoProbe <: AbstractProbe
    xrange::Tuple{Float64, Float64}
    ts::Vector{Float64}
    rhos::Vector{Vector{Float64}}
end

RhoProbe(geo::Cartesian1D) = RhoProbe((0, geo.xmax), [], [])

function sample!(probe::RhoProbe, pic::PIC, t::Float64)
    push!(probe.ts, t)
    push!(probe.rhos, deepcopy(pic.rho))
end

function plt(probe::RhoProbe)
    array = stack(probe.rhos)
    x = LinRange(probe.xrange..., length(probe.rhos[1]))
    y = probe.ts
    lim = maximum(abs.(array))
    heatmap(y, x, array, yflip = true, legend=false, 
            title="Rho", titlefont=font(10), c=:bwr, clim=(-lim, lim))
end

#***** ENERGY PROBE *****

struct EnergyProbe <: AbstractProbe
    dt::Float64
    ensemble_id::Int64
    ts::Vector{Float64}
    Ekin::Vector{Float64}
    Edrift::Vector{Float64}
    Ethermal::Vector{Float64}
    Epot::Vector{Float64}
end

EnergyProbe(dt::Float64, ensemble_id::Int64) = EnergyProbe(dt, ensemble_id, [], [], [], [], [])

function sample!(probe::EnergyProbe, pic::PIC, t::Float64)
    # for leap frog integrator, we need to push dt/2 forward
    particles = pic.particles[probe.ensemble_id]
    nparticles = size(particles.coords, 1)
    E = 0.
    V = 0.
    vmean = zero(eltype(particles.coords)).v
    qmdt = particles.q/particles.m*probe.dt/2
    for p in particles.coords
        v = p.v  + p.E * qmdt
        vmean += v
        E += 0.5*particles.m*sum(v.^2)

        U = interpolate_linear(p.r[1], pic.geo.dx, pic.U)
        V += 0.5*U*particles.q
    end
    vmean /= nparticles
    Edrift = 0.5*particles.m*sum(vmean.^2)*nparticles
    push!(probe.ts, t)
    push!(probe.Ekin, E)
    push!(probe.Edrift, Edrift)
    push!(probe.Ethermal, E - Edrift)
    push!(probe.Epot, V)
end

function plt(probe::EnergyProbe)
    #array = stack(probe.rhos)
    plot(probe.ts, probe.Ekin,
        title="Energy components", label="kinetic", titlefont=font(10))
    plot!(probe.ts, probe.Epot,
        label="potential", titlefont=font(10))
    plot!(probe.ts, probe.Epot .+ probe.Ekin,
        label="total", titlefont=font(10))
    plot!(probe.ts, probe.Edrift,
        label="drift", titlefont=font(10))
    plot!(probe.ts, probe.Ethermal,
        label="chaotic", titlefont=font(10))
end

struct UProbe <: AbstractProbe
    xrange::Tuple{Float64, Float64}
    ts::Vector{Float64}
    Us::Vector{Vector{Float64}}
end

UProbe(geo::Cartesian1D) = UProbe((0, geo.xmax), [], [])

function sample!(probe::UProbe, pic::PIC, t::Float64)
    push!(probe.ts, t)
    push!(probe.Us, deepcopy(pic.U))
end

function plt(probe::UProbe)
    array = stack(probe.Us)
    x = LinRange(probe.xrange..., length(probe.Us[1]))
    y = probe.ts
    lim = maximum(abs.(array))
    heatmap(y, x, array, yflip = true, legend=false, 
            title="U", titlefont=font(10), c=:bwr, clim=(-lim, lim))
end


#***** ELECTRIC FIELD PROBE *****

struct EProbe <: AbstractProbe
    xrange::Tuple{Float64, Float64}
    ts::Vector{Float64}
    Es::Vector{Vector{Float64}}
end

EProbe(geo::Cartesian1D) = EProbe((0, geo.xmax), [], [])

function sample!(probe::EProbe, pic::PIC, t::Float64)
    push!(probe.ts, t)
    push!(probe.Es, map(first, pic.E))
end

function plt(probe::EProbe)
    array = stack(probe.Es)
    x = LinRange(probe.xrange..., length(probe.Es[1]))
    y = probe.ts
    lim = maximum(abs.(array))
    heatmap(y, x, array, yflip = true, legend=false, 
            title="E", titlefont=font(10), c=:bwr, clim=(-lim, lim))
end

#***** NUMBER DENSITY PROBE *****

struct NxProbe <: AbstractProbe
    xrange::Tuple{Float64, Float64}
    nx::Int64
    dx::Float64
    ensemble_id::Int64
    ts::Vector{Float64}
    Nx::Vector{Vector{Float64}}
end

NxProbe(geo::Cartesian1D, ensemble_id::Int64) = NxProbe((0, geo.xmax), geo.nx, geo.xmax/(geo.nx-1), ensemble_id, [], [])

function sample!(probe::NxProbe, pic::PIC, t::Float64)
    push!(probe.ts, t)
    push!(probe.Nx, sample_linear(pic.particles[probe.ensemble_id], probe.nx, probe.dx))
end

function plt(probe::NxProbe)
    array = stack(probe.Nx)
    x = LinRange(probe.xrange..., probe.nx)
    y = probe.ts
    lim = maximum(abs.(array))
    heatmap(y, x, array, yflip = true, legend=false, 
            title="Nx", titlefont=font(10), c=:bwr, clim=(-lim, lim))
end

#***** Vx DISTRIBUTION PROBE *****

struct NvxProbe <: AbstractProbe
    vxrange::Tuple{Float64, Float64}
    nvx::Int64
    dvx::Float64
    ensemble_id::Int64
    ts::Vector{Float64}
    Nvx::Vector{Vector{Float64}}
end

NvxProbe(vxmax::Float64, nvx::Int64, ensemble_id::Int64) = NvxProbe((-vxmax, vxmax), nvx, (2*vxmax)/(nvx-1), ensemble_id, [], [])

function sample!(probe::NvxProbe, pic::PIC, t::Float64)
    push!(probe.ts, t)
    vx = [p.v[1] for p in pic.particles[probe.ensemble_id].coords]
    #println("sampling VX: ", vx)
    push!(probe.Nvx, sample_linear(vx, probe.vxrange..., probe.nvx, probe.dvx))
end

function plt(probe::NvxProbe)
    array = stack(probe.Nvx)
    x = LinRange(probe.vxrange..., probe.nvx)
    y = probe.ts
    lim = maximum(abs.(array))
    heatmap(y, x, array, yflip = true, legend=false, 
            title="Nvx", titlefont=font(10), c=:bwr, clim=(-lim, lim))
end


#***** PHASE SPACE X DISTRIBUTION PROBE *****

struct PSxProbe <: AbstractProbe
    xrange::Tuple{Float64, Float64}
    nx::Int64
    dx::Float64
    vxrange::Tuple{Float64, Float64}
    nvx::Int64
    dvx::Float64
    ensemble_id::Int64
    ts::Vector{Float64}
    PSx::Vector{Matrix{Float64}}
end


PSxProbe(vxmax::Float64, nvx::Int64, geo::Cartesian1D, ensemble_id::Int64) = PSxProbe(
    (0, geo.xmax), geo.nx, geo.xmax/(geo.nx-1),
    (-vxmax, vxmax), nvx, (2*vxmax)/(nvx-1),
    ensemble_id, [], [])


function sample!(probe::PSxProbe, pic::PIC, t::Float64)
    push!(probe.ts, t)
    vx = [p.v[1] for p in pic.particles[probe.ensemble_id].coords]
    x = [p.r[1] for p in pic.particles[probe.ensemble_id].coords]
    #println("sampling VX: ", vx)
    push!(probe.PSx, sample_linear(x, vx, probe.xrange..., probe.nx, probe.dx, probe.vxrange..., probe.nvx, probe.dvx))
end

function plt(probe::PSxProbe)
    #array = stack(probe.PSx)
    #array = probe.PSx[end]
    array = sum(probe.PSx)
    x = LinRange(probe.xrange..., probe.nx)
    y = LinRange(probe.vxrange..., probe.nvx)
    lim = maximum(abs.(array))
    heatmap(y, x, array, yflip = true, legend=false, 
            title="Nvx", titlefont=font(10), c=:bwr, clim=(-lim, lim))
end
