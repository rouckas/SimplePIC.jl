
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



mutable struct Diagnostic
    nt::Int64
    nv::Int64
    nx::Int64
    ts::Vector{Float64}
    rhos::Array{Float64, 2}
    particle_rhos::Array{Float64, 3}
    Us::Array{Float64, 2}
    Es::Array{Float64, 2}
    vs::Array{Float64, 2}
    PSs::Array{Float64, 3}
    xmax::Float64
    vmax::Float64
    tmax::Float64
    vbins::Vector{Float64}
    N::Int64
end

function Diagnostic(nx::Int64, nv::Int64, nt::Int64, nspecies::Int64, xmax::Float64, vmax::Float64, tmax::Float64)
    Diagnostic(nt,
        nv,
        nx,
        zeros(nt),
        zeros(nt, nx),
        zeros(nt, nspecies, nx),
        zeros(nt, nx),
        zeros(nt, nx),
        zeros(nt, nv),
        zeros(nt, nx, nv),
        xmax,
        vmax,
        tmax,
        LinRange(-vmax-xmax/(nv-1),vmax+xmax/(nv-1),nv+1),
        0
        )
end

function histogram_v(particles::ParticleEnsemble, diag::Diagnostic, dir::Int64=1)
    vs = @view diag.vs[diag.N,:]
    for p in particles.coords
        v = p.v[dir]
        if abs(v) >= diag.vmax
            continue
        end
        vs[Int64(floor(diag.nv * (v / (2 * diag.vmax) + 0.5))) + 1] += particles.q
    end
end

function histogram_xv(particles::ParticleEnsemble, pic::PIC, diag::Diagnostic)
    PSs = @view diag.PSs[diag.N,:,:]
    for p in particles.coords
        v = p.v[1]
        x = p.r[1]
        if abs(v) >= diag.vmax
            continue
        end
        PSs[Int64(floor(x*(pic.nx-1)/pic.xmax+0.5)+1),
          Int64(floor(diag.nv * (v / (2 * diag.vmax) + 0.5))) + 1] += particles.q
    end
end

function sample(diag::Diagnostic, t::Float64, pic::PIC)
    diag.N += 1
    
    for p in pic.particles
#         diag.vs[diag.N] += fit(Histogram, p.vx, diag.vbins).weights*p.q  
        histogram_v(p, diag)
#         diag.PSs[diag.N] += fit(Histogram, (p.x, p.vx), (pic.xbins, diag.vbins)).weights*p.q
        histogram_xv(p, pic, diag)

    end
    diag.Us[diag.N,:] .= pic.U
    diag.Es[diag.N,:] .= [e[1] for e in pic.E]
    diag.rhos[diag.N,:] .= pic.rho
    diag.ts[diag.N] = t
    
end

function plt(diag::Diagnostic, pic::PIC)
    
    function plt_record(x, y, array, title)
        lim = maximum(abs.(array))
        return heatmap(x, y, array, yflip = true, legend=false, 
                 title=title, titlefont=font(10), c=:bwr, clim=(-lim, lim))
    end
    
    p1 = plt_record(pic.x, diag.ts[1:diag.N], diag.rhos[1:diag.N,:], "Charge density")
    p2 = plt_record(pic.x, diag.ts[1:diag.N], diag.Us[1:diag.N,:], "Electrical potential")
    p3 = plt_record(pic.x, diag.ts[1:diag.N], diag.Es[1:diag.N,:], "Elextrical field")
    vx = LinRange(-diag.vmax,diag.vmax,diag.nv)
    p4 = plt_record(vx, diag.ts[1:diag.N], diag.vs[1:diag.N,:], "Speed")
    
    return plot(p1, p2, p3, p4, size = (900, 250), layout=(1,4))
end

function plt_PS(diag::Diagnostic, lims=1e-18)
    x = dropdims(sum(diag.PSs, dims=1), dims=1)'
    return heatmap(x, size=(500, 500), legend=false, c=:bwr,  clim=(-lims, lims), title="Phase space")
end