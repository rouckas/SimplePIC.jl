using SimplePIC
using StaticArrays
using Printf
using Plots
using LaTeXStrings

plotdir = "./plots-Oscillations/"
sampling_period = 5
mkpath(plotdir)

vdim = 1

ne = 100000
nx = 401
xmax = 1.
nv = 151
vmax = 20.
dt = 0.0001
dV = 100.
tmax = 0.07
ntmax = Int64(ceil(tmax/dt))
exc = 10

numden = ne/(nx-1)/dV
omega_p = sqrt(q_e^2 * numden /(epsilon_0*m_e))

# electrons = Particles(ne, -e, m_e)
electrons = ParticleEnsemble("electrons", m_e, q_e, ne, Particle1d1vE)

for i in 1:ne
    c = electrons.coords[i]
    c.r = SVector((i-1)*xmax/ne)
    c.v = SVector(exc*sin(c.r[1]/xmax*2*pi)) #+ 0.0000001*rand(Normal(0, 1e-3), ne)
end
#electrons.x += exc*map(sin, electrons.x/xmax*2*pi) #+ 0.0000001*rand(Normal(0, 1e-3), ne)
particles = [electrons]
interactions = [make_interactions(electrons, Interaction[], 1.0)]

geometry = Cartesian1D(nx, xmax, xmax/(nx-1), dV)
BC = BCPeriodic1D()

pic = PIC(particles, interactions, geometry, BC, epsilon_0, vdim)
# pic = PIC(particles, nx, xmax, dV, epsilon_0)
pic.rhobg = ne*q_0/(nx-1)/dV

probes = Dict(
    :energy_probe => EnergyProbe(dt, 1),
    :rho_probe => RhoProbe(pic.geo),
    :U_probe => UProbe(pic.geo),
    :nx_probe1 => NxProbe(pic.geo, 1),
    :nvx_probe => NvxProbe(2e1, 100, 1),
    :E_probe => EProbe(pic.geo),
    :PSx_probe => PSxProbe(2e1, 100, pic.geo, 1),
    )

function pic_sim(pic, probes, dt, ntmax)
    init_leapfrog(pic, dt)

    # diag = Diagnostic(nx, nv, ntmax, xmax, vmax, dt*ntmax)

    for nt in 1:ntmax
        sample(pic)
        poisson_solve(pic)
        interpolate(pic)
        advance(pic, dt)
        

        if nt % sampling_period == 0
            println(nt, " ", length(electrons.coords))
            for (key, probe) in probes
                sample!(probe, pic, nt*dt)
            end
        end
        
    end
end


pic_sim(pic, probes, dt, ntmax)

for (key, probe) in probes
    probeplt = SimplePIC.plt(probe)
    savefig(probeplt, plotdir*"plot_"*string(key)*".png")
end

E_kin = probes[:energy_probe].Ekin
time = probes[:energy_probe].ts
E_pot = probes[:energy_probe].Epot
E_tot = E_kin + E_pot
Emax = maximum(E_tot)
Escale = 1e25

vx = LinRange(probes[:nvx_probe].vxrange..., probes[:nvx_probe].nvx)

nsampl = length(probes[:rho_probe].rhos)
anim = @animate for t in 1:nsampl
    p1 = plot((probes[:rho_probe].rhos[t] .- pic.rhobg).*-1e18,
     fill=true, ylims = (0, 1.5), label="rho(x)")
    p2 = plot(vx, probes[:nvx_probe].Nvx[t]./q_0, fill=true,
      ylims = (0, maximum(probes[:nvx_probe].Nvx[1])*2/q_0), 
      label="rho(v)")
    p3 = plot(E_kin[1:t]*Escale, xlims=(0, nsampl), ylims=(0, Emax*Escale), label="kinetic")
    p3 = plot!(E_pot[1:t]*Escale, xlims=(0, nsampl), label="potential")
    p3 = plot!(E_tot[1:t]*Escale, xlims=(0, nsampl), label="total")
    p3 = plot!(E_kin[1]*Escale*cos.(omega_p*time[1:nsampl]).^2,
     label=L"\cos^2(\omega_p t)", ls=:dot)
    # the energy oscillates at double plasma frequency,
    # as it is a square of velocity / displacement
    plot(p1, p2, p3, layout=(3,1))
end

gif(anim, plotdir*"plot_nxcut_anim.gif", fps = 20)


probe = probes[:PSx_probe]
x = LinRange(probe.xrange..., probe.nx)
println("animating PSx")
anim = @animate for i in 1:(length(probe.ts))
    x = LinRange(probe.xrange..., probe.nx)
    y = LinRange(probe.vxrange..., probe.nvx)
    array = probe.PSx[i] #.- probe2.PSx[i]
    lim = maximum(abs.(array))
    heatmap(x, y, array', yflip = false,
            titlefont=font(10), c=:bwr, clim=(-lim, lim),
            title=@sprintf("PSx t = %.1f s", probe.ts[i]),
            legend=:topright,
            )
end
gif(anim, plotdir*"plot_PSx_anim.gif", fps = 15)


include("PlotPsAnim.jl")
function animate_PS_energy(PS_probe, energy_probe)
    layout = @layout [[a            _
                  b{0.77w,0.75h} c]
                  d{0.28h}]

    println("animating PSx")
    nx_lim = maximum(maximum.(sum.(PS_probe.PSx, dims=2)))/dV
    fv_lim = maximum(maximum.(sum.(PS_probe.PSx, dims=1)))

    @animate for i in 1:(length(probe.ts))
        probeplt = plot(layout = layout, size = (550, 550))
        plot_PS_marginal(PS_probe, i, probeplt, 1, 2, 3)
        plot!(probeplt, subplot=1, ylim=(0, nx_lim))
        plot!(probeplt, subplot=3, xlim=(0, fv_lim))

        plot_energy(energy_probe, i, probeplt, 4)
        p3 = plot!(energy_probe.ts, E_kin[1]*Escale*cos.(omega_p*energy_probe.ts).^2,
            label=L"\cos^2(\omega_p t)", ls=:dot,
            xlims=(0, energy_probe.ts[end]),
            subplot=4,
            )

    end
end
anim = animate_PS_energy(probes[:PSx_probe], probes[:energy_probe])
gif(anim, plotdir*"plot_PS_marginal_anim.gif", fps = 15)