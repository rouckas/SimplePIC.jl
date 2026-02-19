using Printf
using Profile
using Random
using SimplePIC
using StaticArrays

# Simulate the shielding of a charge fixed at the grid center

plotdir = "./plots-DebyeShielding/"
sampling_period = 5
mkpath(plotdir)

Random.seed!(1234)

ne_center = 10000
ne = 200000
ni = 200000
nx = 201
xmax = 0.01
nv = 151
vmax = 2000.
dt = 5e-10
dV = 1e-10
tmax = 5e-7
ntmax = Int64(ceil(tmax/dt))
exc = 0.01

electrons = ParticleEnsemble("electrons", m_e, -q_e, ne, Particle1d3vE)
protons = ParticleEnsemble("protons", 1*amu, q_e, ni, Particle1d3vE)
Hatoms = NeutralEnsemble("Hatoms", 10., 1*amu, 1e22);

# electron_interaction_list = load_interactions_lxcat("data/CS_e_Ar_Phelps.txt", electrons, argon);
# argon_interaction_list = load_interactions_lxcat("data/CS_Arp_Ar_Phelps.txt", electrons, argon);

e_H = Interaction("e_H", [5e-20, 5e-20], [0, 1], 0, electrons, Hatoms)
p_H = Interaction("p_H", [1e-17, 1e-17], [0, 1], 0, protons, Hatoms)

electron_interactions = make_interactions(electrons, [e_H], 1.0);
proton_interactions = make_interactions(protons, [p_H], 1.0);


#electrons.x = collect(0:(ne-1))*xmax/ne
xs = LinRange(xmax*0.0, xmax*1.0, ne)
#xs = LinRange(xmax*1e-9, xmax*(1-1e-9), ne)
init_thermal(electrons, 300.)
init_thermal(protons, 300.)

for i in 1:ne
    electrons.coords[i].r = SVector(xs[i])
    protons.coords[i].r = SVector(xs[i])
end

particles = [electrons, protons]
interactions = [electron_interactions, proton_interactions]

geometry = Cartesian1D(nx, xmax, xmax/(nx-1), dV)
# BC = BCDirichlet1D([-1e-9,1e-9])
BC = BCPeriodic1D()
pic = PIC(particles, interactions, geometry, BC, epsilon_0, 3)
pic.rhobg = 0 #ne*e/(nx-1)/dV

probes = Dict(
    :rho_probe => RhoProbe(pic.geo),
    :U_probe => UProbe(pic.geo),
    :nx_probe1 => NxProbe(pic.geo, 1),
    :nx_probe2 => NxProbe(pic.geo, 2),
    :nvx_probe => NvxProbe(2e5, 100, 1),
    :E_probe => EProbe(pic.geo),
    :PSx_probe => PSxProbe(2e5, 100, pic.geo, 1),
    )

lu = solve_init(pic.geo, pic.BC)

function pic_sim(pic, probes, dt, ntmax)
    particle_bc(pic)
    init_leapfrog(pic, dt)
    for nt in 1:ntmax
        sample(pic)
        center = nx÷2
        pic.rho[center] += ne_center*q_e/dV
        pic.rho .-= ne_center*q_e/(nx-1)/dV
        poisson_solve(pic, lu)
        interpolate(pic)
        advance(pic, dt)
        
        if nt*dt < 1e-7
            t_diag = 1
        else
            t_diag = 4
        end

        if nt % t_diag == 0
            println(nt, " ", length(electrons.coords))
            for (key, probe) in probes
                sample!(probe, pic, nt*dt)
            end
        end
        
    end
end

pic_sim(pic, probes, dt, ntmax)

# @profview pic_sim(pic, probes, dt, ntmax)

using Plots


for (key, probe) in probes
    probeplt = SimplePIC.plt(probe)
    savefig(probeplt, plotdir*"plot_"*string(key)*".png")
end

using Plots

probe = probes[:nx_probe1]
nsampl = length(probe.Nx) ÷ 10
nsampl = 1
x = LinRange(probe.xrange..., probe.nx)
n1 = sum(probe.Nx[end-nsampl:end])./nsampl

probe = probes[:nx_probe2]
n2 = sum(probe.Nx[end-nsampl:end])./nsampl
plt = plot(x, n1)
plt = plot!(x, n2)
plt = plot!(x, n2 .- n1)

savefig(plt, plotdir*"plot_nxcut.png")

probe = probes[:nx_probe1]
x = LinRange(probe.xrange..., probe.nx)
plt = plot(probe.ts, map(sum, probe.Nx))

probe = probes[:nx_probe2]
x = LinRange(probe.xrange..., probe.nx)
plt = plot!(probe.ts, map(sum, probe.Nx))

savefig(plt, plotdir*"plot_nsum.png")


probe1 = probes[:nx_probe1]
probe2 = probes[:nx_probe2]
Uprobe = probes[:U_probe]
x = LinRange(probe1.xrange..., probe1.nx)
println("animating nxcut")
anim = @animate for i in 1:(length(probe.Nx)-nsampl)
    n1 = sum(probe1.Nx[i:i+nsampl])./nsampl
    n2 = sum(probe2.Nx[i:i+nsampl])./nsampl
    U = sum(Uprobe.Us[i:i+nsampl])./nsampl
    Ulims = (minimum(minimum.(Uprobe.Us))/nsampl, maximum(maximum.(Uprobe.Us))/nsampl)
    Ulims = (-0.03, 0.18)
    plt1 = plot(x, n1,
        label=@sprintf("t = %.1f ns", probe1.ts[i]*1e9),
        legend=:topright,
        ylabel="rho (a.u.)"
        )
    plt2 = plot(x, U,
        label=@sprintf("t = %.1f ns", probe1.ts[i]*1e9),
        legend=:topright,
        ylabel="U (V)",
        xlabel="x (m)",
        ylims=Ulims,
        )
    plot!(plt1, x, n2, label="")
    plot!(plt1, x, n2 .- n1, label="")
    plot(plt1, plt2, layout = grid(2, 1, heights=[0.5 ,0.5]))
end
gif(anim, plotdir*"plot_nxcut_anim.gif", fps = 15)

# gifview --unoptimize plot_nxcut_anim.gif