using StaticArrays
using Random
using Printf
using FFTW

include("DirichletPIC.jl")
using .DirichletPIC


Random.seed!(1234)

#DOTAZ
#The fixed background ions are put in by default by ES1


#DOTAZ KONSTANTY
NSP = 2 #Beam + background = 1 + 2
N1 = 512 #Beam
N2 = 64 #Plasma
NG = 64
L = 2*pi
X1 = 0.01
NT = 600
DT = 0.2
V0 = 1.0
WP = 1.0
WPB = (1000)^(-1/2)
EPSI = 1.0
MODE = 1
QM = -1.0
V1 = 0.0

EPS = 1.0/EPSI

nx = NG+1
xmax = L
dt = DT
dV = 1.
dV = L/NG
tmax = DT*NT
#WP = 1.
#QM = 1.
Q = L*WP^2/((N1+N2)*QM*EPSI)
#Q = WP^2*xmax/ne*1/(QM)*epsilon_0*100
M = Q/QM

nv = 201
vmax = 5.

#DOTAZ: Interakce s pasivními ionty?

electrons1 = ParticleEnsemble("electrons1", M, -Q, N1, Particle1d1vE)
electrons2 = ParticleEnsemble("electrons2", M, -Q, N2, Particle1d1vE)

xs = LinRange(xmax*0.0, xmax*1.0, N2)

e1_interactions = make_interactions(electrons1, Interaction[]);
e2_interactions = make_interactions(electrons2, Interaction[]);

particles = [electrons1, electrons2]
interactions = [e1_interactions, e2_interactions]

#DOTAZ: V0 bude modulováno?

for i in 1:N1
    x = (i-1)*xmax/N1
    electrons1.coords[i].r = SVector(  x + X1*cos(2*pi*MODE*x/xmax))
    electrons1.coords[i].v = SVector(V0 + V1*sin(2*pi*MODE*x/xmax))
end

for i in 1:N2
    electrons2.coords[i].r = SVector(xs[i])
end

geometry = Cartesian1D(nx, xmax, xmax/(nx-1), dV)
BC = BCPeriodic1D()

pic = PIC(particles, interactions, geometry, BC, EPS, 1)
#pic.rhobg = 0 #ne*e/(nx-1)/dV

#DOTAZ: Proč ze 2*?
pic.rhobg = 2*(N1+N2)*Q/(nx-1)/dV

probes = Dict(
    :rho_probe  => RhoProbe(pic.geo),
    :U_probe    => UProbe(pic.geo),
    :energy_probe1  => EnergyProbe(dt, 1),
    :energy_probe2  => EnergyProbe(dt, 2),
    :nx_probe1  => NxProbe(pic.geo, 1),
    :nx_probe2  => NxProbe(pic.geo, 2),
    :nvx_probe1 => NvxProbe(2e0, 100, 1),
    :nvx_probe2 => NvxProbe(2e0, 100, 2),
    :E_probe    => EProbe(pic.geo),
    :PSx_probe1 => PSxProbe(2e0, 100, pic.geo, 1),
    :PSx_probe2 => PSxProbe(2e0, 100, pic.geo, 2),
    )

init = solve_init(pic.geo, pic.BC)

ntmax = Int64(ceil(tmax/dt))

function pic_sim(pic, probes, dt, ntmax)
    particle_bc(pic)
    init_leapfrog(pic, dt)

    for nt in 1:ntmax
        
        if true || nt % 5 == 1
            println(nt, " ", length(electrons1.coords))
            for (key, probe) in probes
                sample!(probe, pic, (nt-1)*dt)
            end
        end
        advance(pic, dt)
        sample(pic)
        poisson_solve(pic, init)
        interpolate(pic)
        
    end
end

pic_sim(pic, probes, dt, ntmax)

#@profview pic_sim(pic, probes, dt, ntmax)

using Plots

Folder = "Folder_BeamPlasma/"

for (key, probe) in probes
    probeplt = DirichletPIC.plt(probe)
    println(key)
    savefig(probeplt, joinpath(Folder, "plot_"*string(key)*".png"))
end

using Plots

probe = probes[:nx_probe1]
nsampl = length(probe.Nx) ÷ 10
x = LinRange(probe.xrange..., probe.nx)
n1 = sum(probe.Nx[end-nsampl:end])./nsampl

probe = probes[:nx_probe2]
n2 = sum(probe.Nx[end-nsampl:end])./nsampl
plt = plot(x, n1)
plt = plot!(x, n2)
plt = plot!(x, n2 .- n1)

savefig(plt, joinpath(Folder, "plot_nxcut.png"))

probe = probes[:nx_probe1]
x = LinRange(probe.xrange..., probe.nx)
plt = plot(probe.ts, map(sum, probe.Nx))

probe = probes[:nx_probe2]
x = LinRange(probe.xrange..., probe.nx)
plt = plot!(probe.ts, map(sum, probe.Nx))

savefig(plt, joinpath(Folder, "plot_nsum.png"))


probe = probes[:rho_probe]
freq = fftshift(fftfreq(NG, 1/geometry.dx))
specs = stack(
    [abs.(fftshift(fft(rho[1:end-1]))) for rho in probe.rhos[1:end]]
    )
plt = heatmap(probe.ts[1:end], freq, log10.(specs), clim=(-3, 2),
    ylim=(0, Inf), xlim=(0, Inf))
#plt = plot(freq, specs, yaxis=:log, ylims=(1e-4, Inf))

savefig(plt, joinpath(Folder, "plot_fft.png"))

probe1 = probes[:nx_probe1]
probe2 = probes[:nx_probe2]
Uprobe = probes[:U_probe]
x = LinRange(probe1.xrange..., probe1.nx)
anim = @animate for i in 1:(length(probe1.Nx)-nsampl)
    println("anim ", i)
    n1 = sum(probe1.Nx[i:i+nsampl])./nsampl
    n2 = sum(probe2.Nx[i:i+nsampl])./nsampl
    U = sum(Uprobe.Us[i:i+nsampl])./nsampl
    plt1 = plot(x, n1,
        label=@sprintf("t = %.1f ns", probe1.ts[i]*1e9),
        legend=:topright,
        ylabel="rho (a.u.)"
        )
    plt2 = plot(x, U,
        label=@sprintf("t = %.1f ns", probe1.ts[i]*1e9),
        legend=:topright,
        ylabel="U (V)",
        xlabel="x (m)"
        )
    plot!(plt1, x, n2, label="")
    plot!(plt1, x, n2 .- n1, label="")
    plot(plt1, plt2, layout = grid(2, 1, heights=[0.5 ,0.5]))
end
gif(anim, joinpath(Folder, "plot_nxcut_anim.gif"), fps = 15)

probe1 = probes[:PSx_probe1]
probe2 = probes[:PSx_probe2]
x = LinRange(probe1.xrange..., probe1.nx)
anim = @animate for i in 1:(length(probe1.ts)-nsampl)
    println("anim ", i)
    x = LinRange(probe1.xrange..., probe1.nx)
    y = LinRange(probe1.vxrange..., probe1.nvx)
    array = probe1.PSx[i] .- probe2.PSx[i]
    lim = maximum(abs.(array))
    heatmap(x, y, array', yflip = false, 
            titlefont=font(10), c=:bwr, clim=(-lim, lim),
            title=@sprintf("PSx t = %.1f s", probe1.ts[i]),
            legend=:topright,
            )
end
gif(anim, joinpath(Folder, "plot_PSx_anim.gif"), fps = 15)



