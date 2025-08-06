using StaticArrays
using Random
using Printf
using FFTW

# include("DirichletPIC.jl")
using SimplePIC

plotdir = "./plots-two_stream/"
sampling_period = 5
mkpath(plotdir)

Random.seed!(1234)

NSP = 2 # number of species
L = 2*pi*2
DT = 0.2
NT = 300.
EPSI = 1.0
NG = 32*4*2
IW = 2
A1 = 0
A2 = 0

# species 1
N = 128*16*2
WP = 1.0
WC = 0.0
QM = -1.0
V0 = 1.0
MODE = 2
X1 = 0.01
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
Q = L*WP^2/(N*QM*EPSI)
#Q = WP^2*xmax/ne*1/(QM)*epsilon_0*100
M = Q/QM

nv = 201
vmax = 5.


electrons1 = ParticleEnsemble("electrons1", M, -Q, N, Particle1d1vE)
electrons2 = ParticleEnsemble("electrons2", M, -Q, N, Particle1d1vE)


for i in 1:N
    local x = (i-1)*xmax/N
    electrons1.coords[i].r = SVector(  x + X1*cos(2*pi*MODE*x/xmax))
    electrons1.coords[i].v = SVector( V0 + V1*sin(2*pi*MODE*x/xmax))
    electrons2.coords[i].r = SVector(  x - X1*cos(2*pi*MODE*x/xmax))
    electrons2.coords[i].v = SVector(-V0 + V1*sin(2*pi*MODE*x/xmax))
end


e1_interactions = make_interactions(electrons1, Interaction[]);
e2_interactions = make_interactions(electrons2, Interaction[]);

particles = [electrons1, electrons2]
interactions = [e1_interactions, e2_interactions]

geometry = Cartesian1D(nx, xmax, xmax/(nx-1), dV)
BC = BCPeriodic1D()

pic = PIC(particles, interactions, geometry, BC, EPS, 1)
#pic.rhobg = 0 #ne*e/(nx-1)/dV
pic.rhobg = 2*N*Q/(nx-1)/dV

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
        
        if (nt-1) % sampling_period == 0
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


using Plots

for (key, probe) in probes
    probeplt = SimplePIC.plt(probe)
    println(key)
    savefig(probeplt, plotdir*"plot_"*string(key)*".png")
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

savefig(plt, plotdir*"plot_nxcut.png")

probe = probes[:nx_probe1]
x = LinRange(probe.xrange..., probe.nx)
plt = plot(probe.ts, map(sum, probe.Nx))

probe = probes[:nx_probe2]
x = LinRange(probe.xrange..., probe.nx)
plt = plot!(probe.ts, map(sum, probe.Nx))

savefig(plt, plotdir*"plot_nsum.png")


probe = probes[:rho_probe]
freq = fftshift(fftfreq(NG, 1/geometry.dx))
specs = stack(
    [abs.(fftshift(fft(rho[1:end-1]))) for rho in probe.rhos[1:end]]
    )
plt = heatmap(probe.ts[1:end], freq, log10.(specs), clim=(-3, 2),
    ylim=(0, Inf), xlim=(0, Inf))
#plt = plot(freq, specs, yaxis=:log, ylims=(1e-4, Inf))

savefig(plt, plotdir*"plot_fft.png")


probe1 = probes[:nx_probe1]
probe2 = probes[:nx_probe2]
Uprobe = probes[:U_probe]
x = LinRange(probe1.xrange..., probe1.nx)
println("animating nxcut")
anim = @animate for i in 1:(length(probe1.Nx)-nsampl)
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
gif(anim, plotdir*"plot_nxcut_anim.gif", fps = 15)

probe1 = probes[:PSx_probe1]
probe2 = probes[:PSx_probe2]
x = LinRange(probe1.xrange..., probe1.nx)
println("animating PSx")
anim = @animate for i in 1:(length(probe1.ts)-nsampl)
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
gif(anim, plotdir*"plot_PSx_anim.gif", fps = 15)