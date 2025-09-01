using StaticArrays
using Random
using Printf

if !isdefined(Main, :DirichletPIC)
    include("DirichletPIC.jl")
    using .DirichletPIC
end

Random.seed!(1234)

c_speed = 299792458.0
tmax = 5e-7
dt = 5e-10
ne = 200000
nx = 201
dx = c_speed*dt
xmax = (ne-1)*dx
nv = 151
vmax = 2000.
dV = 1e-10
ntmax = Int64(ceil(tmax/dt))
exc = 0.01

# EXTERNAL CIRCUIT CONDITIONS 2
C = 5e-08
L = 1e-07
R = 5
Q0 = 0
QL = 0
QR = 0
J12 = 0
J32 = 0
Js = []
deltaleft = 0
deltaright = 0

function Voltage(t::Float64)
    return 1e-05*sin(t/tmax)
end

electrons = ParticleEnsemble("electrons", m_e, -q_e, ne, Particle1d3vE)
argonplus = ParticleEnsemble("argonplus", 40*amu*0.1, q_e, ne, Particle1d3vE)
argon = NeutralEnsemble("argon", 10., 40*amu*0.1, 1e19);

T_inject = 300.0
A = dV/dx
vth_e = sqrt(8*k_B*T_inject/(pi*electrons.m))
vth_p = sqrt(8*k_B*T_inject/(pi*argonplus.m))
#inject_density = ne/(A*xmax)
inject_density = 5e10
ne_inject = 1/4*A*inject_density*vth_e*dt
np_inject = 1/4*A*inject_density*vth_p*dt


electron_interaction_list = load_interactions_lxcat("data/CS_e_Ar_Phelps.txt", electrons, argon);
argon_interaction_list = load_interactions_lxcat("data/CS_Arp_Ar_Phelps.txt", electrons, argon);
electron_interactions = make_interactions(electrons, electron_interaction_list);
argonplus_interactions = make_interactions(argonplus, argon_interaction_list, 1.0);


#electrons.x = collect(0:(ne-1))*xmax/ne
xs = LinRange(xmax*0.0, xmax*1.0, ne)
#xs = LinRange(xmax*1e-9, xmax*(1-1e-9), ne)
init_thermal(electrons, 300.)
init_thermal(argonplus, 300.)

for i in 1:ne
    electrons.coords[i].r = SVector(xs[i])
    argonplus.coords[i].r = SVector(xs[i])
end

particles = [electrons, argonplus]
interactions = [electron_interactions, argonplus_interactions]

geometry = Cartesian1D(nx, xmax, xmax/(nx-1), dV)
BC = BCDirichlet1D([-1e-09, 1e-09])
pic = PIC(particles, interactions, geometry, BC, epsilon_0, 3) #particles, interactions, geometry, BC = Boundary condition, epsilon_0, vdim
obvod = Circuit(R, L, C, Q0, Js, Voltage)
pic.rhobg = 0 #ne*e/(nx-1)/dV

probes = Dict(
    :rho_probe => RhoProbe(pic.geo),
    :U_probe => UProbe(pic.geo),
    :nx_probe1 => NxProbe(pic.geo, 1),
    :nx_probe2 => NxProbe(pic.geo, 2),
    :nvx_probe => NvxProbe(2e5, 100, 1),
    :E_probe => EProbe(pic.geo),
    :PSx_probe => PSxProbe(2e5, 100, pic.geo, 1),
    :B_probe => BProbe(pic.geo),
    :T_probe => TProbe(dt, pic.geo, 1),
    :QL_probe => QLProbe(),
    :QR_probe => QRProbe(),
    :J_probe => JProbe(),
    :Q_probe => QProbe()
    )

lu = solve_init(pic.geo, pic.BC)

function pic_sim(pic, probes, dt, ntmax)
    particle_bc(pic)
    init_leapfrog(pic, dt)
    for nt in 1:ntmax
        sample(pic)
        advance_current(pic, dt)
        advance_external(pic, obvod, nt * dt, dt)
        poisson_solve(pic)
        maxwell_solve(pic, dt)
        interpolate(pic)
        advance_v_all(pic, dt) #Boris
        if nt % 5 == 1
            println(nt, " ", length(electrons.coords), " ", rand(Poisson(ne_inject)))
            for (key, probe) in probes
                if probe isa QProbe || probe isa JProbe
                    sample!(probe, obvod, nt*dt)
                else
                    sample!(probe, pic, nt*dt)
                end
            end
        end
    end
end

#=
function pic_sim(pic, probes, dt, ntmax)
    particle_bc(pic)
    init_leapfrog(pic, dt, 0.0)
    for nt in 1:ntmax
        sample(pic)

        poisson_solve(pic, lu)
        interpolate(pic)
        advance(pic, dt)

        inject(electrons, rand(Poisson(ne_inject)), T_inject, xmax, nt*dt, dt)
        inject(argonplus, rand(Poisson(np_inject)), T_inject, xmax, nt*dt, dt)

        if nt % 5 == 1
            println(nt, " ", length(electrons.coords), " ", rand(Poisson(ne_inject)))
            for (key, probe) in probes
                sample!(probe, pic, nt*dt)
            end
        end
    end
end
=#

# pic_sim(pic, probes, dt, ntmax)

pic_sim(pic, probes, dt, ntmax)

using Plots

Folder = "Figures/"

for (key, probe) in probes
    probeplt = DirichletPIC.plt(probe)
    savefig(probeplt, joinpath(Folder, "plot_"*string(key)*".png"))
end

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


probe1 = probes[:nx_probe1]
probe2 = probes[:nx_probe2]
Uprobe = probes[:U_probe]
x = LinRange(probe1.xrange..., probe1.nx)
anim = @animate for i in 1:(length(probe.Nx)-nsampl)
    println("anim ", i)
    n1 = sum(probe1.Nx[i:i+nsampl])./nsampl
    n2 = sum(probe2.Nx[i:i+nsampl])./nsampl
    U = sum(Uprobe.Us[i:i+nsampl])./nsampl
    plt1 = plot(x, n1,
        label=@sprintf("t = %.1f ns", probe1.ts[i]*1e9),
        legend=:topleft,
        ylabel="rho (a.u.)"
        )
    plt2 = plot(x, U,
        label=@sprintf("t = %.1f ns", probe1.ts[i]*1e9),
        legend=:topleft,
        ylabel="U (V)",
        xlabel="x (m)"
        )
    plot!(plt1, x, n2, label="")
    plot!(plt1, x, n2 .- n1, label="")
    plot(plt1, plt2, layout = grid(2, 1, heights=[0.5 ,0.5]))
end
gif(anim, joinpath(Folder, "plot_nxcut_anim.gif"), fps = 15)