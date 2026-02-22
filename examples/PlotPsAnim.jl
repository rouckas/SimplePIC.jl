function plot_PS_marginal(PS_probe, i_time, probeplt, ax_n, ax_PS, ax_v)
    x = LinRange(probe.xrange..., probe.nx)
    y = LinRange(probe.vxrange..., probe.nvx)
    array = PS_probe.PSx[i_time] #.- probe2.PSx[i]
    lim = maximum(abs.(array))

    heatmap!(probeplt, x, y, array', yflip = false, 
        titlefont=font(10), 
        # c=:heat,
        c = :dense,
        clim=(0, lim),
        cbar=false,
        # title=@sprintf("PSx t = %.1f s", probe.ts[i]),
        legend=:topright,
        subplot=ax_PS,
        widen=false,
        annotations = (:bottomleft, (@sprintf("f(vx, x), t = %.2f s", probe.ts[i_time]), 8, :left)),
        xlabel = "x (m)", ylabel = "v (m/s)",
        guidefontsize = 8,
        )

    plot!(probeplt, x, sum(array, dims=2)./dV, fill=true,
        # ylims = (0, nx_ylim),
        xformatter=_->"",
        # yformatter=_->"",
        ylabel="n(x) (m-3)",
        subplot=ax_n,
        widen=false,
        annotations = (:bottomleft, ("n(x)", 8, :left)),
        legend = false,
        yguidefontsize = 8,
        )
    
    plot!(probeplt, sum(array, dims=1)', y, fill=true,
        # xlims = (0, fv_ylim),
        xformatter=_->"",
        yformatter=_->"",
        # label="rho(v)",
        subplot=3,
        widen=false,
        annotations = (:bottomleft, ("f(v)", 8, :left)),
        legend = false,
        guidefontsize = 8,
        )

end

function plot_energy(energy_probe, i_time, probeplt, ax_energy)
    E_kin = energy_probe.Ekin
    time = energy_probe.ts
    E_pot = energy_probe.Epot
    E_tot = E_kin + E_pot
    Emax = maximum(E_tot)
    Escale = 1e25
    t = i_time
    p3 = plot!(probeplt, time[1:t], E_kin[1:t]*Escale, ylims=(0, Emax*Escale), label="kinetic",
        subplot=4)
    p3 = plot!(probeplt, time[1:t], E_pot[1:t]*Escale, label="potential",
        subplot=4)
    p3 = plot!(probeplt, time[1:t], E_tot[1:t]*Escale, label="total",
        xlabel = "t (s)",
        ylabel = "E (1e-25 J)",
        guidefontsize = 8,
        legend=:bottomright,
    subplot=4)
end

function animate_PS_energy(PS_probe, energy_probe, omega_ref, omega_label)
    layout = @layout [[a            _
                  b{0.77w,0.75h} c]
                  d{0.28h}]

    println("animating PSx")
    nx_lim = maximum(maximum.(sum.(PS_probe.PSx, dims=2)))/dV
    fv_lim = maximum(maximum.(sum.(PS_probe.PSx, dims=1)))

    anim = @animate for i in 1:(length(probe.ts))

        probeplt = plot(layout = layout, size = (550, 550))
        plot_PS_marginal(PS_probe, i, probeplt, 1, 2, 3)
        plot!(probeplt, subplot=1, ylim=(0, nx_lim))
        plot!(probeplt, subplot=3, xlim=(0, fv_lim))


        plot_energy(energy_probe, i, probeplt, 4)
        p3 = plot!(time[1:nsampl], E_kin[1]*Escale*cos.(omega_ref*time[1:nsampl]).^2,
            label=omega_label, ls=:dot,
            xlims=(0, time[nsampl]),
            subplot=4,
            xlabel = "t (s)",
            ylabel = "E (1e-25 J)",
            legend=:bottomright,
            guidefontsize = 8,
            )

    end

    return anim
end