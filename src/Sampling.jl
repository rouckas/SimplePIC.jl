@inline function sample_linear!(rho::Vector{Float64}, x::Float64, nx::Int64, dx::Float64)
    xi = x/dx
    xl = floor(Int64, xi)
    w = xi-xl
    #=
    if xl <= -1
        println("sample err ", x, " ", xi, " ", xl)
    end
    if xl >= nx-1
        println("sample max err ", x, " ", xi, " ", xl)
    end
    =#
    rho[xl+1] += 1-w
    rho[xl+2] += w
end

#= @inline function sample_linear!(rho::Vector{Float64}, vsum::Vector{SVector{3, Float64}}, weight::SVector{3, Float64}, x::Float64, nx::Int64, dx::Float64)
    xi = x/dx
    xl = floor(Int64, xi)
    w = xi-xl
    #=
    if xl <= -1
        println("sample err ", x, " ", xi, " ", xl)
    end
    if xl >= nx-1
        println("sample max err ", x, " ", xi, " ", xl)
    end
    =#
    rho[xl+1] += 1-w
    rho[xl+2] += w
    vsum[xl+1] += (1-w)*weight
    vsum[xl+2] += w*weight
end =#


@inline function sample_linear!(rho::Vector{Float64}, x::SVector{1, Float64}, nx::Int64, dx::Float64)
    sample_linear!(rho, x[1], nx, dx)
end

function sample_linear(x::Vector{Float64}, nx::Int64, dx::Float64)
    rho = zeros(nx)
    for xx in x
        sample_linear!(rho, xx, nx, dx)
    end
    return rho
end

#=function sample_linear(p::ParticleEnsemble{P}, nx::Int64, dx::Float64) where P <: Union{Particle1d3v, Particle1d3vE, Particle1d1vE}
    rho = zeros(nx)
    vsum = [zeros(SVector{3, Float64}) for i in 1:nx]
    for x in p.coords
        sample_linear!(rho, vsum, x.r[1], x.v, nx, dx)
    end
    return rho, vsum
end=#

function sample_linear(p::ParticleEnsemble{P}, nx::Int64, dx::Float64) where P <: Union{Particle1d3v, Particle1d3vE, Particle1d1vE}
    rho = zeros(nx)
    for x in p.coords
        sample_linear!(rho, x.r, nx, dx)
    end
    return rho
end

function sample_linear(x::Vector{Float64}, xmin::Float64, xmax::Float64, nx::Int64, dx::Float64)
    rho = zeros(nx)
    #println("sampling ", xmin, " ", xmax)
    for _x in x
        if xmax > _x > xmin
            sample_linear!(rho, _x - xmin, nx, dx)
        end
    end
    return rho
end


@inline function sample_linear!(rho::Matrix{Float64}, x::Float64, y::Float64, nx::Int64, dx::Float64, ny::Int64, dy::Float64)
    xi = x/dx
    xl = floor(Int64, xi)
    xw = xi-xl
    yi = y/dy
    yl = floor(Int64, yi)
    yw = yi-yl
    #=
    if xl <= -1
        println("sample err ", x, " ", xi, " ", xl)
    end
    if xl >= nx-1
        println("sample max err ", x, " ", xi, " ", xl)
    end
    if yl <= -1
        println("sample err ", y, " ", yi, " ", yl)
    end
    if yl >= ny-1
        println("sample max err ", y, " ", yi, " ", yl, " ", dy)
    end
    =#
    rho[xl+1, yl+1] += (1-xw)*(1-yw)
    rho[xl+2, yl+1] += xw*(1-yw)
    rho[xl+1, yl+2] += (1-xw)*yw
    rho[xl+2, yl+2] += xw*yw
end

function sample_linear(x::Vector{Float64}, y::Vector{Float64},
         xmin::Float64, xmax::Float64, nx::Int64, dx::Float64,
         ymin::Float64, ymax::Float64, ny::Int64, dy::Float64)
    rho = zeros(nx, ny)
    for (xx, yy) in zip(x, y)
        if (xmax > xx > xmin) && (ymax > yy > ymin)
            sample_linear!(rho, xx-xmin, yy-ymin, nx, dx, ny, dy)
        end
    end
    return rho
end

