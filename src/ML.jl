

__precompile__()



module ML



export Kernel, PolynomialKernel, GaussianKernel, ExponentialKernel,
    fit!, grad, ∇, center, KRR, KernelRidgeRegressor, RFKM, RandomFourierKernelMachine,
    KPCA, KernelPCA, negative_log_likelihood, rde, mae, rmse




using Distributions: Normal
using Distances: pairwise, colwise, Euclidean, SqEuclidean, euclidean, sqeuclidean





mae(x,y) = mean(abs(x-y))

rmse(x,y) = sqrt(sumabs2(x-y)/length(x))


abstract Kernel

type PolynomialKernel <: Kernel
    α::Float64
    c::Float64
    ξ::Float64
end

PolynomialKernel(;α=1.,c=0.,ξ=2.) = PolynomialKernel(α,c,ξ)

Base.call(k::PolynomialKernel, x::AbstractVector, y::AbstractVector) = (k.α*vecdot(x,y) + k.c).^k.ξ
Base.call(k::PolynomialKernel, x::AbstractArray, y::AbstractArray) = (k.α*(x'*y) + k.c).^k.ξ
Base.call(k::PolynomialKernel, x::AbstractArray) = k(x,x)


type GaussianKernel <: Kernel
    σ::Float64
end

GaussianKernel(;σ=1.) = GaussianKernel(σ)

Base.call(k::GaussianKernel, x::AbstractVector, y::AbstractVector) = exp(-sqeuclidean(x,y)/(2k.σ^2))
Base.call(k::GaussianKernel, x::AbstractArray, y::AbstractVector) = exp(-colwise(SqEuclidean(),x,y)/(2k.σ^2))
Base.call(k::GaussianKernel, x::AbstractVector, y::AbstractArray) = k(y,x)
Base.call(k::GaussianKernel, x::AbstractArray, y::AbstractArray) = exp(-pairwise(SqEuclidean(),x,y)/(2k.σ^2))
Base.call(k::GaussianKernel, x::AbstractArray) = exp(-pairwise(SqEuclidean(),x)/(2k.σ^2))



function grad(k::GaussianKernel, x::AbstractVector, y::AbstractVector)
    (y-x)*(k(x,y)/k.σ^2)
end

function grad(k::GaussianKernel, x::AbstractArray, y::AbstractVector)
    K = k(x,y)
    n,m = size(x,1),length(K)
    dst = Array{eltype(x)}((n,m))
    for i in 1:m
        dst[:,i] = grad(k,x[:,i],y)
    end
    reshape(dst,(n,size(K)...))
end

function grad(k::GaussianKernel, x::AbstractVector, y::AbstractArray)
    K = k(x,y)
    n,m = size(x,1),length(K)
    dst = Array{eltype(x)}((n,m))
    for i in 1:m
        dst[:,i] = grad(k,x,y[:,i])
    end
    reshape(dst,(n,size(K)...))
end

function grad(k::GaussianKernel, x::AbstractMatrix, y::AbstractMatrix)
    n,m,p = size(x,1),size(x,2),size(y,2)
    K = k(x,y)
    dst = Array{eltype(x)}((n,m,p))
    for j in 1:p
        for i in 1:m
            dst[:,i,j] = grad(k,x[:,i],y[:,j])
        end
    end
    dst
end



# symmetric
# function grad(k::GaussianKernel, x::AbstractMatrix)
#   n,m = size(x,1),size(x,2)
#   K = k(x,x)
#   dst = Array{eltype(x)}((n,m,m))
#   for j in 1:m
#       for i in j:m
#           dst[:,i,j] = (x[:,j]-x[:,i])*(K[i,j]/k.σ^2)
#           dst[:,j,i] = -dst[:,i,j]
#       end
#   end
#   dst
# end

# function grad(k::GaussianKernel, x::AbstractArray, y::AbstractArray)
#   n = size(x,1)
#   m,p = prod(size(x)[2:end]),prod(size(y)[2:end])
#   K = k(x,y)
#   shp = size(K)
#   K = reshape(K,(m,p))
#   dst = Array{eltype(x)}((n,m,p))
#   for j in 1:p
#       for i in 1:m
#           dst[:,i,j] = (y[:,j]-x[:,i])*(K[i,j]/k.σ^2)
#       end
#   end
#   reshape(dst,(n,shp...))
# end





function rand_train_test_ind(n,k)
    ind = randperm(n)
    ind[1:k], ind[k+1:end]
end

function uniform_train_test_ind(n,d)
    train = round(Int,1:n/d:n)
    test = setdiff(1:n,train)
    train,test
end


function perturb(x,i,h)
    y = copy(x)
    y[i] += h
    y
end

function fd_grad(f,x;h=1e-4)
    f0 = f(x)
    eltype(x)[(f(perturb(x,i,h))-f0)/h for i in 1:length(x)]
end


function fd_jacobian(f,x;h=1e-4)
    f0 = f(x)
    hcat([(f(perturb(x,i,h))-f0)/h for i in 1:length(x)]...)'
end





type ExponentialKernel <: Kernel
    σ::Float64
end

ExponentialKernel(;σ=1.) = ExponentialKernel(σ)

Base.call(k::ExponentialKernel, x::AbstractVector, y::AbstractVector) = exp(-euclidean(x,y)/(2k.σ^2))
Base.call(k::ExponentialKernel, x::AbstractArray, y::AbstractVector) = exp(-colwise(Euclidean(),x,y)/(2k.σ^2))
Base.call(k::ExponentialKernel, x::AbstractVector, y::AbstractArray) = k(y,x)
Base.call(k::ExponentialKernel, x::AbstractArray, y::AbstractArray) = exp(-pairwise(Euclidean(),x,y)/(2k.σ^2))
Base.call(k::ExponentialKernel, x::AbstractArray) = exp(-pairwise(Euclidean(),x)/(2k.σ^2))



abstract Predictor



# type KernelRidgeRegressor{T,Tk<:Kernel,Ni,No} <: Predictor
#     x::Array{T,Ni}          # features
#     y::Array{T,No}          # labels
#     k::Tk                   # kernel
#     λ::Float64              # regularization strength
#     w::Array{T,No}          # weights
# end


# typealias KRR KernelRidgeRegressor

# function KernelRidgeRegressor(x  ::AbstractArray,                           # features
#                               y  ::AbstractArray;                           # labels
#                               k  ::Kernel  = GaussianKernel(),              # kernel
#                               λ  ::Float64 = 1e-4)                          # regularization strength
#     KernelRidgeRegressor{eltype(x),typeof(k),ndims(x),ndims(y)}(x,y,k,λ,zeros(y))
# end


# typealias sKRR{T,Tk<:Kernel,Ni} KRR{T,Tk,Ni,1} # single valued


# function fit!(m::sKRR)
#     K = m.k(m.x)                                        # kernel matrix
#     m.w = (K + m.λ*I)\m.y                               # solve for weights
# end

# function fit!(m::KRR)
#     K = m.k(m.x)                                        # kernel matrix
#     m.w = (K + m.λ*I)\m.y'                              # solve for weights
# end


# # fit with new features, labels
# function fit!(m::KRR,
#               x::AbstractArray,                 # features
#               y::AbstractArray)                 # labels
#     m.x,m.y = x,y
#     fit!(m)
# end


# Base.call(m::sKRR, x::AbstractVector) = dot(m.w,m.k(x,m.x))
# Base.call(m::sKRR, x::AbstractArray) = m.k(x,m.x)*m.w

# Base.call(m::KRR, x::AbstractVector) = m.w' * m.k(x,m.x)
# Base.call(m::KRR, x::AbstractArray) = m.w' * m.k(x,m.x)'


# grad(m::KRR, x::AbstractVector) = grad(m.k,x,m.x)*m.w

# function grad(m::KRR, x::AbstractMatrix)
#     n,d = size(m.x)
#     k = size(x,2)
#     reshape(reshape(grad(m.k,x,m.x),(n*k,d)) * m.w, (n,k))
# end


# ∇ = grad



mae(x,y) = mean(abs(x-y))
rmse(x,y) = sqrt(sumabs2(x-y)/length(x))




function L_matrix(n::Int, d::Int)
    d == 1 && return I
    @assert n%d == 0 "n=$n must be divisble by d=$d"
    m = div(n,d)
    L = zeros(n,m)
    for i in 1:m
        L[d*(i-1)+1:d*i,i] = 1
    end
    L
end

L_matrix(x::AbstractMatrix, y::AbstractVector) = L_matrix(size(x,2),div(size(x,2),length(y)))




type KernelRidgeRegressor{T,Tk<:Kernel,Ni,No} <: Predictor
    x::Array{T,Ni}          # features
    y::Array{T,No}          # labels
    k::Tk                   # kernel
    λ::Float64              # regularization strength
    w::Array{T,No}          # weights
end


typealias KRR KernelRidgeRegressor

function KernelRidgeRegressor(x  ::AbstractArray,                           # features
                              y  ::AbstractArray;                           # labels
                              k  ::Kernel  = GaussianKernel(),              # kernel
                              λ  ::Float64 = 1e-4)                          # regularization strength
    KernelRidgeRegressor{eltype(x),typeof(k),ndims(x),ndims(y)}(x,y,k,λ,zeros(y))
end

# 1-dim features
function KernelRidgeRegressor(x  ::Vector{Float64},                         # features
                              y  ::Vector{Float64},                         # labels
                              k  ::Kernel = GaussianKernel(),               # kernel
                              λ  ::Float64 = 1e-4)                          # regularization strength
    KernelRidgeRegressor(reshape(x,(1,length(x))),y,k=k,λ=λ)
end


typealias sKRR{T,Tk<:Kernel,Ni} KRR{T,Tk,Ni,1} # single valued



function fit!(m::sKRR)
    # m.w = (K + m.λ*I)\m.y                             # solve for weights
    K = m.k(m.x)                                        # kernel matrix
    L = L_matrix(m.x,m.y)
    KL = K*L
    Λ = I/m.λ
    KLΛ = KL*Λ
    m.w = (K + KLΛ*KL')\(KLΛ*m.y)
end


# fit with new features, labels
function fit!(m::KRR,
              x::AbstractArray,                 # features
              y::AbstractArray)                 # labels
    m.x,m.y = x,y
    fit!(m)
end


Base.call(m::sKRR, x::AbstractVector) = dot(m.w,m.k(x,m.x))
Base.call(m::sKRR, x::AbstractArray) = m.k(x,m.x)*m.w

Base.call(m::KRR, x::AbstractVector) = m.w' * m.k(x,m.x)
Base.call(m::KRR, x::AbstractArray) = m.w' * m.k(x,m.x)'


grad(m::KRR, x::AbstractVector) = grad(m.k,x,m.x)*m.w

function grad(m::KRR, x::AbstractMatrix)
    n,d = size(m.x)
    k = size(x,2)
    reshape(reshape(grad(m.k,x,m.x),(n*k,d)) * m.w, (n,k))
end


∇ = grad









# # uncentered kernel matrix
# K = kpca._get_kernel(X_train)

# # scaled kPCA eigenvectors
# A = (kpca.alphas_/sqrt(kpca.lambdas_)).T

# # centering matrix
# H = empty((Nt,Nt))
# H.fill(-1.0/Nt)
# for i in range(Nt):
#   H[i,i] += 1

# # projection matrix
# Q = dot(H, dot(dot(A.T, A), H))






# def coeff(x, kernel=None, X=None, Q=None, K=None):
#   n = K.shape[0]
#   kx = squeeze(kernel(x, X))

#   # expansion coefficients of sample over training
#   # samples in uncentered feature space
#   c = dot(Q, kx - sum(K, axis=0)/n) + 1.0/n

#   return c


# def preimage(x0, c, kernel=None, X=None, max_steps=100, tol=1E-15):
#   x = x0.copy()

#   # fixed point iteration
#   for _ in range(max_steps):
#       kx = squeeze(kernel(x, X))
#       dx = dot(c*kx, X) / dot(c, kx) - x

#       x = x + dx

#       if dot(dx, dx) < tol:
#           break

#   return x









# abstract Predictor



# type KernelRidgeRegressor{T<:Kernel} <: Predictor
#   x::Matrix{Float64}      # features
#   y::Vector{Float64}      # labels
#   k::T                    # kernel
#   λ::Float64              # regularization strength
#   β::Float64              # gradient regularization strength
#   w::Vector{Float64}      # weights
# end


# typealias KRR KernelRidgeRegressor

# function KernelRidgeRegressor(x  ::Matrix{Float64},                       # features
#                             y  ::Vector{Float64};                         # labels
#                             k  ::Kernel  = GaussianKernel(),              # kernel
#                             λ  ::Float64 = 1e-4,                          # regularization strength
#                             β  ::Float64 = 1e-4)                          # gradient regularization strength
#   KernelRidgeRegressor{typeof(k)}(x,y,k,λ,β,[])
# end



# # 1-dim features
# # function KernelRidgeRegressor(x  ::Vector{Float64},                         # features
# #                               y  ::Vector{Float64},                         # labels
# #                               # dy ::Vector{Float64};                       # gradients
# #                               k  ::Kernel = GaussianKernel(),               # kernel
# #                               λ  ::Float64 = 1e-4,                          # regularization strength
# #                               β  ::Float64 = 1e-2)                          # gradient regularization strength
# #     KernelRidgeRegressor(reshape(x,(1,length(x))),y,reshape(dy,(1,length(dy))),k=k,λ=λ,β=β)
# # end



# # function loss(m::KernelRidgeRegressor)
# #     n,d = size(m.X)
# #     K = m.k(m.X)
# #     # δK = reshape(grad(m.k,m.X,m.X),(n*d,d))
# #     function loss(w)
# #         Kw = K*w
# #         sumabs2(Kw-m.y) + m.λ*dot(w,Kw) #+ m.β*sumabs2(δK*w - vec(m.dy))
# #         # m.β*sumabs(δK*w - vec(m.dy))
# #     end
# #     loss
# # end



# function fit!(m::KernelRidgeRegressor)
#   K = m.k(m.x)                                        # kernel matrix
#   m.w = (K + m.λ*I)\m.y                               # solve for weights
# end


# # function fit!(m::KernelRidgeRegressor;
# #               method     ::Symbol  = :l_bfgs,
# #               xtol       ::Float64 = 1e-8,
# #               ftol       ::Float64 = 1e-8,
# #               grtol      ::Float64 = 1e-8,
# #               iterations ::Int     = 1000)
# #     res = optimize(loss(m),zeros(size(m.X,2)),method=method,
# #         xtol=xtol,ftol=ftol,grtol=grtol,iterations=iterations)
# #     m.w = res.minimum
# #     res
# # end



# # fit with new features, labels
# function fit!(m::KernelRidgeRegressor,
#             x::Matrix{Float64},                       # features
#             y::Vector{Float64})                       # labels
#   m.x,m.y = x,y
#   fit!(m)
# end

# # function fit!(m::KernelRidgeRegressor,
# #               x::Vector{Float64},                       # features
# #               y::Vector{Float64})                       # labels
# #     fit!(m,reshape(x,(1,length(x))),y)
# # end


# Base.call(m::KernelRidgeRegressor, x::AbstractVector) = dot(m.w,m.k(x,m.x))
# Base.call(m::KernelRidgeRegressor, x::AbstractMatrix) = m.k(x,m.x)*m.w
# # Base.call(m::KernelRidgeRegressor, x::Number) = m([x])



# grad(m::KernelRidgeRegressor, x::AbstractVector) = grad(m.k,x,m.x)*m.w

# function grad(m::KRR, x::AbstractMatrix)
#   n,d = size(m.x)
#   k = size(x,2)
#   reshape(reshape(grad(m.k,x,m.x),(n*k,d)) * m.w, (n,k))
# end






# abstract Predictor



# type KernelRidgeRegressor{T<:Kernel} <: Predictor
#   X::Matrix{Float64}      # features
#   y::Vector{Float64}      # labels
#   k::T                    # kernel
#   λ::Float64              # regularization strength
#   w::Vector{Float64}      # weights
# end


# typealias KRR KernelRidgeRegressor

# function KernelRidgeRegressor(X ::Matrix{Float64},                            # features
#                             y ::Vector{Float64};                          # labels
#                             k ::Kernel  = GaussianKernel(),               # kernel
#                             λ ::Float64 = 1e-4)                           # regularization strength
#   KernelRidgeRegressor{typeof(k)}(X,y,k,λ,[])
# end



# # 1-dim features
# function KernelRidgeRegressor(x ::Vector{Float64},                            # features
#                             y ::Vector{Float64};                          # labels
#                             k ::Kernel = GaussianKernel(),                # kernel
#                             λ ::Float64 = 1e-4)                           # regularization strength
#   KernelRidgeRegressor(reshape(x,(1,length(x))),y,k=k,λ=λ)
# end


# function loss(m::KernelRidgeRegressor)
#   K = m.k(m.X)
#   function loss(w)
#       Kw = K*w
#       sumabs2(Kw-m.y) + m.λ*dot(w,Kw)
#   end
#   loss
# end


# function fit!(m::KernelRidgeRegressor;
#             method     ::Symbol  = :l_bfgs,
#             xtol       ::Float64 = 1e-8,
#             ftol       ::Float64 = 1e-8,
#             grtol      ::Float64 = 1e-8,
#             iterations ::Int     = 1000)
#   res = optimize(loss(m),zeros(size(m.X,2)),method=method,
#       xtol=xtol,ftol=ftol,grtol=grtol,iterations=iterations)
#   m.w = res.minimum
#   res
# end



# # function fit!(m::KernelRidgeRegressor)
# #     K = m.k(m.X)                                        # kernel matrix
# #     K[diagind(K)] += m.λ                                # regularization
# #     m.w = K\m.y                                         # solve for weights
# # end



# # fit with new features, labels
# function fit!(m::KernelRidgeRegressor,
#             X::Matrix{Float64},                       # features
#             y::Vector{Float64})                       # labels
#   m.X,m.y = X,y
#   fit!(m)
# end

# function fit!(m::KernelRidgeRegressor,
#             x::Vector{Float64},                       # features
#             y::Vector{Float64})                       # labels
#   fit!(m,reshape(x,(1,length(x))),y)
# end


# Base.call(m::KernelRidgeRegressor, x::AbstractVector) = dot(m.w,m.k(x,m.X))
# Base.call(m::KernelRidgeRegressor, x::AbstractMatrix) = m.k(x,m.X)*m.w
# Base.call(m::KernelRidgeRegressor, x::Number) = m([x])



# grad(m::KernelRidgeRegressor, x::AbstractVector) = grad(m.k,x,m.X)*m.w

# function grad(m::KRR, x::AbstractMatrix)
#   n,d = size(m.X)
#   k = size(x,2)
#   reshape(reshape(grad(m.k,x,m.X),(n*k,d)) * m.w, (n,k))
# end














type RandomFourierKernelMachine <: Predictor
    X::Matrix{Float64}              # features
    y::Vector{Float64}              # labels
    φ::Matrix{Float64}              # random features
    n::Int                          # input dimensionality
    d::Int                          # number of random features
    σ::Float64                      # kernel width
    λ::Float64                      # regularization strength
    w::Vector{Float64}              # weights
end


typealias RFKM RandomFourierKernelMachine



function RandomFourierKernelMachine(X ::Matrix{Float64},            # features
                                    y ::Vector{Float64};            # labels
                                    d ::Int     = 100,              # number of random features
                                    σ ::Float64 = 1.,               # kernel width
                                    λ ::Float64 = 1e-4)             # regularization strength
  RandomFourierKernelMachine(X,y,Matrix{Float64}(),size(X,1),d,σ,λ,[])
end


function kernel_matrix(m::RFKM, x::AbstractArray)
  z = exp(m.φ*x*im)/sqrt(m.d)
  real(z'*z)
end

function kernel_matrix(m::RFKM, x::AbstractArray, y::AbstractArray)
  zx = exp(m.φ*x*im)/sqrt(m.d)
  zy = exp(m.φ*y*im)/sqrt(m.d)
  real(zx'*zy)
end


function fit!(m::RFKM)
    d,n = m.d,m.n
    dist = Normal(0,1/m.σ)                      # normal distribution
    m.φ = rand(dist, (d,n))                     # random features
    z = cos(m.φ*m.X)/sqrt(m.d)                  # projection of X onto random features
    # z0 = centered ? zeros(n) : squeeze(mean(z,2),2)
    # y0 = centered ? zeros(n) : squeeze(mean(y,2),2)
    # zc = z .- z0
    # yc = y .- y0
    m.w = (z*z' + m.λ*I)\(z*m.y)
end


function Base.call(m::RFKM, x::AbstractVector)
    z = cos(m.φ*x)/sqrt(m.d)
    # y = Φ.α'*(z .- Φ.z0) .+ Φ.y0
    dot(m.w,z)
end

function Base.call(m::RFKM, x::AbstractMatrix)
    z = cos(m.φ*x)/sqrt(m.d)
    z'*m.w
end


function grad(m::RFKM, x::AbstractVector)
  dz = m.φ*im .* exp(m.φ*x*im)/sqrt(m.d)
  real(dz'*m.w)
end

function grad(m::RFKM, x::AbstractArray)
  n,d,k = m.n,m.d,size(x,2)
  c = -sin(m.φ*x)/sqrt(d)
  dz = m.φ .* reshape(c,(d,1,k))
  reshape(m.w' * reshape(dz,(d,n*k)), (n,k))
end






# H = I + fill(-1/n,size(K))
# H*K*H


doc"""
centering kernel matrices

$\varphi = \frac{1}{n} \sum_i \varphi(x_i)$

$\tilde K_{ij} = (\varphi(x_i) - \tilde\varphi)^T (\varphi(x_j) - \tilde\varphi)$

$\tilde K_{ij} = K_{ij} - \frac{1}{n} \sum_j K_{ij} - \frac{1}{n} \sum_i K_{ji} + \frac{1}{n^2} \sum_{ij} K_{ij}$

$\tilde L_{ij} = L_{ij} - \frac{1}{n} \sum_j K_{ij} - \frac{1}{n} \sum_i L_{ji} + \frac{1}{n^2} \sum_{ij} K_{ij}$

$\tilde M_{ij} = M_{ij} - \frac{1}{n} \sum_j L_{ij} - \frac{1}{n} \sum_i L_{ji} + \frac{1}{n^2} \sum_{ij} M_{ij}$

"""
function center(K)
    n = size(K,1)
    A = sum(K,1)/n
    K .- A .- A' + sum(K)/n^2
end

# J = fill(-1/n,(m,n))
# (L + J*K)*H
function center(K,L::Matrix)
    n = size(K,1)
    L .- sum(K,1)/n .- sum(L,2)/n + sum(K)/n^2
end

function center(K,L::Vector)
    n = size(K,1)
    L .- vec(sum(K,1))/n .- sum(L)/n + sum(K)/n^2
end



# M + L*J' + J*L' + J*K*J'
function center(K,L,M)
    n = size(K,1)
    M .- sum(L,2)/n .- sum(L,2)'/n + sum(K)/n^2
end








type KernelPCA{T<:Kernel}
    k::T                        # kernel
    x::Matrix{Float64}          # features
    K::Matrix{Float64}          # kernel matrix
    u::Vector{Float64}          # eigenvalues
    v::Matrix{Float64}          # eigenvectors
    n::Int                      # number of principal components
end

typealias KPCA KernelPCA


function KernelPCA(; k = GaussianKernel(), n::Int = -1)
    KernelPCA{typeof(k)}(k,Matrix{Float64}(),Matrix{Float64}(),[],Matrix{Float64}(),n)
end


function fit!(m::KPCA, x::Matrix{Float64})
    n = m.n
    m.x = x                                                 # features
    m.K = m.k(x)                                            # compute kernel matrix
    K = center(m.K)                                         # center kernel matrix
    u,v = eig(Symmetric(K))                                 # compute principal components
    ind = sortperm(u,rev=true)                              # sort eigenvalues
    u = u[ind]
    v = v[:,ind]
    n = n == -1 ? findlast(u.>0,true) : n                   # ignore zero eigenvalues
    m.u = u[1:n]                                            # select n eigenvalues
    m.v = v[:,1:n]                                          # select n eigenvectors
    m.v ./= sqrt(m.u)'                                      # normalize eigenvectors
    return
end


function Base.call(m::KPCA, x::Vector{Float64})
    kx = center(m.K,m.k(x,m.x))         # compute centered kernel matrix
    m.v' * kx                           # project features onto principal components
end

function Base.call(m::KPCA, x::Matrix{Float64})
    K = center(m.K,m.k(x,m.x))          # compute centered kernel matrix
    (K * m.v)'                          # project features onto principal components
end


function grad(m::KPCA, x::Vector{Float64})
    n,d = size(m.x)
    δK = grad(m.k,x,m.x)
    δK = δK .- sum(δK,2)/d
    δK * m.v
end

function grad(m::KPCA, x::Matrix{Float64})
    n,d = size(m.x)
    l = size(x,2)
    δK = grad(m.k,x,m.x)
    δK = δK .- sum(δK,3)/d
    reshape(reshape(δK,(n*l,d)) * m.v, (n,l,m.n))
end

∇ = grad




function negative_log_likelihood(φ::KPCA, y::AbstractVector)
    n = length(φ.u)
    z = φ.v' * y  # projection of labels onto KPCA eigenvectors
    σ1 = d -> sum(z[1:d].^2) / d
    σ2 = d -> sum(z[d+1:n].^2) / (n-d)
    Float64[d/n * log(σ1(d)) + (n-d)/n * log(σ2(d)) for d in 1:n-1]
end



function rde(φ::KPCA, y::AbstractVector)
    nll = negative_log_likelihood(φ,y)
    d = findfirst(diff(nll).<0)
end







end # module










# abstract Predictor

# abstract Loss


# type L2RegularizedQuadraticLoss <: Loss
#   λ::Float64
# end

# L2RegularizedQuadraticLoss(;λ=1e-8) = L2RegularizedQuadraticLoss(λ)



# type QuadraticLoss <: Loss end



# type L1RegularizedQuadraticLoss <: Loss
#   λ::Float64
# end

# L1RegularizedQuadraticLoss(;λ=1e-8) = L1RegularizedQuadraticLoss(λ)







# type KernelRidgeRegressor{Tk<:Kernel,Tl<:Loss} <: Predictor
#   k::Tk                   # kernel
#   l::Tl                   # loss model
#   X::Matrix{Float64}      # features
#   y::Vector{Float64}      # labels
#   w::Vector{Float64}      # weights
# end



# function KernelRidgeRegressor(X ::Matrix{Float64},                            # features
#                             y ::Vector{Float64};                          # labels
#                             k ::Kernel = GaussianKernel(),                # kernel
#                             l ::Loss   = L2RegularizedQuadraticLoss())    # loss
#   KernelRidgeRegressor{typeof(k),typeof(l)}(k,l,X,y,[])
# end



# # 1-dim features
# function KernelRidgeRegressor(x ::Vector{Float64},                            # features
#                             y ::Vector{Float64};                          # labels
#                             k ::Kernel = GaussianKernel(),                # kernel
#                             l ::Loss   = L2RegularizedQuadraticLoss())    # loss
#   KernelRidgeRegressor(reshape(x,(1,length(x))),y,k=k,l=l)
# end




# function loss{Tk<:Kernel}(m::KernelRidgeRegressor{Tk,L1RegularizedQuadraticLoss})
#   K = m.k(m.X)                                # kernel matrix
#   L = chol(K)                                 # cholesky factorization
#   function loss(w)
#       sumabs2(K*w-m.y) + m.l.λ*sumabs(L*w)
#   end
#   loss
# end


# function loss{Tk<:Kernel}(m::KernelRidgeRegressor{Tk,L2RegularizedQuadraticLoss})
#   K = m.k(m.X)
#   function loss(w)
#       Kw = K*w
#       sumabs2(Kw-m.y) + m.l.λ*dot(w,Kw)
#   end
#   loss
# end

# function loss_gradient!{Tk<:Kernel}(m::KernelRidgeRegressor{Tk,L2RegularizedQuadraticLoss})
#   K = m.k(m.X)
#   function loss_gradient!(w, dst)
#       Kw = K*w
#       dst[:] = 2*K*(Kw-m.y) + 2*m.l.λ*Kw
#   end
#   loss_gradient!
# end


# function loss{Tk<:Kernel}(m::KernelRidgeRegressor{Tk,QuadraticLoss})
#   K = m.k(m.X)
#   loss(w) = sumabs2(K*w-m.y)
#   loss
# end

# function loss_gradient!{Tk<:Kernel}(m::KernelRidgeRegressor{Tk,QuadraticLoss})
#   K = m.k(m.X)
#   function loss_gradient!(w, dst)
#       dst[:] = 2*K*(K*w-m.y)
#   end
#   loss_gradient!
# end




# @generated function fit!(m::KernelRidgeRegressor;
#                        method     ::Symbol  = :l_bfgs,
#                        xtol       ::Float64 = 1e-8,
#                        ftol       ::Float64 = 1e-8,
#                        grtol      ::Float64 = 1e-8,
#                        iterations ::Int     = 1000)
#   if method_exists(:loss_gradient!,(m,))  # if loss gradient is implemented
#       return quote
#           res = optimize(loss(m),loss_gradient!(m),zeros(size(m.X,2)),method=method,
#               xtol=xtol,ftol=ftol,grtol=grtol,iterations=iterations)
#           m.w = res.minimum
#           res
#       end
#   else
#       return quote
#           res = optimize(loss(m),zeros(size(m.X,2)),method=method,
#               xtol=xtol,ftol=ftol,grtol=grtol,iterations=iterations)
#           m.w = res.minimum
#           res
#       end
#   end
# end



# function fit!{Tk<:Kernel}(m::KernelRidgeRegressor{Tk,L2RegularizedQuadraticLoss})
#   K = m.k(m.X)                                        # kernel matrix
#   K[diagind(K)] += m.l.λ                              # regularization
#   m.w = K\m.y                                         # solve for weights
# end




# # fit with new features, labels
# function fit!(m::KernelRidgeRegressor,
#             X::Matrix{Float64},                       # features
#             y::Vector{Float64})                       # labels
#   m.X,m.y = X,y
#   fit!(m)
# end

# function fit!(m::KernelRidgeRegressor,
#             x::Vector{Float64},                       # features
#             y::Vector{Float64})                       # labels
#   fit!(m,reshape(x,(1,length(x))),y)
# end





# predict
# Base.call(m::KernelRidgeRegressor, x::AbstractVector) = dot(m.w,m.k(x,m.X))
# Base.call(m::KernelRidgeRegressor, x::AbstractMatrix) = m.k(x,m.X)*m.w
# Base.call(m::KernelRidgeRegressor, x::Number) = m([x])


