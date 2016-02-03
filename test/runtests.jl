using ML
using Base.Test


x,y = rand(10),rand(10)

k = GaussianKernel(σ=0.3)
@test k(x,y) ≈ exp(-norm(x-y)^2/(2*k.σ^2))

k = PolynomialKernel(α=10.0,c=1.5,ξ=2.5)
@test k(x,y) ≈ (k.α*dot(x,y) + k.c)^k.ξ




x_train = linspace(0,1,30)[:,:]'
y_train = sin(vec(x_train))
x_test = linspace(0,1,100)[:,:]'
y_test = sin(vec(x_test))

for k in ( GaussianKernel(σ=0.5),
		   PolynomialKernel(α=5.,c=0.1,ξ=4.),
		   ExponentialKernel(σ=10) )
    m = KRR(x_train,y_train,k=k,λ=1e-8)
    fit!(m)
    @test mae(m(x_test),y_test) < 1e-4
end



