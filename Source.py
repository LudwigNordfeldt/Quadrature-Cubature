import matplotlib.pyplot as mp
from mpl_toolkits.mplot3d import axes3d
import numpy as np
from scipy import optimize
import sympy
from scipy.special import legendre
from scipy.misc import derivative
import scipy.integrate as spint


f = lambda x,y: x*x + y*y
t2 = lambda x: 1 - x**4 
t1 = lambda x: -1 + x**4

def func (x,y):
    if (x**4+y**4 <= 1):
        return x*x + y*y
    else:
        return 0

def subint(a,b,c,d,fun):
    u,v = sympy.symbols('u,v')
    
    f1 = lambda u: (b-a)*u + a
    f2 = lambda u,v: c(f1(u)) + v*( d( f1(u) ) - c( f1(u) ) )

    X = sympy.Matrix([(b-a)*u + a, c(f1(u)) + v*( d( f1(u) ) - c( f1(u) ) ) ])
    Y = sympy.Matrix([u,v])
    J = X.jacobian(Y)
    Jac = J.det()
    
    I = sympy.sympify( fun(f1(u),f2(u,v))*abs(Jac) )
    
    FI = sympy.lambdify( (u,v), I)
    
    return FI
    

def quadrature(fun,a,b,n):
    I = 0
    M = (b-a)/n
    s = np.linspace(a,b,n)
    for i in range(n):
        for j in range(n):
            fij = fun(s[i],s[j])
            if ( (i in [0,n]) and (j in [0,n]) ):
                I += 0.25*fij
            elif ( (i in [0,n]) or (j in [0,n]) ):
                I += 0.5*fij
            else:
                I += fij

    return I*M*M

    
def cubature(a,b,n,fun,constr1,constr2):
    M = (b-a)/2
    Leg = legendre(n)
    R = np.roots(Leg)
    T = len(R)
    Deriv = [ derivative(Leg, r) for r in R]

    weights = []
    for i in range(T):
        weights.append( 2 / ( (1-R[i]*R[i]) * Deriv[i]*Deriv[i] ) )

    u = (a+b)*0.5 + R[0]*0.5*(b-a)
    v = 0.5 * (constr1(u) + constr2(u)) + R[0]*0.5*( constr2(u) - constr1(u) )
        
    I = 0
    s = np.linspace(a,b,n)
    for i in range(T):
        u = (a+b)/2 + R[i]*0.5*(b-a)
        I += weights[i]*0.5*( constr2(u) - constr1(u) )
        
        for j in range(T):
            v = 0.5 * (constr1(u) + constr2(u)) + R[j]*0.5*( constr2(u) - constr1(u) )
            fij = fun(u,v)
            I += fij*weights[j]

    return I*M

def cubature2(fun,a,A,b,B,n):
    h = (A-a)/n
    k = (B-b)/n
    x = a
    y = b
    I = 0

    for i in range(0,n):
        for j in range(0,n):
            I += fun(x,y) + fun(x,y+k) + fun(x+h,y) + fun(x+h,y+k)
            x += h
        x = a
        y += k

    res = (h*k/4)*I
    return res

print ( quadrature ( subint(-1, 1, t1, t2, f), 0, 1, 800 ) )
print()
print ( cubature2 ( func, -1, 1, -1, 1, 9 ) )
print()
print ( spint.dblquad (f, -1 ,1, t1, t2) )

xy = np.linspace(-2,2)
X,Y = np.meshgrid(xy,xy)
Z = X*X + Y*Y

fig = mp.figure()
ax = fig.add_subplot(1,1,1, projection='3d')
ax.plot_wireframe(X,Y,Z, rstride=4, cstride=4)


mp.show()
