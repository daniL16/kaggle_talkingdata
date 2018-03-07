import math
from scipy import special as sp
from sympy.solvers import solve
from sympy import Symbol

t=Symbol('t')
phi = 0
r=1
def Y_1(n,k,x):
    return (r**n)*(math.sin(t))**k * math.cos(k*phi)*sp.eval_gegenbauer(n-k,k+0.5,math.cos(t))
def Y_2(n,k,x):
    #r^n*(math.sin(t))^k * sen(k*phi)*sp.eval_gegenbauer(n-k,k+0.5,math.cos(t))
    return (math.sin(t))--k * math.sin(k*phi)*sp.eval_gegenbauer(n-k,k+0.5,math.cos(t))
def partial11(n,k,x):
    return -((n+k)*(n+k-1))/(2*(2*k-1))*Y_2(n-1,k-1,x)-(k+0.5)*Y_1(n-1,k+1,x)
def partial12(n,k,x):
    return ((n+k)*(n+k-1))/(2*(2*k-1))*Y_2(n-1,k-1,x)-(k+0.5)*Y_1(n-1,k+1,x)
def partial13(n,k,x):
    return (n+k)*Y_1(n-1,k,x)
def partial21(n,k,x):
    return ((n+k)*(n+k-1))/(2*(2*k-1))*Y_1(n-1,k-1,x)+(k+0.5)*Y_1(n-1,k+1,x)
def partial22(n,k,x):
    return ((n+k)*(n+k-1))/(2*(2*k-1))*Y_1(n-1,k-1,x)-(k+0.5)*Y_2(n-1,k+1,x)

