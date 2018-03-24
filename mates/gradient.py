import numpy as np  
import math
from scipy import special as sp
from sympy import plot_implicit, cos, sin, symbols, Eq,gegenbauer,pi,var
import matplotlib.pyplot as plt
 
def Y_1(n,k,theta,phi):
    return (sin(theta))**k * cos(k*phi)*gegenbauer(n-k,k+0.5,cos(theta))
def Y_2(n,k,theta,phi):
    return (sin(theta))**k * sin(k*phi)*gegenbauer(n-k,k+0.5,cos(theta))

def partial11(n,k,theta,phi):
    return -((n+k)*(n+k-1))/(2*(2*k-1))*Y_2(n-1,k-1,theta,phi)-(k+0.5)*Y_2(n-1,k+1,theta,phi)
def partial21(n,k,theta,phi):
    return ((n+k)*(n+k-1))/(2*(2*k-1))*Y_1(n-1,k-1,theta,phi)-(k+0.5)*Y_1(n-1,k+1,theta,phi)
def partial31(n,k,theta,phi):
    return (n+k)*Y_1(n-1,k,theta,phi)

def partial12(n,k,theta,phi):
    return ((n+k)*(n+k-1))/(2*(2*k-1))*Y_1(n-1,k-1,theta,phi)+(k+0.5)*Y_1(n-1,k+1,theta,phi)
def partial22(n,k,theta,phi):
    return ((n+k)*(n+k-1))/(2*(2*k-1))*Y_2(n-1,k-1,theta,phi)-(k+0.5)*Y_2(n-1,k+1,theta,phi)
def partial32(n,k,theta,phi):
    return (n+k)*Y_2(n-1,k,theta,phi)

theta,phi = var('theta phi')


p1 = plot_implicit(Eq(partial11(7,2,theta,phi)),(theta,0,pi),(phi,0,2*pi),show=False)
p2 = plot_implicit(Eq(partial21(7,2,theta,phi)),(theta,0,pi),(phi,0,2*pi),show=False,line_color='r')
p3 = plot_implicit(Eq(partial31(7,2,theta,phi)),(theta,0,pi),(phi,0,2*pi),show=False,line_color='g')
points1,action = p1[0].get_points()
points1 = np.array([(x_int.mid, y_int.mid) for x_int, y_int in points1])
points2,action = p2[0].get_points()
points2 = np.array([(x_int.mid, y_int.mid) for x_int, y_int in points2])
points3,action = p3[0].get_points()
points3 = np.array([(x_int.mid, y_int.mid) for x_int, y_int in points3])
points1 = set([tuple(x) for x in points1])
points2 = set([tuple(x) for x in points2])
points3 = set([tuple(x) for x in points3])
print(set.intersection(points1,points2))
p1.extend(p2)
p1.extend(p3)
p1.show()