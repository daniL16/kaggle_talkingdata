import numpy as np  
import math
from scipy import special as sp
from sympy import plot_implicit, cos, sin, symbols, Eq,gegenbauer,pi,var
import matplotlib.pyplot as plt
from mayavi import mlab
from mpl_toolkits.mplot3d import Axes3D

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

def printPoints(points,color):
    curva_pt = np.array([(sin(theta)*sin(phi),sin(theta)*cos(phi),cos(theta)) for theta,phi in points])

    xx = np.array([ np.float(pt[0]) for pt in curva_pt ])
    yy = np.array([ np.float(pt[1]) for pt in curva_pt ])
    zz = np.array([ np.float(pt[2]) for pt in curva_pt ])
    
    mlab.points3d(xx, yy, zz, scale_factor=0.05,color=color)


theta,phi = var('theta phi')
n = 20
k = 9
#parciales 1
p1 = plot_implicit(Eq(partial11(n,k,theta,phi),0),(theta,0,pi),(phi,0,2*pi),show=False)
p2 = plot_implicit(Eq(partial21(n,k,theta,phi),0),(theta,0,pi),(phi,0,2*pi),show=False,line_color='r')
p3 = plot_implicit(Eq(partial31(n,k,theta,phi),0),(theta,0,pi),(phi,0,2*pi),show=False,line_color='g')

#obtener puntos de las graficas
points1,action = p1[0].get_points()
points2,action = p2[0].get_points()
points3,action = p3[0].get_points()
points1 = np.array([(float("{0:.5f}".format(x_int.mid)), float("{0:.5f}".format(y_int.mid))) for x_int, y_int in points1])
points2 = np.array([(float("{0:.5f}".format(x_int.mid)), float("{0:.5f}".format(y_int.mid))) for x_int, y_int in points2])
points3 = np.array([(float("{0:.5f}".format(x_int.mid)), float("{0:.5f}".format(y_int.mid))) for x_int, y_int in points3])
#np.array a set
points1 = set([tuple(x) for x in points1])
points2 = set([tuple(x) for x in points2])
points3 = set([tuple(x) for x in points3])

#puntos de corte de las curvas
cool_points = set.intersection(points1,points2,points3)
#print(cool_points)
#p1.extend(p2)
#p1.extend(p3)
#p1.show()

#pinta la esfera
theta, phi = np.linspace(0, 2 * np.pi, 50), np.linspace(0, np.pi, 25)
THETA, PHI = np.meshgrid(theta, phi)
R = 1.0
X = R * np.sin(PHI) * np.cos(THETA)
Y = R * np.sin(PHI) * np.sin(THETA)
Z = R * np.cos(PHI)
mlab.figure(1, bgcolor=(1, 1, 1), fgcolor=(0, 0, 0), size=(600, 600))
mlab.clf()
mlab.mesh(X , Y ,Z, color=(0.9,0.9,0.9))

printPoints(cool_points,color=(1,0,0.3))
name = 'points'+str(n)+str(k)+'.png'
#mlab.savefig(name)
mlab.show()
