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

theta,phi = var('theta phi')

#parciales 1
p1 = plot_implicit(Eq(partial11(4,2,theta,phi),0),(theta,0,pi),(phi,0,2*pi),show=False)
p2 = plot_implicit(Eq(partial21(4,2,theta,phi),0),(theta,0,pi),(phi,0,2*pi),show=False,line_color='r')
p3 = plot_implicit(Eq(partial31(4,2,theta,phi),0),(theta,0,pi),(phi,0,2*pi),show=False,line_color='g')
points1,action = p1[0].get_points()

points1 = np.array([(float("{0:.4f}".format(x_int.mid)), float("{0:.4f}".format(y_int.mid))) for x_int, y_int in points1])
points2,action = p2[0].get_points()
points2 = np.array([(float("{0:.4f}".format(x_int.mid)), float("{0:.4f}".format(y_int.mid))) for x_int, y_int in points2])
points3,action = p3[0].get_points()
points3 = np.array([(float("{0:.4f}".format(x_int.mid)), float("{0:.4f}".format(y_int.mid))) for x_int, y_int in points3])
points1 = set([tuple(x) for x in points1])
points2 = set([tuple(x) for x in points2])
points3 = set([tuple(x) for x in points3])


cool_points = set.intersection(points1,points2,points3)
#print(cool_points)
#p1.extend(p2)
#p1.extend(p3)
#p1.show()

curva1_pt = np.array([(sin(theta)*sin(phi),sin(theta)*cos(phi),cos(theta)) for theta,phi in points1])

xx = np.array([ np.float(pt[0]) for pt in curva1_pt ])
yy = np.array([ np.float(pt[1]) for pt in curva1_pt ])
zz = np.array([ np.float(pt[2]) for pt in curva1_pt ])


theta, phi = np.linspace(0, 2 * np.pi, 13), np.linspace(0, np.pi, 7)
THETA, PHI = np.meshgrid(theta, phi)
R = 1.0
X = R * np.sin(PHI) * np.cos(THETA)
Y = R * np.sin(PHI) * np.sin(THETA)
Z = R * np.cos(PHI)

mlab.figure(1, bgcolor=(1, 1, 1), fgcolor=(0, 0, 0), size=(600, 600))
mlab.clf()


mlab.mesh(X , Y ,Z, color=(0.0,0.5,0.5))
mlab.points3d(xx, yy, zz, scale_factor=0.05)

curva2_pt = np.array([(sin(theta)*sin(phi),sin(theta)*cos(phi),cos(theta)) for theta,phi in points2])

xx = np.array([ np.float(pt[0]) for pt in curva2_pt ])
yy = np.array([ np.float(pt[1]) for pt in curva2_pt ])
zz = np.array([ np.float(pt[2]) for pt in curva2_pt ])

mlab.points3d(xx, yy, zz, scale_factor=0.05,color=(0.2, 0.4, 0.5))

curva3_pt = np.array([(sin(theta)*sin(phi),sin(theta)*cos(phi),cos(theta)) for theta,phi in points3])

xx = np.array([ np.float(pt[0]) for pt in curva3_pt ])
yy = np.array([ np.float(pt[1]) for pt in curva3_pt ])
zz = np.array([ np.float(pt[2]) for pt in curva3_pt ])

mlab.points3d(xx, yy, zz, scale_factor=0.05,color=(0.2, 0.1, 0.6))

mlab.show()
#parciales2
"""

p1 = plot_implicit(Eq(partial12(4,2,theta,phi),0),(theta,0,pi),(phi,0,2*pi),show=False)
p2 = plot_implicit(Eq(partial22(4,2,theta,phi),0),(theta,0,pi),(phi,0,2*pi),show=False,line_color='r')
p3 = plot_implicit(Eq(partial32(4,2,theta,phi),0),(theta,0,pi),(phi,0,2*pi),show=False,line_color='g')
points1,action = p1[0].get_points()
points1 = np.array([(x_int.mid, y_int.mid) for x_int, y_int in points1])
points2,action = p2[0].get_points()
points2 = np.array([(x_int.mid, y_int.mid) for x_int, y_int in points2])
points3,action = p3[0].get_points()
points3 = np.array([(x_int.mid, y_int.mid) for x_int, y_int in points3])
points1 = set([tuple(x) for x in points1])
points2 = set([tuple(x) for x in points2])
points3 = set([tuple(x) for x in points3])


print(set.intersection(points1,points2,points3))
p1.extend(p2)
p1.extend(p3)
p1.show()"""
