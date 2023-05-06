import sympy as sp
from math import pi,inf
import math
from sympy import  symbols
from sympy.solvers import nsolve
from sympy.solvers.solveset import solveset_real
from scipy.optimize import fsolve,root_scalar
import numpy as np
import matplotlib.pyplot as plt

# theta1, theta2, d3, phi ,x ,y = dynamicsymbols('theta1 theta2 d3 phi x y')

def find_roots(sym_exp, symbol, interval_array):
    roots = []
    for i in range(len(interval_array)):
        try:
            root = float(sp.nsolve(sym_exp,(interval_array[i], interval_array[i+1]+1.e-5), solver = 'bisect'))
            roots.append(root)
        except:
            continue
    return roots

q = [1.9350964181814378,-1.8236665044578992,4.690415759823429]
q_0_rad = q[0]
q_1_rad = q[1]


theta1_sym, theta2_sym, d3_sym, phi_sym ,x_sym ,y_sym, L_sym, R_sym, r_sym ,t_sym = symbols ('theta1_sym, theta2_sym, d3_sym, phi_sym ,x_sym ,y_sym L_sym, R_sym, r_sym, t_sym',real=True)

theta1_eq = (x_sym+r_sym*(sp.cos(phi_sym)*sp.cos(sp.pi/3)-sp.sin(phi_sym)*sp.sin(sp.pi/3))-R_sym*sp.cos(theta1_sym))**2 + (y_sym+r_sym*(sp.sin(phi_sym)*sp.cos(sp.pi/3) + sp.cos(phi_sym)*sp.sin(sp.pi/3))-R_sym*sp.sin(theta1_sym))**2 - L_sym**2
theta2_eq = (x_sym-R_sym*sp.cos(theta2_sym))**2 + (y_sym-R_sym*sp.sin(theta2_sym))**2 - L_sym**2
d3_eq = (x_sym+r_sym*sp.cos(phi_sym))**2+(y_sym+r_sym*sp.sin(phi_sym)+R_sym)**2-d3_sym**2


a = (theta1_eq-theta2_eq).expand()
b = (d3_eq-theta2_eq).expand()


ans = sp.solve([a,b],[x_sym,y_sym], dict = True)
x_phi = ans[0][x_sym]
y_phi = ans[0][y_sym]

theta2_eq_phi = theta2_eq.subs({x_sym:x_phi, y_sym:y_phi})
theta2_eq_half_tangent = theta2_eq_phi.subs({sp.sin(phi_sym):2*t_sym/(1+t_sym**2),sp.cos(phi_sym):((1-t_sym**2)/(1+t_sym**2)),sp.tan(phi_sym/2):t_sym})
t_num_eq = theta2_eq_half_tangent.subs({theta1_sym:q_0_rad,theta2_sym:q_1_rad,d3_sym:q[2],L_sym:3.5,R_sym:4,r_sym:2})
phi_num_eq = theta2_eq_phi.subs({theta1_sym:q_0_rad,theta2_sym:q_1_rad,d3_sym:q[2],L_sym:3.5,R_sym:4,r_sym:2})




init_guesses = np.linspace(-pi,pi,24)
phi = find_roots(phi_num_eq, phi_sym, init_guesses)





x = [float(x_phi.subs(({theta1_sym:q_0_rad,theta2_sym:q_1_rad,d3_sym:q[2],L_sym:3.5,R_sym:4,r_sym:2, phi_sym: angle}))) for angle in phi]
y = [float(y_phi.subs(({theta1_sym:q_0_rad,theta2_sym:q_1_rad,d3_sym:q[2],L_sym:3.5,R_sym:4,r_sym:2, phi_sym: angle}))) for angle in phi]

d =  [x,y,phi]



