import sympy as sp
# from sympy import*
from math import pi
from sympy.physics.vector import init_vprinting
from sympy.physics.mechanics import dynamicsymbols
from sympy import  symbols
from scipy.optimize import fsolve

# theta1, theta2, d3, phi ,x ,y = dynamicsymbols('theta1 theta2 d3 phi x y')

q = [110,141,4.690415759823429]
q_0_rad = q[0]*pi/180
q_1_rad = q[1]*pi/180


# theta1, theta2, d3, phi ,x ,y, L, R, r ,t = symbols ('theta1, theta2, d3, phi ,x ,y L, R, r, t')

# # theta1_eq = (x+r*(sp.cos(phi)*cos(sp.pi/3)-sin(phi)*sin(sp.pi/3))-R*cos(theta1))**2 + (y+r*(sin(phi)*cos(sp.pi/3) + cos(phi)*sin(sp.pi/3))-R*sin(theta1))**2 - L**2
# theta1_eq = (x+r*sp.cos(phi +sp.pi/3)-R*sp.cos(theta1))**2 + (y+r*sp.sin(phi + sp.pi/3)-R*sp.sin(theta1))**2 - L**2
# theta2_eq = (x-R*sp.cos(theta2))**2 + (y-R*sp.sin(theta2))**2 - L**2
# d3_eq = (x+r*sp.cos(phi))**2+(y+r*sp.sin(phi)+R)**2-d3**2

# a = (theta1_eq-theta2_eq).expand()
# b = (d3_eq-theta2_eq).expand()

# ans = sp.solve([a,b],x,y, dict=True)
# x_phi = ans[0][x]
# y_phi = ans[0][y]
# # print(x_phi)
# # print(y_phi)


# theta2_eq_phi = theta2_eq.subs({x:x_phi, y:y_phi}).expand()
# theta2_eq_half_tangent = theta2_eq_phi.subs({sp.sin(phi):2*t/(1+t**2),sp.cos(phi):((1-t**2)/(1+t**2))})
# print(theta2_eq_half_tangent)
# phi_solve = sp.nsolve(d3_eq_half_tangent,t ,dict=true)




theta1_sym, theta2_sym, d3_sym, phi_sym ,x_sym ,y_sym, L_sym, R_sym, r_sym ,t_sym = symbols ('theta1_sym, theta2_sym, d3_sym, phi_sym ,x_sym ,y_sym L_sym, R_sym, r_sym, t_sym',real=True)

theta1_eq = (x_sym+r_sym*(sp.cos(phi_sym)*sp.cos(sp.pi/3)-sp.sin(phi_sym)*sp.sin(sp.pi/3))-R_sym*sp.cos(theta1_sym))**2 + (y_sym+r_sym*(sp.sin(phi_sym)*sp.cos(sp.pi/3) + sp.cos(phi_sym)*sp.sin(sp.pi/3))-R_sym*sp.sin(theta1_sym))**2 - L_sym**2
theta2_eq = (x_sym-R_sym*sp.cos(theta2_sym))**2 + (y_sym-R_sym*sp.sin(theta2_sym))**2 - L_sym**2
d3_eq = (x_sym+r_sym*sp.cos(phi_sym))**2+(y_sym+r_sym*sp.sin(phi_sym)+R_sym)**2-d3_sym**2

# # substitute numerical values
# theta1_eq = theta1_eq.subs({theta1_sym:q_0_rad,theta2_sym:q_1_rad,d3_sym:q[2],L_sym:3.5,R_sym:4,r_sym:2})
# theta2_eq = theta2_eq.subs({theta1_sym:q_0_rad,theta2_sym:q_1_rad,d3_sym:q[2],L_sym:3.5,R_sym:4,r_sym:2})
# d3_eq = d3_eq.subs({theta1_sym:q_0_rad,theta2_sym:q_1_rad,d3_sym:q[2],L_sym:3.5,R_sym:4,r_sym:2})

# sol1 = sp.solve([theta1_eq,theta1_eq,d3_eq],[x_sym,y_sym,phi_sym],dict= True)
a = (theta1_eq-theta2_eq)
b = (d3_eq-theta2_eq)


ans = sp.solve([a,b],[x_sym,y_sym],dict=True)
x_phi = ans[0][x_sym]
y_phi = ans[0][y_sym]

theta2_eq_phi = theta2_eq.subs({x_sym:x_phi, y_sym:y_phi})
theta2_eq_half_tangent = theta2_eq_phi.subs({sp.sin(phi_sym):2*t_sym/(1+t_sym**2),sp.cos(phi_sym):((1-t_sym**2)/(1+t_sym**2)),sp.tan(phi_sym/2):t_sym})
t_num_eq = theta2_eq_half_tangent.subs({theta1_sym:q_0_rad,theta2_sym:q_1_rad,d3_sym:q[2],L_sym:3.5,R_sym:4,r_sym:2})
# # t_solution = sp.solve(t_num_eq, t_sym, manual = True)
# # t_solution = fsolve(t_lambda, 100)
# # t_lambda = sp.lambdify(t_sym, t_num_eq)

# # t_num_eq = t_num_eq.simplify()
# # sol_t = sp.real_roots(t_num_eq)

# print(t_num_eq)

