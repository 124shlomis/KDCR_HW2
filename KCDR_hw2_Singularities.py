import sympy as sp
from sympy import  symbols
from math import pi,atan2,atan,sqrt,cos,sin
from sympy.solvers import nsolve
from sympy.solvers.solveset import solveset_real
from scipy.optimize import fsolve,root_scalar
from scipy import interpolate
import numpy as np
import matplotlib.pyplot as plt

R = 4.
r = 2.
L = 3.5
def inverse_kin_per_time_step(x,elbows):
    # x = x[0], y = x[1], phi(deg) = x[2]
    # q[0] = theta1, q[1] = theta 2 q[2] = theta3
    q = [0,0,0]
    phi = x[2]*np.pi/180 # translate to rad
    x1 = x[0] + r*cos(phi+pi/3)
    y1 = x[1] + r*sin(phi+pi/3)
    x2 = x[0]
    y2 = x[1]
    x3 = x[0] + r*cos(phi)
    y3 = x[1] + r*sin(phi)
    
    q[2] =sqrt(x3**2 + (y3+R)**2)
    
    # shortcuts for atan2
    a1 = 4*R**2*(x1**2+y1**2)-(L**2-x1**2-y1**2-R**2)**2
    b1 = L**2-x1**2-y1**2-R**2
    
    a2 = 4*R**2*(x2**2+y2**2)-(L**2-x2**2-y2**2-R**2)**2
    b2 = L**2-x2**2-y2**2-R**2
    
      
    q[0] = np.arctan2((-2*R*y1),(-2*R*x1))+ np.arctan2(elbows[0]*sqrt(a1),b1)
    q[1] = np.arctan2((-2*R*y2),(-2*R*x2))+ np.arctan2(elbows[1]*sqrt(a2),b2)
    
    return q
    
def inverse_kin(x,elbows):
    q_inv = [inverse_kin_per_time_step(list(x_time_step),elbows) for x_time_step in list(x.T)]
    return q_inv

def find_roots(sym_exp, symbol, interval_array):
    roots = []
    for i in range(len(interval_array)):
        try:
            root = float(sp.nsolve(sym_exp,(interval_array[i], interval_array[i+1]+1.e-5), solver = 'bisect'))
            roots.append(root)
        except:
            continue
    return roots

theta1_sym, theta2_sym, d3_sym, phi_sym ,x_sym ,y_sym, L_sym, R_sym, r_sym ,t_sym = symbols ('theta1_sym, theta2_sym, d3_sym, phi_sym ,x_sym ,y_sym L_sym, R_sym, r_sym, t_sym',real=True)

theta1_eq = (x_sym+r_sym*(sp.cos(phi_sym)*sp.cos(sp.pi/3)-sp.sin(phi_sym)*sp.sin(sp.pi/3))-R_sym*sp.cos(theta1_sym))**2 + (y_sym+r_sym*(sp.sin(phi_sym)*sp.cos(sp.pi/3) + sp.cos(phi_sym)*sp.sin(sp.pi/3))-R_sym*sp.sin(theta1_sym))**2 - L_sym**2
theta2_eq = (x_sym-R_sym*sp.cos(theta2_sym))**2 + (y_sym-R_sym*sp.sin(theta2_sym))**2 - L_sym**2
d3_eq = (x_sym+r_sym*sp.cos(phi_sym))**2+(y_sym+r_sym*sp.sin(phi_sym)+R_sym)**2-d3_sym**2

# Find Jacobians

#Jx

F = sp.Matrix([[theta1_eq], [theta2_eq], [d3_eq]])
x_vec = (x_sym,y_sym,phi_sym)

Jx = F.jacobian(x_vec)
for row in range(Jx.shape[0]):
    for col in range (Jx.shape[1]):
        # Jx[row,col] =  Jx[row,col].simplify()
        print(f'Jx_{row+1}{col+1} =  {Jx[row,col]}')

det_Jx = sp.det(Jx)

# Jq
q_vec = (theta1_sym,theta2_sym,d3_sym)
Fq = -1*F
Jq = Fq.jacobian(q_vec)
for row in range(Jq.shape[0]):
    for col in range (Jq.shape[1]):
        # Jq[row,col] =  Jq[row,col].simplify()
        print(f'Jq_{row+1}{col+1} =  {Jq[row,col]}')
        
det_Jq = sp.det(Jq)

# get q from x vector
elbows_permutations = [[1,1],[-1,-1],[1,-1],[-1,1]] 

x_sing = np.linspace (0,2,20001)
y_sing = np.zeros(x_sing.shape)
phi_sing = 10.*pi/180.*np.ones(x_sing.shape)
x_list = list(zip(x_sing, y_sing,phi_sing))
x_list = [list(x_time_step) for x_time_step in x_list]

q_dict = []
x_valid = []
for counter,elbow in enumerate(elbows_permutations):
    x_valid_elbow = []
    q_dict_elbow = {}
    for time_step in x_list:
        try:
            q_time_step= inverse_kin_per_time_step(time_step, elbow)
            q_dict_elbow[time_step[0]]=q_time_step
            x_valid_elbow.append(time_step[0])
        except:
            continue
    q_dict.append(q_dict_elbow)
    x_valid.append(x_valid_elbow)
    
# find and plot numerical det(Jx) as function of x

det_Jx = det_Jx.subs({L_sym : L, r_sym: r, R_sym : R , y_sym: 0., phi_sym : 10.*pi/180}).expand()
det_Jx_lambda = sp.lambdify([x_sym, theta1_sym,theta2_sym,d3_sym], det_Jx)

det_Jx_num = []
for elbow in q_dict:
    det_J_num_elbow = []
    for (key,value) in elbow.items():
        det_J_num_elbow.append(det_Jx_lambda(key,value[0],value[1],value[2]))
    det_Jx_num.append(det_J_num_elbow)
    
# plot graph with all det Jx for x in range

fig, axs = plt.subplots()
axs.set_title(f'Det Jx for various solutions')
axs.set_xlabel('x [m]')
axs.set_ylabel('Det Jx')
axs.plot(x_valid[0], det_Jx_num[0] , label = 'q1 - elbow [1, 1]',linestyle = '--')
axs.plot(x_valid[1], det_Jx_num[1] , label = 'q2 - elbow [-1,-1]',linestyle = '--')    
axs.plot(x_valid[2], det_Jx_num[2] , label = 'q3 - elbow [1,-1]',linestyle = '--')      
axs.plot(x_valid[3], det_Jx_num[3] , label = 'q4 - elbow [-1,1]',linestyle = '--')  
axs.grid('True')
axs.set_xlim([0,2])
axs.legend(fontsize = '10', loc = 'best')

fig.tight_layout() 

# find and plot numerical det(Jq) as function of x

det_Jq = det_Jq.subs({L_sym : L, r_sym: r, R_sym : R , y_sym: 0., phi_sym : 10.*pi/180}).expand()
det_Jq_lambda = sp.lambdify([x_sym, theta1_sym,theta2_sym,d3_sym], det_Jq)

det_Jq_num = []
for elbow in q_dict:
    det_J_num_elbow = []
    for (key,value) in elbow.items():
        det_J_num_elbow.append(det_Jq_lambda(key,value[0],value[1],value[2]))
    det_Jq_num.append(det_J_num_elbow)
    
# plot graph with all det Jq for x in range

fig2, axs2 = plt.subplots()
axs2.set_title(f'Det Jq for various solutions')
axs2.set_xlabel('x [m]')
axs2.set_ylabel('Det Jq')
axs2.plot(x_valid[0], det_Jq_num[0] , label = 'q1 - elbow [1, 1]')
axs2.plot(x_valid[1], det_Jq_num[1] , label = 'q2 - elbow [-1,-1]')       
axs2.plot(x_valid[2], det_Jq_num[2] , label = 'q3 - elbow [1,-1]')       
axs2.plot(x_valid[3], det_Jq_num[3] , label = 'q4 - elbow [-1,1]')   
axs2.grid('True')
axs2.set_xlim([0,2])
axs2.legend(fontsize = '10', loc = 'best')

fig2.tight_layout()             
       

# find exact x values for det Jx = 0:
x_sing_values = {}
for counter,elbow in enumerate(elbows_permutations):
    try:
        f = interpolate.interp1d(np.array(det_Jx_num[counter]), np.array(x_valid[counter]), assume_sorted = False)
        root_detJx = 0 
        x_sing_values[str(elbow)]=f(root_detJx)
    except:
        continue
