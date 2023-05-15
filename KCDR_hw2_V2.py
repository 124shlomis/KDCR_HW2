import numpy as np
import matplotlib.pyplot as plt
from matplotlib import style
from scipy.optimize import bisect
from math import pi,sqrt,cos,sin
import sympy as sp
from sympy import  symbols

# global constants


R = 4.
r = 2.
L = 3.5

def find_roots(sym_exp, symbol, interval_array):
    roots = []
    sym_exp =  sp.lambdify(symbol,sym_exp)
    for i in range(len(interval_array)):
        try:
            root = float(bisect(sym_exp,interval_array[i], interval_array[i+1]+1.e-5))
            roots.append(root)
        except:
            continue
    return roots

def inverse_kin_per_elbow(x,elbows):
    # input :: type == list(3) : x = x[0], y = x[1], phi(deg) = x[2]
    # output :: type == list(3) : q[0] = theta1, q[1] = theta 2 q[2] = theta3
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

def inverse_kinematics(x):
    # input :: type == list(3) : x[m] = x[0], y[m] = x[1], phi(deg) = x[2]
    # output :: type == np.array (4,,3) : matrix for all possible solutions  when each row of matrix represents possible sol:
    # q[0] = theta1 [rad], q[1] = theta2[rad] ,q[2] = d3[m]
    elbows_permutations = [[1,1],[-1,-1],[1,-1],[-1,1]]
    q_poss = []
    for count,elbow_poss in enumerate(elbows_permutations):
        q_poss.append(inverse_kin_per_elbow(x,elbow_poss))
    # tool coordinates
    phi = x[2]*pi/180 # translate to rad
    x1 = x[0] + r*cos(phi+(pi/3))
    y1 = x[1] + r*sin(phi+(pi/3))
    x2 = x[0]
    y2 = x[1]
    x3 = x[0] + r*cos(phi)
    y3 = x[1] + r*sin(phi)
    # plot circle
    theta = np.linspace(0, 2*np.pi, 100)
    R_rob = R
    x_circ = R_rob*np.cos(theta)
    y_circ = R_rob*np.sin(theta)
   
        
    plt.rcParams['axes.grid'] = True
    fig_width = 8
    fig_height = 8.5
    graph_style = 'seaborn-white'
    tool_plot =[]    
    style.use(graph_style)
    for count,q in enumerate(q_poss):
        fig,axs = plt.subplots()
        tool_plot.append(plt.Polygon([[x1, y1],
                            [x2, y2],
                            [x3, y3]],facecolor='gray',edgecolor='black', label='_nolegend_'))        
       
        axs.plot(x_circ,y_circ, color = 'black',label='_nolegend_', linewidth = 3)
        axs.add_patch(tool_plot[-1])    
        axs.plot([x1,R*cos(q[0])],[y1,R*sin(q[0])], color = 'blue',marker = 'o', label = 'theta1')
        axs.plot([x2,R*cos(q[1])],[y2,R*sin(q[1])], color = 'green',marker = 'o', label = 'theta2')
        axs.plot([x3,0],[y3,-R], color = 'red', marker = 'o',label = 'd3')
        axs.set(xlabel='x [m]', ylabel='y [m]')
        axs.legend(['theta1','theta2','d3'], loc= 'upper right')
        axs.set_title(f'Inverse kinematics config for elbows {str(elbows_permutations[count])}')
        axs.set_aspect('equal', 'box')
        axs.grid()
        
        fig.tight_layout()
        plt.show()

    return np.array(q_poss)

def filter_solutions(d):
    # input : numpy matrix of all mathematical solutions for tool position
    # output : numpy matrix of all physicly allowed solutions  d
    d_valid = []
    for sol in list(d):
        if sqrt((sol[0]+r*cos(sol[2]))**2 + (sol[1]+r*sin(sol[2]))**2)>R:
            continue
        d_valid.append(sol)
    return np.array(d_valid)
        
def forward_kinematics(q):
    # input :: type == list(3) : theta1 [rad] = q[0], theta2 [rad] = q[1], d3[m] = q[2]
    # output :: type == np.array (var,3) : matrix for all possible solutions  when each row of matrix represents possible sol:
    q_0_rad = q[0]
    q_1_rad = q[1]
    
    theta1_sym, theta2_sym, d3_sym, phi_sym ,x_sym ,y_sym, L_sym, R_sym, r_sym ,t_sym = symbols ('theta1_sym, theta2_sym, d3_sym, phi_sym ,x_sym ,y_sym L_sym, R_sym, r_sym, t_sym')

    theta1_eq = (x_sym+r_sym*(sp.cos(phi_sym)*sp.cos(sp.pi/3)-sp.sin(phi_sym)*sp.sin(sp.pi/3))-R_sym*sp.cos(theta1_sym))**2 + (y_sym+r_sym*(sp.sin(phi_sym)*sp.cos(sp.pi/3) + sp.cos(phi_sym)*sp.sin(sp.pi/3))-R_sym*sp.sin(theta1_sym))**2 - L_sym**2  
    theta2_eq = (x_sym-R_sym*sp.cos(theta2_sym))**2 + (y_sym-R_sym*sp.sin(theta2_sym))**2 - L_sym**2
    d3_eq = (x_sym+r_sym*sp.cos(phi_sym))**2+(y_sym+r_sym*sp.sin(phi_sym)+R_sym)**2-d3_sym**2
    
    a = (theta1_eq-theta2_eq).expand()
    b = (d3_eq-theta2_eq).expand()

    ans = sp.solve([a,b],x_sym,y_sym, dict=True)
    x_phi = ans[0][x_sym]
    y_phi = ans[0][y_sym]

    theta2_eq_phi =theta2_eq.subs({x_sym:x_phi, y_sym:y_phi})
    phi_num_eq = theta2_eq_phi.subs({theta1_sym:q_0_rad,theta2_sym:q_1_rad,d3_sym:q[2],L_sym:3.5,R_sym:4,r_sym:2})
   
    init_guesses = np.linspace(-pi,pi,36)
    phi = find_roots(phi_num_eq, phi_sym, init_guesses)


    x = [float(x_phi.subs(({theta1_sym:q_0_rad,theta2_sym:q_1_rad,d3_sym:q[2],L_sym:3.5,R_sym:4,r_sym:2, phi_sym: angle}))) for angle in phi]
    y = [float(y_phi.subs(({theta1_sym:q_0_rad,theta2_sym:q_1_rad,d3_sym:q[2],L_sym:3.5,R_sym:4,r_sym:2, phi_sym: angle}))) for angle in phi]
    d= np.array([x , y, phi]).T
    # check for non-feasible solutions ( e.g d<0)
    d = filter_solutions(d)
        
    plt.rcParams['axes.grid'] = True
    graph_style = 'seaborn-white'
    
    # plot circle
    style.use(graph_style)
    theta = np.linspace(0, 2*np.pi, 100)
    R_rob = R
    x_circ = R_rob*np.cos(theta)
    y_circ = R_rob*np.sin(theta)
    tool_plot = []
    for count,config in enumerate(list(d)):
        phi = config[2] 
        x1 = config[0] + r*cos(phi+(pi/3))
        y1 = config[1] + r*sin(phi+(pi/3))
        x2 = config[0]
        y2 = config[1]
        x3 = config[0] + r*cos(phi)
        y3 = config[1] + r*sin(phi)
        tool_plot.append(plt.Polygon([[x1, y1],
                            [x2, y2],
                            [x3, y3]],facecolor='gray',edgecolor='black',label='_nolegend_'))
        fig, axs = plt.subplots()
        axs.set_title(f'Forward kinematics config {count+1}')
        axs.set_xlabel('x [m]')
        axs.set_ylabel('y [m]')       
        axs.plot(x_circ,y_circ, color = 'black',label='_nolegend_', linewidth = 3)
        axs.add_patch(tool_plot[-1])
        axs.plot([x1,R*cos(q[0])],[y1,R*sin(q[0])], color = 'blue',marker = 'o', label = 'theta1')
        axs.plot([x2,R*cos(q[1])],[y2,R*sin(q[1])], color = 'green',marker = 'o', label = 'theta2')
        axs.plot([x3,0],[y3,-R], color = 'red', marker = 'o',label = 'd3')
        axs.set_aspect('equal', 'box')
        axs.legend(['theta1','theta2','d3'])
        axs.grid()
        
        fig.tight_layout()
        plt.show()
        
    return d

def main():
    plt.close('all')
    x = [-3, -1 , 45.]
    q = inverse_kinematics(x)
    d = forward_kinematics(list(q)[1])
    # print(q)
    # print(d)

if __name__ == '__main__':
    main()
