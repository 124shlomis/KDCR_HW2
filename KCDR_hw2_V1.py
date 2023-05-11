import numpy as np
import matplotlib.pyplot as plt
from matplotlib import style
from scipy.optimize import bisect
from math import pi,atan2,atan,sqrt,cos,sin
import itertools
import sympy as sp
from sympy.physics.mechanics import dynamicsymbols
from sympy import diff, symbols

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

def forward_kin_per_time_step(q):
    #q[0] - theta 1 , q[1] - theta2 , q[2] - d3
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
    theta2_eq_half_tangent = theta2_eq_phi.subs({sp.sin(phi_sym):2*t_sym/(1+t_sym**2),sp.cos(phi_sym):((1-t_sym**2)/(1+t_sym**2)),sp.tan(phi_sym/2):t_sym})
    phi_num_eq = theta2_eq_phi.subs({theta1_sym:q_0_rad,theta2_sym:q_1_rad,d3_sym:q[2],L_sym:3.5,R_sym:4,r_sym:2})
   
    init_guesses = np.linspace(-pi,pi,36)
    phi = find_roots(phi_num_eq, phi_sym, init_guesses)


    x = [float(x_phi.subs(({theta1_sym:q_0_rad,theta2_sym:q_1_rad,d3_sym:q[2],L_sym:3.5,R_sym:4,r_sym:2, phi_sym: angle}))) for angle in phi]
    y = [float(y_phi.subs(({theta1_sym:q_0_rad,theta2_sym:q_1_rad,d3_sym:q[2],L_sym:3.5,R_sym:4,r_sym:2, phi_sym: angle}))) for angle in phi]
    d_calc = np.array([x , y, phi])    
    return d_calc



def draw_inv_per_time_step(x):
    # draws all possibilites of invers kin for given x
    elbows_permutations = [[1,1],[-1,-1],[1,-1],[-1,1]] 
    q_poss = []
    for count,elbow_poss in enumerate(elbows_permutations):
        q_poss.append(inverse_kin_per_time_step(x,elbow_poss))
           
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
        
    style.use(graph_style)
    fig2, axs = plt.subplots(2, 2,figsize=(fig_width, fig_height), tight_layout=True)
    fig2.suptitle("Joints values for different possibilities of inverse kinematics ", fontsize=16)
    
    tool2= plt.Polygon([[x1, y1],
                       [x2, y2],
                       [x3, y3]],facecolor='gray',edgecolor='black',label='_nolegend_')
    axs[0, 0].plot(x_circ,y_circ, color = 'black',label='_nolegend_', linewidth = 3)
    axs[0, 0].add_patch(tool2)    
    axs[0, 0].plot([x1,R*cos(q_poss[0][0])],[y1,R*sin(q_poss[0][0])], color = 'blue',marker = 'o', label = 'theta1')
    axs[0, 0].plot([x2,R*cos(q_poss[0][1])],[y2,R*sin(q_poss[0][1])], color = 'green',marker = 'o', label = 'theta2')
    axs[0, 0].plot([x3,0],[y3,-R], color = 'red', marker = 'o',label = 'd3')
    axs[0, 0].set(xlabel='x [m]', ylabel='y [m]')
    axs[0, 0].legend(['theta1','theta2','d3'], loc= 'upper right')
    axs[0, 0].set_title('Config 1')
    axs[0, 0].grid()

   
    tool3= plt.Polygon([[x1, y1],
                        [x2, y2],
                        [x3, y3]],facecolor='gray',edgecolor='black',label='_nolegend_')
    axs[0, 1].plot(x_circ,y_circ, color = 'black',label='_nolegend_', linewidth = 3) 
    axs[0, 1].add_patch(tool3)    
    axs[0, 1].plot([x1,R*cos(q_poss[1][0])],[y1,R*sin(q_poss[1][0])], color = 'blue',marker = 'o', label = 'theta1')
    axs[0, 1].plot([x2,R*cos(q_poss[1][1])],[y2,R*sin(q_poss[1][1])], color = 'green',marker = 'o', label = 'theta2')
    axs[0, 1].plot([x3,0],[y3,-R], color = 'red', marker = 'o',label = 'd3')
    axs[0, 1].set(xlabel='x [m]', ylabel='y [m]')
    axs[0, 1].legend(['theta1','theta2','d3'],loc= 'upper right')
    axs[0, 1].set_title('Config 2')
    axs[0, 1].grid()



    tool4= plt.Polygon([[x1, y1],
                    [x2, y2],
                    [x3, y3]],facecolor='gray',edgecolor='black',label='_nolegend_')    
    axs[1, 0].plot(x_circ,y_circ, color = 'black',label='_nolegend_', linewidth = 3)    
    axs[1, 0].add_patch(tool4)    
    axs[1, 0].plot([x1,R*cos(q_poss[2][0])],[y1,R*sin(q_poss[2][0])], color = 'blue',marker = 'o', label = 'theta1')
    axs[1, 0].plot([x2,R*cos(q_poss[2][1])],[y2,R*sin(q_poss[2][1])], color = 'green',marker = 'o', label = 'theta2')
    axs[1, 0].plot([x3,0],[y3,-R], color = 'red', marker = 'o',label = 'd3')
    axs[1, 0].set(xlabel='x [m]', ylabel='y [m]')
    axs[1, 0].legend(['theta1','theta2','d3'],loc= 'upper right')
    axs[1, 0].set_title('Config 3')
    axs[1, 0].grid()



    tool5= plt.Polygon([[x1, y1],
                    [x2, y2],
                    [x3, y3]],facecolor='gray',edgecolor='black',label='_nolegend_')
    axs[1, 1].plot(x_circ,y_circ, color = 'black',label='_nolegend_', linewidth = 3)
    axs[1, 1].add_patch(tool5)    
    axs[1, 1].plot([x1,R*cos(q_poss[3][0])],[y1,R*sin(q_poss[3][0])], color = 'blue',marker = 'o', label = 'theta1')
    axs[1, 1].plot([x2,R*cos(q_poss[3][1])],[y2,R*sin(q_poss[3][1])], color = 'green',marker = 'o', label = 'theta2')
    axs[1, 1].plot([x3,0],[y3,-R], color = 'red', marker = 'o',label = 'd3')
    axs[1, 1].set(xlabel='x [m]', ylabel='y [m]')
    axs[1, 1].legend(['theta1','theta2','d3'],loc= 'upper right')
    axs[1, 1].set_title('Config 4')
    axs[1, 1].grid()

  
    fig2.tight_layout()
    return

def draw_tool_traj(x,q,print_interval):
    graph_style = 'seaborn-white'
    # plot circle
    x = list(x.T)
    # filter print values from vector:
    x_filter = [x[ind] for ind,time_step in enumerate(t) if time_step in print_interval]
    q_filter = [q[ind] for ind,time_step in enumerate(t) if time_step in print_interval]
    style.use(graph_style)
    theta = np.linspace(0, 2*np.pi, 100)
    R_rob = R
    x_circ = R_rob*np.cos(theta)
    y_circ = R_rob*np.sin(theta)
    tool_plot = []
    fig, axs = plt.subplots()
    axs.set_title(f'Tool trajectory path')
    axs.set_xlabel('x [m]')
    axs.set_ylabel('y [m]')       
    axs.plot(x_circ,y_circ, color = 'black',label='_nolegend_', linewidth = 3)
    for xt,qt in zip(x_filter,q_filter):
        phi =xt[2]*pi/180 
        x1 = xt[0] + r*cos(phi+(pi/3))
        y1 = xt[1] + r*sin(phi+(pi/3))
        x2 = xt[0]
        y2 = xt[1]
        x3 = xt[0] + r*cos(phi)
        y3 = xt[1] + r*sin(phi)
        tool_plot.append(plt.Polygon([[x1, y1],
                            [x2, y2],
                            [x3, y3]],facecolor='gray',edgecolor='black', label='_nolegend_'))        
        axs.add_patch(tool_plot[-1])
        axs.plot([x1,R*cos(qt[0])],[y1,R*sin(qt[0])], color = 'blue',marker = 'o', label = 'theta1')
        axs.plot([x2,R*cos(qt[1])],[y2,R*sin(qt[1])], color = 'green',marker = 'o', label = 'theta2')
        axs.plot([x3,0],[y3,-R], color = 'red', marker = 'o',label = 'd3')
        axs.set_aspect('equal', 'box')
        axs.legend(['theta1','theta2','d3'])
        axs.grid()
        
        fig.tight_layout()
    
    # draw joints values:
    q_array = np.array(q)
    style.use(graph_style)
    fig2, axs2 = plt.subplots(3, 1,figsize = (6,8), tight_layout=True)
    fig2.suptitle("Joints values for constant velocity profile", fontsize=16)

    axs2[0].plot(t, q_array[:,0]*180/pi)
    axs2[0].set_title(r'$\theta_1$ ')
    axs2[0].set(xlabel='time [sec]', ylabel='angle [deg]')
    axs2[0].grid()
   
    axs2[1].plot(t, q_array[:,1]*180/pi)
    axs2[1].set_title(r'$\theta_2 $ ')
    axs2[1].set(xlabel='time [sec]', ylabel='angle [deg]')
    axs2[1].grid()


    axs2[2].plot(t, q_array[:,2])
    axs2[2].set_title(r'$d_3$')
    axs2[2].set(xlabel='time [sec]', ylabel='pos [m]')
    axs2[2].grid()

    
    fig2.tight_layout()
    
    return

def draw_forward_kin_per_time_step(q):
    
    d = list(forward_kin_per_time_step(q).T)
    
    plt.rcParams['axes.grid'] = True
    graph_style = 'seaborn-white'
    
    # plot circle
    style.use(graph_style)
    theta = np.linspace(0, 2*np.pi, 100)
    R_rob = R
    x_circ = R_rob*np.cos(theta)
    y_circ = R_rob*np.sin(theta)
    tool_plot = []
    for count,config in enumerate(d):
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
        axs.set_title(f'Config {count+1}')
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
        
    return np.array(d).T

def rad2deg(x):
    x[2] = x[2]*180/pi
    return x

def inverse_forward_error(q):
    err = []
    d_poss = forward_kin_per_time_step(q)
    d_poss = np.array([rad2deg(d) for d in list(d_poss.T)]).T
    # all posible solutions
    q_poss = []
    for tool_pos in (list(d_poss.T)):
        for elbow in elbows_permutations:
            q_poss.append(inverse_kin_per_time_step(tool_pos,elbow))
    # filter q solutions based on tolerance value
    q_filt = []
    for q_sol in q_poss:
        if abs(q_sol[0]-q[0])<= 1.e-5  and abs(q_sol[1]-q[1])<= 1.e-5  and abs(q_sol[2]-q[2])<= 1.e-5:
            q_filt.append(np.array(q_sol))
            break
    #compare error:
    for q_sol in q_filt:
        err.append(np.linalg.norm(np.array(q_sol) - np.array(q)))
    print(f'error is {err}')
    return err
    
def x_plan(t):
# t- time vector
# output : x – matrix (3,len(t)), the position of the tool’s origin in time t.
    x = np.outer(((x_b-x_a)/T),t)+x_a[:,None]
    return x            

def q_plan(t,elbows):
# input - time vector
# output : q – the joints parameters in time t.(6,len(t))
    x = x_plan(t)
    q =inverse_kin(x,elbows)
    return q



###### numerical input for inverse #####
elbows = [1,-1]
elbows_permutations = [[1,1],[-1,-1],[1,-1],[-1,1]] 


R = 4.
r = 2.
L = 3.5
x= [-3,-1, 45]


########################################

plt.close('all')        

# draw_inv(x)

# draw_inv_per_time_step(x)


# # x_from_inv = forward_kin_per_time_step(q)

# # draw forward kinematics
q = inverse_kin_per_time_step(x, [1,-1])
# d_calc = draw_forward_kin_per_time_step(q)




# # calculate error

error = inverse_forward_error(q)

# traj plan numerical values
traj_elbow = [1,-1]
x_a = np.array([-3, -2 , 45])
x_b = np.array([-2 , 0 , 0])
theta_start = 0
theta_fin = -pi/4
T=2
t= np.linspace(0,2,21)

x_p = x_plan(t) 
q_p = q_plan(t, traj_elbow) 

draw_inv_per_time_step(x_a)
draw_inv_per_time_step(x_b)
draw_tool_traj(x_p,q_p,[0,0.5,1,1.5,2])


# q= inverse_kin_per_time_step(x, [1,1])



