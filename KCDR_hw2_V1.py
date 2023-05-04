import numpy as np
import matplotlib.pyplot as plt
from matplotlib import style
from math import pi,atan2,atan,sqrt,cos,sin
import itertools
import sympy as sp
from sympy.physics.mechanics import dynamicsymbols
from sympy import diff, symbols

def inverse_kin(x,elbows):
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

def forward_kin(q):
    #q[0] - theta 1 , q[1] - theta2 , q[2] - d3
    q_0_rad = q[0]*pi/180.
    q_1_rad = q[1]*pi/180
    
    theta1_sym, theta2_sym, d3_sym, phi_sym ,x_sym ,y_sym, L_sym, R_sym, r_sym ,t_sym = symbols ('theta1_sym, theta2_sym, d3_sym, phi_sym ,x_sym ,y_sym L_sym, R_sym, r_sym, t_sym')

    theta1_eq = (x_sym+r_sym*(sp.cos(phi_sym)*sp.cos(sp.pi/3)-sp.sin(phi_sym)*sp.sin(sp.pi/3))-R_sym*sp.cos(theta1_sym))**2 + (y_sym+r_sym*(sp.sin(phi_sym)*sp.cos(sp.pi/3) + sp.cos(phi_sym)*sp.sin(sp.pi/3))-R_sym*sp.sin(theta1_sym))**2 - L_sym**2  
    theta2_eq = (x_sym-R_sym*sp.cos(theta2_sym))**2 + (y_sym-R_sym*sp.sin(theta2_sym))**2 - L_sym**2
    d3_eq = (x_sym+r_sym*sp.cos(phi_sym))**2+(y_sym+r_sym*sp.sin(phi_sym)+R_sym)**2-d3_sym**2
    
    a = (theta1_eq-d3_eq)
    b = (theta2_eq-d3_eq)

    ans = sp.solve([a,b],x_sym,y_sym, dict=True)
    x_phi = ans[0][x_sym]
    y_phi = ans[0][y_sym]

    d3_eq_phi = d3_eq.subs({x_sym:x_phi, y_sym:y_phi})
    d3_eq_half_tangent = d3_eq_phi.subs({sp.sin(phi_sym):2*t_sym/(1+t_sym**2),sp.cos(phi_sym):((1-t_sym**2)/(1+t_sym**2))})
    t_num_eq = d3_eq_half_tangent.subs({theta1_sym:q_0_rad,theta2_sym:q_1_rad,d3_sym:q[2],L_sym:3.5,R_sym:4,r_sym:2})
    t_num_eq = t_num_eq.simplify()
    sol_t = sp.nonlinsolve(t_num_eq,t_sym)
    
    
    print(sol_t)
    return 0



def draw_inv(x):
    # draws all possibilites of invers kin for given x
    elbows_permutaions = [[1,1],[-1,-1],[1,-1],[-1,1]] 
    q_poss = []
    for count,elbow_poss in enumerate(elbows_permutaions):
        q_poss.append(inverse_kin(x,elbow_poss))
           
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
                       [x3, y3]],facecolor='gray',label='_nolegend_')
    axs[0, 0].plot(x_circ,y_circ, color = 'black',label='_nolegend_', linewidth = 3)
    axs[0, 0].add_patch(tool2)    
    axs[0, 0].plot([x1,R*cos(q_poss[0][0])],[y1,R*sin(q_poss[0][0])], color = 'blue',marker = 'o', label = 'theta1')
    axs[0, 0].plot([x2,R*cos(q_poss[0][1])],[y2,R*sin(q_poss[0][1])], color = 'green',marker = 'o', label = 'theta2')
    axs[0, 0].plot([x3,R*cos(q_poss[0][2])],[y3,R*sin(q_poss[0][2])], color = 'red', marker = 'o',label = 'd3')
    axs[0, 0].set(xlabel='x [m]', ylabel='y [m]')
    axs[0, 0].legend(['theta1','theta2','d3'])
    axs[0, 0].set_title('Config 1')

   
    tool3= plt.Polygon([[x1, y1],
                        [x2, y2],
                        [x3, y3]],facecolor='gray',label='_nolegend_')
    axs[0, 1].plot(x_circ,y_circ, color = 'black',label='_nolegend_', linewidth = 3) 
    axs[0, 1].add_patch(tool3)    
    axs[0, 1].plot([x1,R*cos(q_poss[1][0])],[y1,R*sin(q_poss[1][0])], color = 'blue',marker = 'o', label = 'theta1')
    axs[0, 1].plot([x2,R*cos(q_poss[1][1])],[y2,R*sin(q_poss[1][1])], color = 'green',marker = 'o', label = 'theta2')
    axs[0, 1].plot([x3,R*cos(q_poss[1][2])],[y3,R*sin(q_poss[1][2])], color = 'red', marker = 'o',label = 'd3')
    axs[0, 1].set(xlabel='x [m]', ylabel='y [m]')
    axs[0, 1].legend(['theta1','theta2','d3'])
    axs[0, 1].set_title('Config 2')



    tool4= plt.Polygon([[x1, y1],
                    [x2, y2],
                    [x3, y3]],facecolor='gray',label='_nolegend_')    
    axs[1, 0].plot(x_circ,y_circ, color = 'black',label='_nolegend_', linewidth = 3)    
    axs[1, 0].add_patch(tool4)    
    axs[1, 0].plot([x1,R*cos(q_poss[2][0])],[y1,R*sin(q_poss[2][0])], color = 'blue',marker = 'o', label = 'theta1')
    axs[1, 0].plot([x2,R*cos(q_poss[2][1])],[y2,R*sin(q_poss[2][1])], color = 'green',marker = 'o', label = 'theta2')
    axs[1, 0].plot([x3,R*cos(q_poss[2][2])],[y3,R*sin(q_poss[2][2])], color = 'red', marker = 'o',label = 'd3')
    axs[1, 0].set(xlabel='x [m]', ylabel='y [m]')
    axs[1, 0].legend(['theta1','theta2','d3'])
    axs[1, 0].set_title('Config 3')



    tool5= plt.Polygon([[x1, y1],
                    [x2, y2],
                    [x3, y3]],facecolor='gray',label='_nolegend_')
    axs[1, 1].plot(x_circ,y_circ, color = 'black',label='_nolegend_', linewidth = 3)
    axs[1, 1].add_patch(tool5)    
    axs[1, 1].plot([x1,R*cos(q_poss[3][0])],[y1,R*sin(q_poss[3][0])], color = 'blue',marker = 'o', label = 'theta1')
    axs[1, 1].plot([x2,R*cos(q_poss[3][1])],[y2,R*sin(q_poss[3][1])], color = 'green',marker = 'o', label = 'theta2')
    axs[1, 1].plot([x3,R*cos(q_poss[3][2])],[y3,R*sin(q_poss[3][2])], color = 'red', marker = 'o',label = 'd3')
    axs[1, 1].set(xlabel='x [m]', ylabel='y [m]')
    axs[1, 1].legend(['theta1','theta2','d3'])
    axs[1, 1].set_title('Config 4')




    
    fig2.tight_layout()


    return



###### numerical input for inverse #####
elbows = [1,1]
R = 4.
r = 2.
L = 3.5
x= [-3,-1, 45]


########################################

plt.close('all')        

draw_inv(x)

# draw forward kinematics

q = inverse_kin(x, [1,1])
q_deg = [angle*180/pi for angle in q if angle in q[:-1]]
q = q_deg + q[2]
# x_calc = forward_kin(q)




