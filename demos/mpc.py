
import sys
sys.path.insert(0, '/home/smoggy/Downloads/forces_pro_client/')  # On Windows, note the doubly-escaped backslashes
import forcespro
import numpy as np
import casadi
import matplotlib.pyplot as plt
from math import sqrt
import matplotlib.animation as animation

# system
dt = 0.1
target = np.array([240, 605, 0, 0])
num_agent = 5

agent_list = np.array([[0, 50, 0, 10, -10],  #px, py, r, vx, vy
                       [0, 50, 15, 10, -10], 
                       [0, 50, 0, 10, -10],
                       [30, 30, 10, 0, 0],
                       [0, 0, 0, 0, 0]])



A = np.array([[1, 0, dt, 0],
            [0, 1, 0, dt],
            [0, 0, 1, 0],
            [0, 0, 0, 1]])
B = np.array([[0.5*(dt**2), 0],
            [0, 0.5*(dt**2)],
            [dt, 0],
            [0, dt]])

nx = 4
nu = 2
ns = 1

# MPC setup
N = 15
Q = np.eye(nx)
lam = 1000


umin = -40 * np.ones([nu])
umax = 40 * np.ones([nu])
xmin = np.array([0, 0, -10, -10])
xmax = np.array([480, 640, 10, 10])

# FORCESPRO multistage form
# assume variable ordering z[i] = [si, ui, x_i+1] for i=0...N-1
model = forcespro.nlp.SymbolicModel(N)
model.nvar = ns+nu+nx  # number of stage variables
model.neq = nx   # number of equality constraints
model.nh = num_agent    # number of nonlinear inequality constraints
model.npar = 3*num_agent  # number of parameters

model.lb = np.concatenate(([0]*ns, umin, xmin), 0)
model.ub = np.concatenate(([+float("inf")]*ns, umax, xmax), 0)

model.eq = lambda z: A @ casadi.reshape(z[ns+nu:],-1,1) + B @ casadi.reshape(z[ns:ns+nu],-1,1)

model.objective = lambda z: casadi.reshape(z[ns+nu:]-target,1,-1) @ Q @ casadi.reshape(z[ns+nu:]-target,-1,1) + lam*z[0]

def ineq_constraint(z, p):
    p_reshaped = (p.reshape((-1, num_agent))).T
    lis = [(z[ns+nu] - p_reshaped[j,0])**2 + (z[ns+nu+1] - p_reshaped[j,1])**2 - p_reshaped[j,2]**2 + z[0] for j in range(num_agent)]

    return casadi.vertcat(*lis)

model.ineq = ineq_constraint

model.hu = [+float("inf")]*num_agent
model.hl = [0]*num_agent

model.E = np.concatenate([np.zeros((4, 3)), np.eye(4)], axis=1) 


model.xinitidx = range(3, nu + nx + 1)

# Generate FORCESPRO solver
# -------------------------

# set options
options = forcespro.CodeOptions('MPC_SOLVER')
options.printlevel = 0
options.overwrite = 1
options.nlp.bfgs_init = None
options.maxit = 2000
options.printlevel  = 1
                            
options.optlevel    = 3
options.overwrite   = 1
options.cleanup     = 1
options.timing      = 1
options.parallel    = 1
options.threadSafeStorage = True
options.noVariableElimination = 1


# generate code
solver = model.generate_solver(options)

# Run simulation
# --------------

x1 = [0, 0, 0, 0]
kmax = 50
x = np.zeros((nx, kmax + 1))
x[:, 0] = x1
u = np.zeros((nu, kmax))
s = np.zeros((1, kmax))
problem = {}

solvetime = []
iters = []

fig = plt.figure(figsize=(8,8))
ax = plt.axes()
ax.set_xlim(-10, 60)
ax.set_ylim(-10, 60)
plt.grid()
plt.ion()

for k in range(kmax):

    ax.patches = []
    circle2 = plt.Circle((x[0,k],x[1,k]), 0.3, color='r')
    ax.add_patch(circle2)

    problem["xinit"] = x[:, k]

    all_params = np.array([[[agent_list[j,0]+(k+i)*dt*agent_list[j,3], agent_list[j,1]+(k+i)*dt*agent_list[j,4], agent_list[j,2]] for j in range(num_agent)] for i in range(N)])
    problem["all_parameters"] = all_params.flatten()
    # call the solver
    solverout, exitflag, info = solver.solve(problem)
    if exitflag < 0:
        print(exitflag)
        break

    s[0, k] = solverout["x01"][0]
    u[0, k] = solverout["x01"][1]
    u[1, k] = solverout["x01"][2]


    solvetime.append(info.solvetime)
    iters.append(info.it)
    c = np.concatenate([s[:, k], u[:, k], x[:, k]])
    a = model.eq(c)
    b = a.full()
    x[:, k + 1] = b.reshape(nx,)

    
    
    params = np.array([[agent_list[i,0]+k*dt*agent_list[i,3], agent_list[i,1]+k*dt*agent_list[i,4], agent_list[i,2], agent_list[i,3], agent_list[i,4]] for i in range(num_agent)])
    for param in params:
        circle1 = plt.Circle((param[0], param[1]), param[2], color='lightgrey')
        ax.add_patch(circle1)
    
    ax.scatter([i[3] for i in solverout.values()],
               [i[4] for i in solverout.values()], s=0.2)
    ax.set_xlim(-10, 60)
    ax.set_ylim(-10, 60)
    plt.pause(0.1)
    plt.cla()
    


fig = plt.figure(figsize=(8,8))
ax = plt.axes()
px = []
py = []
line, = plt.plot(px, py)
plt.grid()


def init():
    px.clear()
    py.clear()
    line.set_data(px, py)    
    return line,

def update(step):
    px.append(x[0, step])
    py.append(x[1, step])
    line.set_data(px, py)

    ax.patches = []
    for agent in agent_list:
        circle = plt.Circle((agent[0] + step*dt*agent[3], agent[1] + step*dt*agent[4]), agent[2], color='lightgrey')
        ax.add_patch(circle)

    
    lb = min(min(px),min(py)-10,-10)
    ub = max(max(px),max(py)+10, 60)
    ax.set_xlim(lb, ub)
    ax.set_ylim(lb, ub)
    return line, 

ani = animation.FuncAnimation(fig = fig, 
                              func = update,
                              init_func = init,
                              blit = False,
                              frames = x.shape[1], 
                              interval = 100)
plt.show()
