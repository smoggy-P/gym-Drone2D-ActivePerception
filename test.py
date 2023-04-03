
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
target = np.array([50, 51, 0, 0])
A = np.array([[1, 0, dt, 0],
            [0, 1, 0, dt],
            [0, 0, 1, 0],
            [0, 0, 0, 1]])
B = np.array([[0.5*dt**2, 0],
            [0, 0.5*dt**2],
            [dt, 0],
            [0, dt]])

nx = 4
nu = 2

# MPC setup
N = 10
Q = np.eye(nx)
R = np.eye(nu)
lam = 1000


umin = -40 * np.ones([nu])
umax = 40 * np.ones([nu])
xmin = np.array([-200, -200, -20, -20])
xmax = np.array([200, 200, 20, 20])

# FORCESPRO multistage form
# assume variable ordering z[i] = [si, ui, x_i+1] for i=0...N-1
model = forcespro.nlp.SymbolicModel(N)
model.nvar = 7  # number of stage variables
model.neq = 4   # number of equality constraints
model.nh = 1    # number of nonlinear inequality constraints
model.npar = 4


model.objective = lambda z: casadi.horzcat(z[3]-target[0], z[4]-target[1], z[5]-target[2], z[6]-target[3]) @ Q @ casadi.vertcat(z[3]-target[0], z[4]-target[1], z[5]-target[2], z[6]-target[3]) + lam*z[0]

model.lb = np.concatenate(([0], umin, xmin), 0)
model.ub = np.concatenate(([+float("inf")], umax, xmax), 0)
model.eq = lambda z: casadi.vertcat(casadi.dot(A[0, :], casadi.vertcat(z[3], z[4], z[5], z[6])) + casadi.dot(B[0, :], casadi.vertcat(z[1], z[2])),
                                    casadi.dot(A[1, :], casadi.vertcat(z[3], z[4], z[5], z[6])) + casadi.dot(B[1, :], casadi.vertcat(z[1], z[2])),
                                    casadi.dot(A[2, :], casadi.vertcat(z[3], z[4], z[5], z[6])) + casadi.dot(B[2, :], casadi.vertcat(z[1], z[2])),
                                    casadi.dot(A[3, :], casadi.vertcat(z[3], z[4], z[5], z[6])) + casadi.dot(B[3, :], casadi.vertcat(z[1], z[2])))

for i in range(N):
    if i > 0:
        model.ineq[i] = lambda z, p: casadi.vertcat((z[3] - p[0] - p[2]*i*dt )**2 + (z[4] - p[1]- p[3]*i*dt)**2 + z[0])
        model.hu[i] = [+float("inf")]                 # upper bound for nonlinear constraints
        model.hl[i] = [25]  
    else:
        model.ineq[i] = lambda z, p: casadi.vertcat((z[3] - p[0] - p[2]*i*dt )**2 + (z[4] - p[1]- p[3]*i*dt)**2)
        model.hu[i] = [+float("inf")]                 # upper bound for nonlinear constraints
        model.hl[i] = [-float("inf")]
# model.ineq = lambda z: casadi.vertcat((z[2] - 20)**2 + (z[3] - 20)**2)

                     # lower bound for nonlinear constraints
model.E = np.concatenate([np.zeros((4, 3)), np.eye(4)], axis=1) 
  

model.xinitidx = range(3, nu + nx + 1)




# Generate FORCESPRO solver
# -------------------------

# set options
options = forcespro.CodeOptions()
options.printlevel = 0
options.overwrite = 1
options.nlp.bfgs_init = None
options.maxit = 20000

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

for k in range(kmax):
    problem["xinit"] = x[:, k]
    params = np.array([30-k/2,30-k/2,-5,-5])
    problem["all_parameters"] = np.tile(params, (model.N,))

    # call the solver
    solverout, exitflag, info = solver.solve(problem)
    if exitflag < 0:
        break

    # 太他吗傻逼了
    s[0, k] = solverout["x01"][0]
    u[0, k] = solverout["x01"][1]
    u[1, k] = solverout["x01"][2]

    solvetime.append(info.solvetime)
    iters.append(info.it)
    c = np.concatenate([s[:, k], u[:, k], x[:, k]])
    a = model.eq(c)
    b = a.full()
    x[:, k + 1] = b.reshape(nx,)


fig = plt.figure()
ax = plt.axes()
px = []
py = []
line, = plt.plot(px, py)


def init():
    px.clear()
    py.clear()
    line.set_data(px, py)    
    return line,

def update(step):
    px.append(x[0, step])
    py.append(x[1, step])
    line.set_data(px, py)


    circle = plt.Circle((30 - step/2, 30 - step/2), 5, color='lightgrey')
    ax.patches = []
    ax.add_patch(circle)

    
    lb = min(min(px),min(py)-10)
    ub = max(max(px),max(py)+10)
    ax.set_xlim(lb, ub)
    ax.set_ylim(lb, ub)
    return line, 


ani = animation.FuncAnimation(fig = fig, 
                              func = update,
                              init_func = init,
                              blit = False,
                              frames = x.shape[1], 
                              interval = 200)
plt.show()

