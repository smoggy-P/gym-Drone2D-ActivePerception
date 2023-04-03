
import sys
sys.path.insert(0, '/home/smoggy/Downloads/forces_pro_client/')  # On Windows, note the doubly-escaped backslashes
import forcespro
import numpy as np
import casadi
import matplotlib.pyplot as plt
from math import sqrt

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


umin = -40 * np.ones([nu])
umax = 40 * np.ones([nu])
xmin = np.array([-200, -200, -40, -40])
xmax = np.array([200, 200, 40, 40])

# FORCESPRO multistage form
# assume variable ordering z[i] = [ui, x_i+1] for i=0...N-1
model = forcespro.nlp.SymbolicModel(N)
model.nvar = 6  # number of stage variables
model.neq = 4   # number of equality constraints
model.nh = 1    # number of nonlinear inequality constraints

model.objective = lambda z: casadi.horzcat(z[2]-target[0], z[3]-target[1], z[4]-target[2], z[5]-target[3]) @ Q @ casadi.vertcat(z[2]-target[0], z[3]-target[1], z[4]-target[2], z[5]-target[3])
model.lb = np.concatenate((umin, xmin), 0)
model.ub = np.concatenate((umax, xmax), 0)
model.eq = lambda z: casadi.vertcat(casadi.dot(A[0, :], casadi.vertcat(z[2], z[3], z[4], z[5])) + casadi.dot(B[0, :], casadi.vertcat(z[0], z[1])),
                                        casadi.dot(A[1, :], casadi.vertcat(z[2], z[3], z[4], z[5])) + casadi.dot(B[1, :], casadi.vertcat(z[0], z[1])),
                                        casadi.dot(A[2, :], casadi.vertcat(z[2], z[3], z[4], z[5])) + casadi.dot(B[2, :], casadi.vertcat(z[0], z[1])),
                                        casadi.dot(A[3, :], casadi.vertcat(z[2], z[3], z[4], z[5])) + casadi.dot(B[3, :], casadi.vertcat(z[0], z[1])))
model.ineq = lambda z: casadi.vertcat((z[2] - 20)**2 + (z[3] - 20)**2)
model.hu = [+float("inf")]                 # upper bound for nonlinear constraints
model.hl = [100]                       # lower bound for nonlinear constraints
model.E = np.concatenate([np.zeros((4, 2)), np.eye(4)], axis=1) 
  

model.xinitidx = range(2, nu + nx)




# Generate FORCESPRO solver
# -------------------------

# set options
options = forcespro.CodeOptions()
options.printlevel = 0
options.overwrite = 1
options.nlp.bfgs_init = None
options.forcenonconvex = 1

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

    # call the solver
    solverout, exitflag, info = solver.solve(problem)
    assert exitflag >= 0, "Some problem in solver"

    # 太他吗傻逼了
    u[0, k] = solverout["x01"][0]
    u[1, k] = solverout["x01"][1]

    solvetime.append(info.solvetime)
    iters.append(info.it)
    c = np.concatenate([u[:, k], x[:, k]])
    a = model.eq(c)
    b = a.full()
    x[:, k + 1] = b.reshape(nx,)


# Plot results
# ------------

fig = plt.gcf()
plt.plot(x[0, range(1, kmax + 1)], x[1, range(1, kmax + 1)])

plt.show()
