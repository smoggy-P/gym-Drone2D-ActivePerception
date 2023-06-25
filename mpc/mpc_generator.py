import sys
sys.path.insert(0, '/home/cc/moji_ws/forces_pro_client/')  # On Windows, note the doubly-escaped backslashes
import forcespro
import numpy as np
import casadi
import matplotlib.pyplot as plt
from math import sqrt, radians
import matplotlib.animation as animation
# system
dt = 0.1
target = np.array([240, 605, 0, 0])
num_agent = 5

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
N = 25
Q = 0.1 * np.eye(nx)
lam = 1


umin = -40 * np.ones([nu])
umax = 40 * np.ones([nu])
xmin = np.array([6, 6, -float("inf"), -float("inf")])
xmax = np.array([480-6, 640-6, float("inf"), float("inf")])

# FORCESPRO multistage form
# assume variable ordering z[i] = [si, ui, x_i+1] for i=0...N-1
model = forcespro.nlp.SymbolicModel(N)
model.nvar = ns+nu+nx  # number of stage variables
model.neq = nx   # number of equality constraints
model.nh = num_agent + 1    # number of nonlinear inequality constraints
model.npar = 5*num_agent + 3  # number of parameters

model.lb = np.concatenate(([0]*ns, umin, xmin), 0)
model.ub = np.concatenate(([+float("inf")]*ns, umax, xmax), 0)

model.eq = lambda z: A @ casadi.reshape(z[ns+nu:],-1,1) + B @ casadi.reshape(z[ns:ns+nu],-1,1)

model.objective = lambda z, p: casadi.reshape(z[ns+nu:]-casadi.vertcat(p[-2:], casadi.SX.zeros(2)),1,-1) @ Q @ casadi.reshape(z[ns+nu:]-casadi.vertcat(p[-2:], casadi.SX.zeros(2)),-1,1) + lam*z[0]

def ineq_constraint(z, p):
    p_reshaped = (p[:-3].reshape((-1, num_agent))).T

    # Collision constraints
    constraints = []

    for i in range(num_agent):
        x_hat = z[ns+nu] - p_reshaped[i,0]
        y_hat = z[ns+nu+1] - p_reshaped[i,1]

        con = (casadi.cos((p_reshaped[i,4])) * x_hat + casadi.sin((p_reshaped[i,4])) * y_hat)**2 * (p_reshaped[i,3]+5) ** 2 + (casadi.cos((p_reshaped[i,4])) * y_hat - casadi.sin((p_reshaped[i,4])) * x_hat)**2 * (p_reshaped[i,2]+5) ** 2 - ((p_reshaped[i,2]+5) * (p_reshaped[i,3]+5))**2 + z[0]
        constraints.append(con)

    # Velocity Constraints
    constraints.append(z[ns+nu+2]**2 + z[ns+nu+3]**2 - p[-3]**2)

    return casadi.vertcat(*constraints)

model.ineq = ineq_constraint

model.hu = [+float("inf")]*num_agent + [0]
model.hl = [0]*num_agent + [-float("inf")]

model.E = np.concatenate([np.zeros((4, 3)), np.eye(4)], axis=1) 


model.xinitidx = range(3, nu + nx + 1)

# Generate FORCESPRO solver
# -------------------------

# set options
options = forcespro.CodeOptions('MPC_SOLVER')
options.printlevel = 0
options.overwrite = 1
options.nlp.bfgs_init = None
options.maxit = 5000
                            
options.optlevel    = 3
options.overwrite   = 1
options.cleanup     = 1
options.timing      = 1
options.parallel    = 1
options.threadSafeStorage = True
options.noVariableElimination = 0


# generate code
solver = model.generate_solver(options)