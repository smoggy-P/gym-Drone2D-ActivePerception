import numpy as np
import cvxpy as cp

# Define the prediction horizon and control horizon
N = 4
M = 5

# Define the state and control constraints
x_min = np.array([-10, -10, -10, -10])
x_max = np.array([10, 10, 10, 10])
u_min = np.array([-1, -1])
u_max = np.array([1, 1])

# Define the initial state
x0 = np.array([7, 7, 1, 0])

# Define the system dynamics
A = np.array([[1, 0, 1, 0],
              [0, 1, 0, 1],
              [0, 0, 1, 0],
              [0, 0, 0, 1]])
B = np.array([[0, 0],
              [0, 0],
              [1, 0],
              [0, 1]])

# Define the cost function matrices
Q = np.eye(4)
R = np.eye(2)

# Define the obstacle constraints
obstacle_x = np.array([5, 3])
obstacle_r = 5

# Define the optimization variables
x = cp.Variable((4, N+1))
u = cp.Variable((2, N))

# Define the constraints
constraints = []
for i in range(N):
    constraints += [x[:,i+1] == A@x[:,i] + B@u[:,i]]
    # constraints += [cp.norm(x[:2,i+1] - obstacle_x)**2 >= obstacle_r**2]
    constraints += [x_min <= x[:,i], x[:,i] <= x_max]
    constraints += [u_min <= u[:,i], u[:,i] <= u_max]
constraints += [x[:,0] == x0]

# Define the cost function
cost = 0
for i in range(N):
    cost += cp.quad_form(x[:,i], Q) + cp.quad_form(u[:,i], R)

# Form the optimization problem
prob = cp.Problem(cp.Minimize(cost), constraints)

# Solve the optimization problem
result = prob.solve()

# Extract the optimal control sequence
u_opt = u.value

# Simulate the system to get the state trajectory
x_sim = np.zeros((4, N+1))
x_sim[:,0] = x0
for i in range(N):
    x_sim[:,i+1] = A@x_sim[:,i] + B@u_opt[:,i]

print(x_sim)
# Plot the results
import matplotlib.pyplot as plt
plt.plot(x_sim[0,:], x_sim[1,:], 'o-')

plt.plot(obstacle_x[0], obstacle_x[1], 'ro')
circle = plt.Circle(obstacle_x, obstacle_r, color='r', fill=False)
plt.show()