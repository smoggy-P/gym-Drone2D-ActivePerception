import matplotlib.pyplot as plt
import numpy as np
from numpy.linalg import inv
from math import cos, sin, sqrt
from matplotlib.patches import Circle



a = 2
b = 5
x1 = 10
y1 = 0
x0 = 2
y0 = 3
theta = 3.14/4
A = inv(np.array([
    [cos(theta), -sin(theta)],
    [sin(theta), cos(theta)]
]))

import cvxpy as cp

x = cp.Variable((2))
constraint = [cp.quad_form(A@(x-np.array([x0, y0])), np.array([[1/a**2,0],[0,1/b**2]])) <= 1]


cost = cp.norm(x - np.array([x1, y1]))
prob = cp.Problem(cp.Minimize(cost), constraint)

# Solve the optimization problem
result = prob.solve()
x2, y2 = x.value[0], x.value[1]
x3, y3 = 2*x0 - x2, 2*y0-y2
k = -(x1-x2)/(y1-y2)
r = abs(k*(x3-x2)+y2-y3)/sqrt(k**2+1)/2
dis = sqrt((x2-x1)**2+(y2-y1)**2)

x4 = (r/dis)*(x2-x1)+x2
y4 = (r/dis)*(y2-y1)+y2

plt.gca().add_artist(Circle((x4, y4), r))



t = np.linspace(0, 2 * np.pi, 100)
x = x0 + a * np.cos(t) * np.cos(theta) - b * np.sin(t) * np.sin(theta)
y = y0 + a * np.cos(t) * np.sin(theta) + b * np.sin(t) * np.cos(theta)

plt.plot(x, y, 'b-')
plt.axis('equal')
# plt.xlim(-a, a)
# plt.ylim(-b, b)
# ax.plot(x, y)
# ax.plot(x, y_normal, label="Normal Line")
plt.scatter(x1, y1, color="red", label="Point Outside Ellipse")
plt.scatter(x2, y2, color="blue", label="Point on Ellipse")
plt.legend()
# plt.set_aspect("equal", "datalim")
plt.show()
