import numpy as np
from filterpy.kalman import KalmanFilter
from filterpy.common import Q_discrete_white_noise

dt = 0.1
my_filter = KalmanFilter(dim_x=4, dim_z=2)

my_filter.x = np.array([[0.],
                        [0.],
                        [0.],
                        [0.]])       # initial state 

my_filter.F = np.array([[1.,0.,dt,0.],
                        [0.,1.,0.,dt],
                        [0.,0.,1.,0.],
                        [0.,0.,0.,1.]])    # state transition matrix

my_filter.H = np.array([[1.,0.,0.,0.],
                        [0.,1.,0.,0.]])    # Measurement function
my_filter.P *= 10.                 # covariance matrix
my_filter.R = 5                      # state uncertainty
my_filter.Q = Q_discrete_white_noise(dim=4, dt=0.1, var=0.1) # process uncertainty

import matplotlib.pyplot as plt
T = np.arange(0, 10, dt)
dt = 0.1
v = 1
x0 = 0
y0 = 0
sigma = 0
x = x0 + v * T + sigma * np.random.randn(len(T))
y = y0 + v * T + sigma * np.random.randn(len(T))
measured_x = []
measured_x.append(my_filter.x.flatten())

for i, t in enumerate(T):
    my_filter.predict()
    my_filter.update(z=np.array([[x[i]],[y[i]]]))
    measured_x.append(my_filter.x.flatten())


# Visualize measurements
measured_x = np.array(measured_x)
plt.plot(T, measured_x[:, 2], label='Filtered result')
plt.xlabel('T')
plt.ylabel('vel')
plt.legend()
plt.show()