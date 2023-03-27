import numpy as np
class KalmanFilter:

    def __init__(self, mu, Sigma):
        
        # check that initial state makes sense
        Dx = mu.shape[0]
        assert mu.shape == (Dx, 1)
        assert Sigma.shape == (Dx, Dx)

        self.mu_preds = []
        self.Sigma_preds = []
        self.mu_upds = []
        self.Sigma_upds = []

        self.ts = []

        self.mu_preds.append(mu)
        self.Sigma_preds.append(Sigma)

        self.mu_upds.append(mu)
        self.Sigma_upds.append(Sigma)
        self.ts.append(0.) # this is time t = 0
            
        # the dimensionality of the state vector
        self.Dx = Dx
    
        noise_var_x_pos = 0.1 # variance of spatial process noise
        noise_var_x_vel = 0.1 # variance of velocity process noise
        noise_var_z = 0.1 # variance of measurement noise for z_x and z_y

        self.F = np.array([[1,0,0.1,0  ],
                           [0,1,0  ,0.1],
                           [0,0,1  ,0  ],
                           [0,0,0  ,1  ]], dtype=np.float64) 
        self.H = np.array([[1,0,0,0],
                           [0,1,0,0]], dtype=np.float64) 
        self.Sigma_x = np.array([[noise_var_x_pos,0,0,0],
                                 [0,noise_var_x_pos,0,0],
                                 [0,0,noise_var_x_vel,0],
                                 [0,0,0,noise_var_x_vel]], dtype=np.float64)  
        self.Sigma_z = np.array([[noise_var_z,0],
                                 [0,noise_var_z]], dtype=np.float64)  
    
    def predict(self):
        # get the last (i.e. updated) state from the previous timestep
        mu_prev = self.mu_upds[-1]
        Sigma_prev = self.Sigma_upds[-1]
        t = self.ts[-1]
        mu = self.F.dot(mu_prev)
        Sigma = self.F.dot(Sigma_prev).dot(self.F.T) + self.Sigma_x

        self.mu_preds.append(mu)
        self.Sigma_preds.append(Sigma)

        self.mu_upds.append(mu)
        self.Sigma_upds.append(Sigma)
        self.ts.append(t + 1)
    
    def update(self, z):
        # get the latest predicted state, which should be the current time step
        self.predict()
        mu = self.mu_upds[-1]
        Sigma = self.Sigma_upds[-1]
        assert len(mu.shape) == 2
        assert mu.shape[1] == 1
        z = z.reshape(-1,1)

        e = z - self.H.dot(mu)
        print(e)
        S = self.Sigma_z + self.H.dot(Sigma).dot(self.H.T)
        K = Sigma.dot(self.H.T).dot(np.linalg.inv(S))
        
        mu_upd = mu + K.dot(e)
        I = np.eye(4)
        Sigma_upd = (I - K.dot(self.H)).dot(Sigma)
        assert mu_upd.shape == mu.shape
        assert Sigma_upd.shape == Sigma.shape

        self.mu_upds[-1] = mu_upd
        self.Sigma_upds[-1] = Sigma_upd

S_init1 = np.diag([1, 1, .1, .1])
tracker = KalmanFilter(mu=np.array([[2.],
                                    [0.],
                                    [0.],
                                    [0.]]), Sigma=S_init1)

import matplotlib.pyplot as plt
T = np.arange(0, 10, 0.1)
dt = 0.1
v = 1
x0 = 0
y0 = 0
sigma = 0
x = x0 + v * T + sigma * np.random.randn(len(T))
y = y0 + v * T + sigma * np.random.randn(len(T))
measured_x = []
measured_x.append((tracker.mu_upds[-1]).flatten())
for i, t in enumerate(T):
    tracker.update(z=np.array([[x[i]],[y[i]]]))
    measured_x.append((tracker.mu_upds[-1]).flatten())


# Visualize measurements
measured_x = np.array(measured_x)
plt.plot(T, measured_x[:-1, 0], label='Filtered result')
plt.plot(T, x, label='GT')
plt.xlabel('T')
plt.ylabel('x')
plt.legend()
plt.show()