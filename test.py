import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Circle
import matplotlib.animation as animation

# Define the Agent and VelocityObstacle classes as before

class Agent:
    def __init__(self, position, velocity, radius):
        self.position = np.array(position, dtype=float)
        self.velocity = np.array(velocity, dtype=float)
        self.radius = radius

class VelocityObstacle:
    def __init__(self, primary, secondary, tau=5.0, eps=0.01):
        self.primary = primary
        self.secondary = secondary
        self.tau = tau
        self.eps = eps

    def compute_new_velocity(self):
        rel_position = self.secondary.position - self.primary.position
        rel_velocity = self.primary.velocity - self.secondary.velocity

        dist = np.linalg.norm(rel_position)
        u_rel_position = rel_position / dist

        cos_theta = np.dot(rel_velocity, u_rel_position) / np.linalg.norm(rel_velocity)

        if dist < self.primary.radius + self.secondary.radius:  # already colliding
            return -self.primary.velocity

        # check if relative velocity is outside VO
        if cos_theta < 0 and dist / np.linalg.norm(rel_velocity) > self.tau:
            return self.primary.velocity

        else:
            leg_dist = self.secondary.radius + self.eps + self.primary.radius * self.tau / dist
            left_leg_direction = rel_position - np.sqrt(leg_dist**2 - self.secondary.radius**2) * np.array([-u_rel_position[1], u_rel_position[0]])
            right_leg_direction = rel_position + np.sqrt(leg_dist**2 - self.secondary.radius**2) * np.array([-u_rel_position[1], u_rel_position[0]])
            if np.linalg.norm(right_leg_direction - rel_velocity) < np.linalg.norm(left_leg_direction - rel_velocity):
                new_velocity = self.primary.velocity + right_leg_direction - rel_velocity
            else:
                new_velocity = self.primary.velocity + left_leg_direction - rel_velocity
            return new_velocity

# Create a simulation with two agents moving towards each other
primary = Agent([0, 0], [1, 0], 1)
secondary = Agent([10, 0], [-1, 0], 1)

# Initialize plot
fig, ax = plt.subplots()
ax.set_xlim(-5, 15)
ax.set_ylim(-10, 10)

# Plot agents
circle1 = Circle(primary.position, primary.radius, fill=False)
circle2 = Circle(secondary.position, secondary.radius, fill=False)
ax.add_patch(circle1)
ax.add_patch(circle2)

# Plot trajectories
trajectory1, = ax.plot([], [], 'r-')
trajectory2, = ax.plot([], [], 'b-')

# Function to initialize the animation
def init():
    trajectory1.set_data([], [])
    trajectory2.set_data([], [])
    return trajectory1, trajectory2,

# Function to update the animation each frame
def update(frame):
    # Calculate the new velocities using the VO
    vo = VelocityObstacle(primary, secondary)
    primary.velocity = vo.compute_new_velocity()

    vo = VelocityObstacle(secondary, primary)
    secondary.velocity = vo.compute_new_velocity()

    # Update the positions of the agents
    primary.position += primary.velocity * 0.1
    secondary.position += secondary.velocity * 0.1

    # Update the agent circles and trajectories
    circle1.center = primary.position
    circle2.center = secondary.position
    trajectory1.set_xdata(np.append(trajectory1.get_xdata(), primary.position[0]))
    trajectory1.set_ydata(np.append(trajectory1.get_ydata(), primary.position[1]))
    trajectory2.set_xdata(np.append(trajectory2.get_xdata(), secondary.position[0]))
    trajectory2.set_ydata(np.append(trajectory2.get_ydata(), secondary.position[1]))

    return trajectory1, trajectory2,

# Run the animation
ani = animation.FuncAnimation(fig, update, frames=np.arange(0, 200), init_func=init, blit=True)

# Display the animation
plt.show()
