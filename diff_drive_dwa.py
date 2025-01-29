# from matplotlib import pyplot as plt
# from scipy import interpolate
# import random
# import math
# import numpy as np
#
# obstacles_on = 1;
#
# class parameters:
#     def __init__(self):
#         self.dt = 0.1; #integration step
#         self.goal = np.array([1.5,1.5]);
#         self.obstacles = np.array([
#         [1,1,0.4],
#         #[1.5,1,0.2],
#         #[1.8,1,0.2]
#         ]); #x,y,R
#         self.v = 0.2; #nominal speed
#         self.r0 = 0.1; #when the robot is at this distance from goal then stop
#         self.R = 0.1; #radius of the robot
#         self.r_buffer = self.R; #extra distance to start turning
#         self.v_max = 0.5; #max speed
#         self.omega_min = -2;
#         self.omega_max = 2;
#         self.n_omega = 10; #number of omega's to generate
#         self.prediction_horizon = 25; #simulate this much time ahea ahead
#         self.pause = 0.001
#         self.fps = 10
#
# def dwa(x0,y0,theta0,v,parms):
#
#     n_omega = parms.n_omega;
#     prediction_horizon = parms.prediction_horizon;
#     omega_min = parms.omega_min;
#     omega_max = parms.omega_max;
#     x_goal = parms.goal[0]
#     y_goal = parms.goal[1]
#     h = parms.dt
#
#     #1) generate n different omegas
#     omega_all = np.linspace(omega_min,omega_max,n_omega);
#
#     #2) generate n trajectories for various omega
#     #3) generate n cost function for various trajectories
#     cost_all = np.zeros(n_omega);
#     for i in range(0,n_omega):
#         z0 = [x0, y0, theta0]
#         z = np.array([z0]);
#         for j in range(0,prediction_horizon):
#             z0 = euler_integration([0, h],z0,[v,omega_all[i]],parms)
#             z = np.vstack([z, z0])
#
#         #cost to goal
#         for j in range(0,prediction_horizon+1):
#             x = z[j,0]; y = z[j,1]
#             cost_goal = np.sqrt((x_goal - x)**2 + (y_goal - y)**2)
#             cost_all[i] += cost_goal;
#
#         #cost of obstacle
#         if (obstacles_on==1):
#             for j in range(0,prediction_horizon+1):
#                 x = z[j,0];
#                 y = z[j,1];
#                 n_obstacles,n = parms.obstacles.shape
#                 for k in range(0,n_obstacles):
#                     x_obs = parms.obstacles[k,0]
#                     y_obs = parms.obstacles[k,1]
#                     r_obs = parms.obstacles[k,2] + parms.r_buffer
#                     dist_obstacle = np.sqrt(  (x_obs - x)**2 + (y_obs - y)**2 - r_obs**2 )
#                     #dist_obstacle = np.sqrt(  (x_obs - x)**2 + (y_obs - y)**2 )
#                     cost_all[i] += 0.5*(1/dist_obstacle);
#                     # if (dist_obstacle - r_obs < 0): #if the car get too close add a cost
#                     #     cost_all[i] += 10*dist_obstacle;
#
#
#     #4) choose omega with the lowest cost
#     index = 0;
#     min_cost = cost_all[index];
#     omega = omega_all[index]
#     for i in range(1,n_omega):
#         if (cost_all[i]<min_cost):
#             index = i;
#             min_cost = cost_all[i];
#             omega = omega_all[i];
#
#     return omega
#
# def animate(t,z,parms):
#
#     t_interp = np.arange(t[0],t[len(t)-1],1/parms.fps)
#     [m,n] = np.shape(z)
#     shape = (len(t_interp),n)
#     z_interp = np.zeros(shape)
#
#     for i in range(0,n):
#         f = interpolate.interp1d(t, z[:,i])
#         z_interp[:,i] = f(t_interp)
#
#     R = parms.R;
#     phi = np.arange(0,2*np.pi,0.1)
#
#     x_goal = parms.goal[0]
#     y_goal = parms.goal[1]
#
#     n_obstacles,n = parms.obstacles.shape
#
#     for i in range(0,len(t_interp)):
#         x = z_interp[i,0]
#         y = z_interp[i,1]
#         theta = z_interp[i,2]
#
#         x_robot = x + R*np.cos(phi)
#         y_robot = y + R*np.sin(phi)
#
#         x2 = x + R*np.cos(theta)
#         y2 = y + R*np.sin(theta)
#
#         line, = plt.plot([x, x2],[y, y2],color="black")
#         robot,  = plt.plot(x_robot,y_robot,color='black')
#         shape, = plt.plot(z_interp[0:i,0],z_interp[0:i,1],color='blue');
#         goal,  = plt.plot(x_goal, y_goal, 'ko', markersize=10, markerfacecolor='black')
#
#         if (obstacles_on==1):
#             for i in range(0,n_obstacles):
#                 x_center_obs = parms.obstacles[i,0];
#                 y_center_obs = parms.obstacles[i,1];
#                 r_obs = parms.obstacles[i,2];
#
#                 x_obs = x_center_obs + r_obs*np.cos(phi)
#                 y_obs = y_center_obs + r_obs*np.sin(phi)
#
#                 plt.plot(x_obs,y_obs,color='red')
#
#         plt.xlim(-2,2)
#         plt.ylim(-2,2)
#         plt.gca().set_aspect('equal')
#         plt.pause(parms.pause)
#         line.remove()
#         robot.remove()
#         shape.remove()
#
# plt.close()
#
#
# def euler_integration(tspan,z0,u,parms):
#     v = u[0]
#
#     v_max = parms.v_max;
#     if (v>=v_max):
#         v = v_max;
#
#     omega = u[1]
#     h = tspan[1]-tspan[0]
#
#     x0 = z0[0]
#     y0 = z0[1]
#     theta0 = z0[2]
#
#     xdot_c = v*math.cos(theta0)
#     ydot_c = v*math.sin(theta0)
#     thetadot = omega
#
#     x1 = x0 + xdot_c*h
#     y1 = y0 + ydot_c*h
#     theta1 = theta0 + thetadot*h
#
#     z1 = [x1, y1, theta1]
#     return z1
#
# parms = parameters()
# direction = random.choice([-1, 1]) #randomly generate a direction to turn in the beginnig
#
# #initial condition, [x0, y0, theta0]
# z0 = [0, 0, 0]
#
# N = 500 #end time
# h = parms.dt
# x_goal = parms.goal[0]
# y_goal = parms.goal[1]
# r0 = parms.r0
# # %%%%% the controls are v = speed and omega = direction
# # %%%%%% v = 0.5r(phidot_r + phidot_l)
# # %%%%%% omega = 0.5 (r/b)*(phitdot_r - phidot_l)
# z = np.array([z0]);
# t = np.array([0]);
#
# for i in range(0,N):
#
#     x = z0[0]; y = z0[1]; theta = z0[2];
#     theta_des = np.arctan2( y_goal - y,x_goal - x);
#
#     v = parms.v;
#     omega = dwa(x,y,theta,v,parms);
#
#     r = np.sqrt((x_goal - x)**2 + (y_goal - y)**2)
#     if (r < r0):
#         v = 0 #come to stop
#         break; #exit for loop
#
#     z0 = euler_integration([0, h],z0,[v,omega],parms)
#     z = np.vstack([z, z0])
#     t = np.append(t,t[-1]+h)
#
# animate(t,z,parms)

###################################################################
import numpy as np
import math
import matplotlib.pyplot as plt
from scipy import interpolate
from skopt import gp_minimize  # Bayesian Optimization

# Enable obstacle avoidance
obstacles_on = 1


# ------------------- Parameters Class -------------------
class Parameters:
    def __init__(self):
        self.dt = 0.2  # Smaller time step for better control
        self.goal = np.array([1.6, 1.6])  # Goal position (x, y)

        # Define multiple obstacles in the environment
        self.obstacles = np.array([
            [1, 1, 0.1],
            [1.3, 1.2, 0.1],
            [1.8, 1, 0.1],
            [1.4, 1.6, 0.1]
        ])  # (x, y, radius)

        self.v = 0.3  # Slightly faster speed
        self.r0 = 0.1  # Stop threshold when reaching the goal
        self.R = 0.1  # Robot radius
        self.r_buffer = 0.1  # Safety buffer around obstacles
        self.v_max = 0.5  # Max speed
        self.omega_min = -2  # Allow wider turns
        self.omega_max = 2  # Allow wider turns
        self.n_omega = 15  # Reduce number of omega samples for efficiency
        self.prediction_horizon = 20  # Reduce prediction steps
        self.pause = 0.005  # Faster animation
        self.fps = 10  # Reduce FPS for efficiency
        self.w_obs = 0.05  # Adjusted obstacle avoidance weight


# ------------------- Dynamic Window Approach (DWA) -------------------
def dwa(x0, y0, theta0, v, parms):
    n_omega = parms.n_omega
    omega_all = np.linspace(parms.omega_min, parms.omega_max, n_omega)
    cost_all = np.zeros(n_omega)

    for i in range(n_omega):
        z0 = [x0, y0, theta0]
        z = np.array([z0])

        # Simulate trajectory for given omega
        for j in range(parms.prediction_horizon):
            z0 = euler_integration([0, parms.dt], z0, [v, omega_all[i]], parms)
            z = np.vstack([z, z0])

        # Compute cost: Goal cost + Obstacle cost
        for j in range(parms.prediction_horizon + 1):
            x, y = z[j, 0], z[j, 1]

            # Cost to goal (weighted more strongly to prioritize goal-seeking)
            cost_goal = 2 * np.sqrt((parms.goal[0] - x) ** 2 + (parms.goal[1] - y) ** 2)
            cost_all[i] += cost_goal

            # Cost of obstacles (avoid collisions)
            if obstacles_on:
                for obs in parms.obstacles:
                    x_obs, y_obs, r_obs = obs
                    dist_obstacle = np.sqrt((x_obs - x) ** 2 + (y_obs - y) ** 2) - (r_obs + parms.r_buffer)
                    if dist_obstacle > 0:
                        cost_all[i] += parms.w_obs * (1 / dist_obstacle)
                    else:  # High penalty for collision
                        cost_all[i] += 100

    # Select the omega with the lowest cost
    index = np.argmin(cost_all)
    return omega_all[index]


# ------------------- Euler Integration -------------------
def euler_integration(tspan, z0, u, parms):
    v, omega = u
    v = min(v, parms.v_max)

    x0, y0, theta0 = z0
    h = tspan[1] - tspan[0]

    x1 = x0 + v * math.cos(theta0) * h
    y1 = y0 + v * math.sin(theta0) * h
    theta1 = theta0 + omega * h

    return [x1, y1, theta1]


# ------------------- Animation Function -------------------
def animate(t, z, parms):
    t_interp = np.linspace(t[0], t[-1], num=len(t) * parms.fps)
    z_interp = np.array([np.interp(t_interp, t, z[:, i]) for i in range(z.shape[1])]).T

    R = parms.R
    phi = np.linspace(0, 2 * np.pi, 100)

    for i in range(len(t_interp)):
        x, y, theta = z_interp[i]

        plt.clf()
        plt.xlim(-0.5, 2)
        plt.ylim(-0.5, 2)
        plt.gca().set_aspect('equal')

        # Draw Robot
        plt.plot(x + R * np.cos(phi), y + R * np.sin(phi), 'g', linewidth=2)  # Green robot
        plt.plot([x, x + R * np.cos(theta)], [y, y + R * np.sin(theta)], 'k', linewidth=2)  # Orientation

        # Draw Goal
        plt.plot(parms.goal[0], parms.goal[1], 'ko', markersize=10, markerfacecolor='black')

        # Draw Obstacles
        for obs in parms.obstacles:
            x_obs, y_obs, r_obs = obs
            plt.plot(x_obs + r_obs * np.cos(phi), y_obs + r_obs * np.sin(phi), 'r', linewidth=2)

        # Draw Path
        plt.plot(z_interp[:i, 0], z_interp[:i, 1], 'b', linewidth=1.5)

        plt.pause(parms.pause)

    plt.show()


# ------------------- Running the Simulation -------------------
def run_simulation(parms):
    z0 = [0, 0, 0]  # Initial state
    z = np.array([z0])
    t = np.array([0])

    for _ in range(400):  # Reduced iterations for better performance
        x, y, theta = z0
        if np.linalg.norm(parms.goal - np.array([x, y])) < parms.r0:
            break  # Stop if goal is reached

        omega = dwa(x, y, theta, parms.v, parms)
        z0 = euler_integration([0, parms.dt], z0, [parms.v, omega], parms)
        z = np.vstack([z, z0])
        t = np.append(t, t[-1] + parms.dt)

    animate(t, z, parms)


# ------------------- Run Everything -------------------
if __name__ == "__main__":
    parms = Parameters()
    run_simulation(parms)
