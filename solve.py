import numpy as np

from mobility_list import bead_moblity

from forces import total_force

from velocity import bead_vel

sin = np.sin
cos = np.cos

def euler(x_old, v, dt):
    """Integrates the velocity using euler method.

        Args:
            x_old : Position vector of beads at time t
            v     : Velocity of beads at time t
            dt    : Timestep

        Returns:
            x_new : Updated posiion vector of beads at time t+dt
    """
    x_new = x_old[:]
    for i, x in enumerate(x_old):
        x_new[i] = x + v[i]*dt

    return x_new[:]



def solve(n, dt, t_f):
    """Solves for the circular motion of artificial beads at each time step.

    Args:
        n          : Number of beads
        dt         : Timestep for each iteration
        t_f        : Final time for simulation

    Returns:
        list_r_dt  : List of position vectors of each bead for every timestep
                     till t_f
    """
    # Bead_radius in m
    bead_r = 2.2e-6

    # Water viscosity in Pa s
    eta = 0.001

    # Density of bead in kg/m3
    rho_bead = 1600

    # Density of water in kg/m3
    rho_water = 1000

    # Pumping direction in radians
    alpha = np.radians(30)

    # The angle from vertical till cone axis
    tilt_angle = np.radians(30)

    # Semi-cone angle formed due to rotaion of beads
    semi_cone_angle = np.radians(30)

    # Angular velocity in rad/s
    omega = 2*np.pi*0.5

    # Magnetic susceptibility
    mag_sus = 1.63

    # Magnetic field amplitude
    B = 1.8e-3

    # Initial position of beads
    r_list = []
    #theta = tilt_angle + semi_cone_angle
    #right_angle = np.radians(90)

    #z_k = 2.0*bead_r*sin(right_angle-theta)
    #x_i = 2.0*bead_r*cos(right_angle-theta)

    for i in range(n):
        #r_list.append(np.mat([i*x_i, 0.0 , bead_r+i*z_k]).T)
        r_list.append(np.mat([0.0 , 0.0 , bead_r+2*i*bead_r]).T)

    list_r_dt = []
    list_r_dt.append(r_list)

    # Time iteration
    for t in np.arange(0.0, t_f, dt):
        mob_list = bead_moblity(bead_r, eta, r_list[:])
        list_total_F = total_force(bead_r, eta, rho_bead, rho_water, B, alpha,
                                   tilt_angle, semi_cone_angle, omega, t,
                                   mag_sus, r_list[:], mob_list
                                  )
        v = bead_vel(mob_list, list_total_F)
        r_list = euler(r_list[:], v, dt)
        list_r_dt.append(r_list)

    return list_r_dt[:]
