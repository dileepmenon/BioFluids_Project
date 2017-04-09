import numpy as np


def bead_vel(mob_list, total_force_list):
    """Computes the velocity of each bead.

    Args:
        mob_list          : List containing mobilites of each bead as list
        total_force_list  : List containing total force acting on each bead

    Returns:
        list_vel: List containing the velocity of each bead
    """
    list_vel = []

    for i, mu_i in enumerate(mob_list):
        s = np.mat([0.0, 0.0, 0.0)].T
        for j, mu_j in enumerate(mu_i):
            s += mu_j*total_force_list[j]
        list_vel.append(s)

    return list_vel
