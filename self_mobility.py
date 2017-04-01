import numpy as np


def self_mob(a, eta, z):
    """Calculates and returns the self mobility matrix for a bead

        Args:
            a   : bead radius in micrometer
            eta : water viscosity in kg/m3
            z   : z-cordinate of bead in m
    """
    const = 1.0/(6*np.pi*eta*a)

    mat = np.matrix(np.zeros((3, 3)))

    a = 1.0 - (9.0/8.0)*(a/z) + (1.0/2.0)*(a/z)**3 - (1.0/8.0)*(a/z)**5
    b = 1.0 - (9.0/16.0)*(a/z) + (1.0/8.0)*(a/z)**3 - (1.0/16.0)*(a/z)**5

    mat[0, 0] = b
    mat[1, 1] = b
    mat[2, 2] = a

    return const*mat







