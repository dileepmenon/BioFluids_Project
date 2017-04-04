import numpy as np

from external_mag_field import B_ext

square = np.square
sqrt = np.sqrt


def gravity_force(a, rho_bead, rho_water):
    """Computes the vectorial gravitational force acting on a single bead

    Args:
        a         : Bead radius in m
        rho_bead  : Density of bead in kg/m3
        rho_water : Density of water in kg/m3

    Returns:
        F_g       : A column vector, where each component represents the force
                    in the x, y and z direction respectively
    """
    # Bouyant mass of magnetic bead in kg
    m = ((4.0*np.pi*a**3)/3.0) * (rho_bead-rho_water)

    # Acceleration due to gravity in m/s^2
    g = 9.8

    F_g = -1*m*g*np.mat([0, 0, 1]).T
    return F_g


def magnetic_moment(bead_r, alpha, tilt_angle, semi_cone_angle,
                   omega, t, mag_sus, r_list):
    """Calculates the magnetic moments of each bead

        Args:
            bead_r          : Bead radius in m
            alpha           : Pumping direction in radians
            tilt angle      : Cone tilt angle from vertical in radians
            semi_cone_angle : Half of the cone angle in radians
            omega           : Angular velocity in radians/s
            t               : Time in s
            mag_sus         : Magnetic susceptibility of beads
            r_list          : List of all position vectors of beads in column
                              form
        Returns:
            r_comb_list     : A list containing  r_j1, r_j2, r_j3, ... etc
                              where, r_j1 = [r_11, r_21, r_31, ...., r_n1]
                                     r_j2 = [r_12, r_22, r_32, ...., r_n2]

            r_mod_comb_list : A list containing mod(r_j1), mod(r_j2),..mod(r_jn)
                              where, r_j1 = [|r_11|, |r_21|, |r_31|,.., |r_n1|]
                                     r_j2 = [|r_12|, |r_22|, |r_32|,.., |r_n2|]

            m_vec_list      : A list containing magnetic moments m1, m2,.....mn
                              of each bead
    """
    mu_o = 4.0*np.pi*1e-7 # Permeability of free space in T.m/A
    k_const = mu_o/((4.0/3.0)*np.pi*bead_r**3*mag_sus) # B_i = k_const * m_i
    c = mu_o/(4.0*np.pi)

    # External magnetic field
    mag_ext = B_ext(alpha, tilt_angle, semi_cone_angle, omega, t)

    # Stores list of r_ji combinations of each i
    # eg: r_comb_list = [r_j1, r_j2, r_j3, ...]
    # where, r_j1 = [r_11, r_21, r_31, ...., r_n1]
    #        r_j2 = [r_12, r_22, r_32, ...., r_n2]
    r_comb_list = []
    r_mod_comb_list = [] # Same as r_comb_list but stores modulus of each r_ji

    # Have to solve equation of the form a.x = b, where x is a list that
    # consists of components  mx1, my1, mz1, mx2, my2, mz2 ...... etc
    # where each triplet (mx_i, my_i, mz_i) are components of m_i
    # The total magnetic field for a bead (sum of external + contributions from
    # other beads) is then given by B_i = k.m_i
    a = []
    b = []

    n = len(r_list)

    for i, r_i in enumerate(r_list):
        r_ji_list = []
        mod_r_ji_list = []
        for j, r_j in enumerate(r_list):
            if i != j:
                v = r_j - r_i
                r_ji_list.append(v)
                mod_r_ji_list.append(sqrt(np.sum(square(v))))
            else:
                # This value won't be used it is just for index to be not empty
                r_ji_list.append(r_j - r_i)
                mod_r_ji_list.append(1)

        r = r_ji_list[:]
        r_comb_list.append(r)

        r_mod_comb_list.append(mod_r_ji_list)
        mod_r_ji_arr = np.array(mod_r_ji_list)

        t = -1*c*(1.0/mod_r_ji_arr**3)
        u = 3*c*(1.0/mod_r_ji_arr**5)

        p = 3*i

        l = np.zeros((3, n*3))
        l[0:3, p:p+3] = -k_const*np.identity(3)

        for k in range(0, n*3, 3):
            if k != p:
                g = k/3
                l1 = [t[g]+u[g]*r[g][0, 0]**2, u[g]*r[g][1, 0]*r[g][0, 0],
                      u[g]*r[g][2, 0]*r[g][0, 0]
                     ]
                l2 = [u[g]*r[g][0, 0]*r[g][1, 0], t[g]+u[g]*r[g][1, 0]**2,
                      u[g]*r[g][2, 0]*r[g][1, 0]
                     ]
                l3 = [u[g]*r[g][0, 0]*r[g][2, 0],  u[g]*r[g][1, 0]*r[g][2, 0],
                      t[g]+u[g]*r[g][2, 0]**2
                     ]
                l[0:3, k:k+3] = np.array([l1[:], l2[:], l3[:]])

        for l_arr in l:
            a.append(l_arr)

        for k in range(3):
            b.append(-1.0*mag_ext[k, 0])


    a = np.array(a)
    b = np.array(b)

    # Array consisting of mx1, my1, mz1, mx2, my2, mz2.....etc
    m_arr = np.linalg.solve(a, b)

    # list containing column vectors m1 = [mx1 , my1, mz1], m2 = [mx2, my2,
    # mz2], ...... etc
    m_vec_list  = []

    for k in range(0, n*3, 3):
        m_vec_list.append(np.mat([m_arr[k],
                                  m_arr[k+1],
                                  m_arr[k+2]
                                 ]
                                ).T
                         )

    return r_comb_list, r_mod_comb_list, m_vec_list



def magnetic_force(list_r_ji, r_mod_comb_list, m_vec_list):
    """Calculates the total magnetic force acting on each bead induced by other 
    beads

    Args:
        list_r_ji       : A list containing  r_j1, r_j2, r_j3, ... etc
                          where, r_j1 = [r_11, r_21, r_31, ...., r_n1]
                                 r_j2 = [r_12, r_22, r_32, ...., r_n2]

        r_mod_comb_list : A list containing mod(r_j1), mod(r_j2),..mod(r_jn)
                          where, r_j1 = [|r_11|, |r_21|, |r_31|,.., |r_n1|]
                                 r_j2 = [|r_12|, |r_22|, |r_32|,.., |r_n2|]

        m_vec_list      : A list containing magnetic moments m1, m2,.....mn
                          of each bead

    Returns:
           list_Fi      : It consists of the total magnetic force induced on
                          bead i by all other beads, where i = 1, 2, 3,..., n
    """
    mu_o = 4.0*np.pi*1e-7 # Permeability of free space in T.m/A
    c = mu_o/(4.0*np.pi)

    # Stores total magnetic force on bead i which is the sum of
    # all magnetic forces induced by other beads
    list_Fi = []

    m = m_vec_list[:]

    # To calculate dot product
    dot = np.tensordot

    for i, r_i in enumerate(list_r_ji):
        F_ji = []
        for j, r_ji in enumerate(r_i):
            if i != j :
                d1 = dot(m[j], m[i], axes=2)
                d2 = dot(m[i], r_ji, axes=2)
                d3 = dot(m[j], r_ji, axes=2)

                mod_r_ji = r_mod_comb_list[i][j]

                a1 = (3.0/mod_r_ji**5) * (r_ji*d1+m[j]*d2+m[i]*d3)
                a2 = (-15.0/mod_r_ji**7) * (r_ji*d3*d2)
                F_ji.append(c*(a1+a2))

        list_Fi.append(np.sum(F_ji, axis=0))

    return list_Fi

