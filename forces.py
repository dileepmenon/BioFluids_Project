import numpy as np

from external_mag_field import B_ext

from mobility_list import bead_moblity

square = np.square
sqrt = np.sqrt


def gravity_force(a, rho_bead, rho_water):
    """Computes the vectorial gravitational force acting on a single bead.

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
    """Calculates the magnetic moments of each bead.

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
    beads.

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


def contact_force(F_g, list_r_ji, list_unit_rji, mob_list, list_Fb):
    """Finds the contact force acting on each bead due to neighbouring
    beads.

    Args:
        F_g             : A column vector, where each component represents the
                          gravity force in the x, y and z direction
                          respectively

        list_r_ji       : A list containing  r_j1, r_j2, r_j3, ... etc
                          where, r_j1 = [r_11, r_21, r_31, ...., r_n1]
                                 r_j2 = [r_12, r_22, r_32, ...., r_n2]

        list_unit_rji   : A list containing unit vectors of r_j1, r_j2,..r_jn
                          where, unit(r_j1) = [unit(r_11), unit(r_21),
                                               unit(r_31),.., unit(r_n1)
                                              ]
                                 unit(r_j2) = [unit(r_12), unit(r_22),
                                               unit(r_32),.., unit(r_n2)
                                              ]

        mob_list        : A list containing mu_1, mu_2, mu_3, ......, mu_n
                          where, mu_1 = [mu_11, mu_12, mu_13, ......, mu_1n]
                                 mu_2 = [mu_21, mu_22, mu_23, ......, mu_2n]

        list_Fb         : A list containing the total magnetic force induced on
                          bead i by all other beads, where i = 1, 2, 3,..., n

    Returns:
        list_F_contact : A list containing the contact forces acting on
                         neighbouring beads. For n beads there will be n-1
                         contact forces


    """
    # To solve ax = b, where x is a column vector [x1, x2, x3, ......]
    # [F_12 = x1*unit(r_12), F_23 = x2*unit(r_23),......, F_n-1(n) =
    # x(n-1)*unit(r_(n-1)n)];  n => number of beads
    n = len(list_Fb)
    a = np.zeros((n-1, n-1))
    a_row_1 = []
    b = []

    # To find first row of 'a' subject to condition r_21.(v2 - v1) = 0
    # Here vi represents the velocity of bead i
    # v1 = 0 as the 1st bead is fixed and hence r_21.v2 = 0
    dot = np.tensordot

    r_21 = list_r_ji[0][1]
    unit_r_12 = list_unit_rji[1][0] # Note index change compared to r_21

    k_21 = -1.0*mob_list[1][0]*(mob_list[0][0].I)
    mu_12, mu_22 = mob_list[0][1], mob_list[1][1]

    vec_coeff = ((k_21*mu_12)+mu_22)*unit_r_12

    # Coeffcient in (row-1, col-1) of matrix a
    a_row1_coeff1 = dot(r_21, vec_coeff, axes=2)
    a_row_1.append(a_row1_coeff1)

    # To find coefficients in (row-1, col-i) of matrix 'a' where i = 2, 3, 4...
    for i, r_i in enumerate(list_unit_rji[2:], 1):
        unit_r = r_i[i]
        vec_coeff = (-k_21*mob_list[0][i])*(unit_r) \
                     + (k_21*mob_list[0][i+1])*(unit_r) \
                     - (mob_list[1][i]*unit_r) \
                     + (mob_list[1][i+1]*unit_r)
        a_row1_coeffi = dot(r_21, vec_coeff, axes=2)
        a_row_1.append(a_row1_coeffi)

    a[0] = a_row_1[:]

    # To find first element of b-matrix
    coeff_sum = 0.0
    for k, B_k in enumerate(list_Fb[1:], 1):
        Fg_Bk = F_g + B_k
        l =  (k_21*mob_list[0][i]+mob_list[1][i])*Fg_Bk
        coeff_sum += dot(r_21, l, axes=2)

    b.append(-coeff_sum)

    # To find coefficients of (row-i, col-1) where i = 2, 3, 4, ... of matrix
    # 'a' subject to condition  r_(i+1)(i).(v(i+1) - vi) = 0 where
    # i = 2, 3, 4, ....n; vi represents the velocity of bead i and v(i+1)
    # of bead i+1
    mu_11_inv = mob_list[0][0].I # Inverse of matrix mu_11

    for j, r_j in enumerate(list_r_ji[1:-1], 1):
        r = r_j[j+1]

        k1 = -1.0*mob_list[j][0]*mu_11_inv
        k2 = -1.0*mob_list[j+1][0]*mu_11_inv
        k = k2 - k1

        c = mob_list[j+1][1] - mob_list[j][1]

        vec_coeff = (k*mu_12 + c)*unit_r_12
        a_rowi_coeff1 = dot(r, vec_coeff, axes=2)
        a[j][0] = a_rowi_coeff1

    #To find (row-k, col-i) coefficients of matrix 'a' where k = 2, 3, 4,..
    # and i = 2, 3, 4, ...
    for k, r_k in enumerate(list_r_ji[1:-1], 1):
        a_row_i = []
        for i, r_i in enumerate(list_unit_rji[2:], 1):
            unit_r = r_i[i]

            k1 = -1.0*mob_list[k][0]*mu_11_inv
            k2 = -1.0*mob_list[k+1][0]*mu_11_inv

            vec_coeff = ((-k2+k1)*mob_list[0][i])*(unit_r) \
                        + ((k2-k1)*mob_list[0][i+1])*(unit_r) \
                        + ((-mob_list[k+1][i]+mob_list[k][i])*unit_r) \
                        + ((mob_list[k+1][i+1]-mob_list[k][i+1])*unit_r)

            a_rowk_coeffi = dot(r_k[k+1], vec_coeff, axes=2)
            a_row_i.append(a_rowk_coeffi)
        a[k][1:] = a_row_i[:]

    # To find [b2, b3, b4 .....] of b-matrix
    for m, r_m in enumerate(list_r_ji[1:-1], 1): # check index
        coeff_sum = 0.0
        for k, B_k in enumerate(list_Fb[1:], 1):
            Fg_Bk = F_g + B_k

            k1 = -1.0*mob_list[m][0]*mu_11_inv
            k2 = -1.0*mob_list[m+1][0]*mu_11_inv

            l =  ((k2-k1)*mob_list[0][k]
                  + (mob_list[m+1][k] - mob_list[m][k])
                 )*Fg_Bk
            coeff_sum += dot(r_m[m+1], l, axes=2)
        b.append(-coeff_sum)

    # Array containing scalars x1, x2 ,x3, ..... where contact forces are given
    # by F_12 = x1*unit(r_12), F_23 = x2*unit(r_23),......, F_n-1(n) =
    # x(n-1)*unit(r_(n-1)n);  n => number of beads
    arr_coeff_F_con = np.linalg.solve(np.array(a), np.array(b))
    list_F_contact = []

    for i, (r_i, x_i) in enumerate(zip(list_unit_rji[1:], arr_coeff_F_con)):
        list_F_contact.append(x_i*r_i[i])

    return list_F_contact



def total_force(bead_r, eta, rho_bead, rho_water, alpha, tilt_angle,
                semi_cone_angle, omega, t, mag_sus, r_list):
    """
    Computes the total force (gravity+magnetic+contact+constraint) acting on
    each bead.

    Args:
        bead_r            : Bead radius in m
        eta               : Water viscosity in kg/m3
        rho_bead          : Density of bead in kg/m3
        rho_water         : Density of water in kg/m3
        alpha             : Pumping direction in radians
        tilt angle        : Cone tilt angle from vertical in radians
        semi_cone_angle   : Half of the cone angle in radians
        omega             : Angular velocity in radians/s
        t                 : Time in s
        mag_sus           : Magnetic susceptibility of beads
        r_list            : List of all position vectors of beads in column
                            form

    Returns:
        list_total_force  : List containing the total force acting on each
                            bead
    """
    F_g = gravity_force(bead_r, rho_bead, rho_water)

    list_r_ji, list_mod_r_ji, m_vec_list = magnetic_moment(bead_r, alpha,
                                                           tilt_angle,
                                                           semi_cone_angle,
                                                           omega, t, mag_sus,
                                                           r_list
                                                          )
    # Unit vectors of r_ji list
    list_unit_r_ji = []
    arr_r_ji = np.array(list_r_ji, dtype=float)
    arr_mod_r_ji = np.array(list_mod_r_ji, dtype=float)

    for r_i, mod_ri in zip(arr_r_ji, arr_mod_r_ji):
        list_unit_r_ji.append(list(r_i/mod_ri))

    # List containing total magnetic force acting on each bead
    list_Fb = magnetic_force(list_r_ji, list_mod_r_ji, m_vec_list)

    # List containing mobilites of each bead as list
    mob_list = bead_moblity(bead_r, eta, r_list)

    # List containing contact force between neighbouring beads
    list_F_contact = contact_force(F_g, list_r_ji, list_unit_rji,
                                   mob_list, list_Fb
                                  )
    # list containing total forces acting on bead j where j = 2, 3, 4... n
    list_total_force_j = []

    # Total force for beads 2, 3.... n-1
    for i, F_b in enumerate(list_Fb[1:-1]):
        list_total_force_j.append(F_g
                                  + F_b
                                  + list_F_contact[i]
                                  - list_F_contact[i+1]
                                 )
    # Adding nth beads total force
    list_total_force_j += [F_g + F_b[-1] + list_F_contact[-1]]

    # For finding first beads total force
    s = np.mat([0.0, 0.0, 0.0]).T
    for f, mu in zip(list_total_force_j, mob_list[0][1:]):
       s += mu*f

    mu_11_inv = mob_list[0][0].I

    # Total force acting on bead 1
    F1 = -mu_11_inv*s

    list_total_force = [F1] + list_total_force_j

    return list_total_force
