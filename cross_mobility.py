import numpy as np

square = np.square
sqrt = np.sqrt


def mu_RP(a, eta, r, mod_r, unit_r):
    """Computes and returns the Rotne-Prager tensor in unbounded fluid at
    column vector r
    """
    const = 1.0/(8*np.pi*eta)
    I = np.mat(np.identity(3))

    # To get tensor product of vector r with itself
    ten_prod = np.mat(np.zeros((3, 3)))
    ten_mat = np.tensordot(r.T, r.T, axes=0)
    for num, i in enumerate(ten_mat[0]):
        ten_prod[num] = i[0]

    m = (1.0/mod_r) * (I+ten_prod)
    n = ((2.0*a**2)/(3.0*mod_r**3)) * (I-3*ten_prod)

    return const*(m+n)


def cross_mob_bead(a, eta, x_vec_i, x_vec_j):
    """Calculates and returns the cross mobility matrix of a bead i with
    respect to bead j.

        Args:
            a       : Read radius in m
            eta     : Water viscosity in kg/m3
            x_vec_i : Column vector of x, y, z of bead i
            x_vec_j : Column vector of x, y, z of bead j
    """
    # z-cordinate of bead i in m
    z_i = x_vec_i[2, 0]

    # z-cordinate of bead j in m
    z_j = x_vec_j[2, 0]

    # Conjugate of vector x_vec_j is such that only z-cord will be negative
    x_vec_jconj = x_vec_j.copy()
    x_vec_jconj[2, 0] = -1*x_vec_j[2, 0]

    r_ij = x_vec_i - x_vec_j
    mod_r_ij = sqrt(np.sum(square(r_ij))) # modulus of r_ij
    unit_r_ij = r_ij / mod_r_ij  # Unit vector of r_ij

    r_ijconj = x_vec_i - x_vec_jconj
    mod_r_ijconj = sqrt(np.sum(square(r_ijconj))) # modulus of r_ij conjugate
    unit_r_ijconj = r_ijconj / mod_r_ijconj # Unit vector of r_ij conjugate

    # Contributions of the source doublet and stokeslet doublet taken care by
    # tensor delta_mu
    delta_mu = np.mat(np.zeros((3, 3)))

    # Coefficients of tensor delta_mu
    const = 1.0/(4*np.pi*eta)

    # Coefficient in (row-1, col-1) and (row-2, col-2) of tensor delta_mu
    c1 = (z_i*z_j) * (-1.0*(1.0/mod_r_ijconj**3)
                      + 3.0*(r_ijconj[0, 0]**2/mod_r_ijconj**5)
                     )
    c2 = (a**2*r_ijconj[2, 0]**2) * (1.0/mod_r_ijconj**5
                                     - 5.0*(r_ijconj[0, 0]**2/mod_r_ijconj**7)
                                    )
    c3 = (a**4/3.0) * (1.0/mod_r_ijconj**5
                       - 5.0*((r_ijconj[0, 0]**2+r_ijconj[2, 0]**2)/mod_r_ijconj**7)
                       + 35.0*((r_ijconj[0, 0]**2*r_ijconj[2, 0]**2)/mod_r_ijconj**9)
                      )
    # Below expression also for coeff_22
    coeff_11 = const*(c1 + c2 +c3)


    # Coefficient in (row-3, col-3) of tensor delta_mu
    d1 = (z_i*z_j) * (1.0*(1.0/mod_r_ijconj**3)
                      - 3.0*(r_ijconj[2, 0]**2/mod_r_ijconj**5)
                     )
    d2 = (a**2*r_ijconj[2, 0]**2) * (-3.0/mod_r_ijconj**5
                                     + 5.0*(r_ijconj[2, 0]**2/mod_r_ijconj**7)
                                    )
    d3 = (a**4/3.0) * (-3.0/mod_r_ijconj**5
                       + 30.0*(r_ijconj[2, 0]**2/mod_r_ijconj**7)
                       - 35.0*(r_ijconj[2, 0]**4/mod_r_ijconj**9)
                      )
    coeff_33 = const*(d1 + d2 +d3)

    # Coefficient in (row-1, col-2) and (row-2, col-1) of tensor delta_mu
    e1 = (3.0*z_i*z_j) * ((r_ijconj[0, 0]*r_ijconj[1, 0])/mod_r_ijconj**5)

    e2 = (-5.0*a**2*r_ijconj[2, 0]**2) * ((r_ijconj[0, 0]*r_ijconj[1, 0])/mod_r_ijconj**7)
    e3 = ((5.0*a**4)/3.0) * ((r_ijconj[0, 0]*r_ijconj[1, 0])
                             * (-1.0/mod_r_ijconj**7
                               + 7.0*(r_ijconj[2, 0]**2/mod_r_ijconj**9)
                               )
                            )
    coeff_12 = const*(e1 + e2 +e3)


    # Coefficient in (row-1, col-3) and (row-3, col-1) of tensor delta_mu
    f1 = (z_j*r_ijconj[0, 0]) * (1.0*(1.0/mod_r_ijconj**3)
                                 - 3.0*((z_i*r_ijconj[2, 0])/mod_r_ijconj**5)
                                )
    f2 = (a**2*r_ijconj[0, 0]*r_ijconj[2, 0]) * (-2.0/mod_r_ijconj**5
                                                 + 5.0*(r_ijconj[2, 0]**2/mod_r_ijconj**7)
                                                )
    f3 = ((5.0*a**4*r_ijconj[0, 0]*r_ijconj[2, 0])/3.0) * (3.0/mod_r_ijconj**7
                                                           - ((7.0*(r_ijconj[2, 0]**2))
                                                              /mod_r_ijconj**9
                                                             )
                                                          )
    coeff_13 = const*(f1 + f2 + f3)

    # Coefficient in (row-3, col-2) and (row-2, col-3) of tensor delta_mu
    g1 = (z_j*r_ijconj[1, 0]) * (1.0*(1.0/mod_r_ijconj**3)
                                 + 3.0*((z_i*r_ijconj[2, 0])/mod_r_ijconj**5)
                                )
    g2 = (-5.0*a**2*r_ijconj[1, 0]) * (r_ijconj[2, 0]**3/mod_r_ijconj**7)
    g3 = ((5.0*a**4*r_ijconj[1, 0]*r_ijconj[2, 0])/3.0) * (-3.0/mod_r_ijconj**7
                                                           + ((7.0*(r_ijconj[2, 0]**2))
                                                              /mod_r_ijconj**9
                                                             )
                                                          )
    coeff_32 = const*(g1 + g2 + g3)


    ten_prod[0, 0], ten_prod[1, 1], ten_prod[2, 2] = (coeff_11,
                                                      coeff_11,
                                                      coeff_33,
                                                     )
    ten_prod[0, 1], ten_prod[1, 0] = coeff_12, coeff_12
    ten_prod[0, 2], ten_prod[2, 0] = coeff_13, coeff_13
    ten_prod[1, 2], ten_prod[2, 1] = coeff_32, coeff_32


    mu = mu_RP(a, eta, r_ij, mod_r_ij, unit_r_ij) \
         - mu_RP(a, eta, r_ijconj, mod_r_ijconj, unit_r_ijconj) \
         + delta_mu

    return mu







