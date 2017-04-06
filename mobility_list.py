from self_mobility import self_mob

from cross_mobility import cross_mob_bead


def bead_moblity(a, eta, r_list):
    """Returns a list of lists where each inner list contains both the
       cross-mobilities and self-mobility of each bead.

    Args:
        a        : Bead radius in m
        eta      : Water viscosity in kg/m3
        r_list   : List of all position vectors of beads in column
                   form

    Returns:
        mob_list : A list containing mu_1, mu_2, mu_3, ......, mu_n
                   where, mu_1 = [mu_11, mu_12, mu_13, ......, mu_1n]
                          mu_2 = [mu_21, mu_22, mu_23, ......, mu_2n]
                   mu_ij => cross-mobility
                   mu_ii => self-mobility
    """
    mob_list = []

    for i, r_i in enumerate(r_list):
        mu_i = []
        for j, r_j in enumerate(r_list):
            if i != j:
                mu_i.append(cross_mob_bead(a, eta, r_i, r_j))
            else:
                mu_i.append(self_mob(a, eta, r_j[2, 0]))
        mob_list.append(mu_i)

    return mob_list
