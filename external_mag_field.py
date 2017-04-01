import numpy as np

cos = np.cos
sin = np.sin

def B_ext(alpha, tilt_angle, semi_cone_angle, omega, t):
    """Calculates the time dependent magnetic field where
        Args:
            alpha           : Pumping direction
            tilt angle      : Cone tilt angle from vertical
            semi_cone_angle : Half of the cone angle
            omega           : angular velocity
            t               : time

        Returns:
            A 3x1 matrix with each component denoting the external magnetic
            field strength in respective direction
    """
    B1_ext = np.matrix(np.zeros((3, 3)))
    B2_ext = np.matrix(np.zeros((3, 3)))
    B3_ext = np.matrix(np.zeros((3, 1)))

    B1_ext[0, 0], B1_ext[0, 1]  = cos(alpha), -1*sin(alpha)
    B1_ext[1, 0], B1_ext[1, 1]  = sin(alpha), cos(alpha)
    B1_ext[2, 2] = 1.0

    B2_ext[0, 0], B2_ext[0, 2]  = cos(tilt_angle), sin(tilt_angle)
    B2_ext[1, 1] = 1.0
    B2_ext[2, 0], B2_ext[2, 2]  = -1*sin(tilt_angle), cos(tilt_angle)

    B3_ext[0, 0] = -1*sin(semi_cone_angle)*cos(omega*t)
    B3_ext[1, 0] = sin(semi_cone_angle)*sin(omega*t)
    B3_ext[2, 0] = cos(semi_cone_angle)

    return B1_ext*B2_ext*B3_ext
