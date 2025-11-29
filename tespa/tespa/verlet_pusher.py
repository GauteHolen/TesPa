
import numpy as np
from numba import njit

@njit
def verlet_pusher_Fext(q,m,E_vec,B_vec,r,v,dt,Fext):
    """_summary_

    Args:
        q (float): charge
        m (float): mass
        E_vec (3D array): Electric field at position r
        B_vec (3D array): Magnetic field at position r
        r (3D vector): position vector
        v (3D vector): velocity vector
        dt (float): timestep
        Fext (3D vector): External force to be applied

    Returns:
        (3D array,3D array): position and velocity vector
    """
    a = (q * (E_vec + np.cross(v, B_vec)) + Fext) / m

    # Update position (uses current velocity + acceleration)
    r_new = r + v * dt + 0.5 * a * dt**2

    # Compute fields at the new position if needed.
    # For now assume constant fields (replace this section):
    E_new = E_vec
    B_new = B_vec

    # Acceleration at new position (explicit: uses old velocity)
    a_new = (q * (E_new + np.cross(v, B_new)) + Fext) / m

    # Velocity update (uses a and a_new)
    v_new = v + 0.5 * (a + a_new) * dt

    return r_new, v_new