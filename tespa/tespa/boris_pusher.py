from numba import njit
import numpy as np

@njit
def boris_pusher(q,m,E_vec,B_vec,r,v,dt):
    """_summary_

    Args:
        q (float): charge
        m (float): mass
        E_vec (3D array): Electric field at position r
        B_vec (3D array): Magnetic field at position r
        r (3D vector): position vector
        v (3D vector): velocity vector
        dt (float): timestep

    Returns:
        (3D array,3D array): position and velocity vector
    """
    t = q / m * B_vec * 0.5 * dt
    
    s = 2.0 * t / (1.0 + np.dot(t, t))
    v_minus = v + q / m  * E_vec * 0.5 * dt
    v_prime = v_minus + np.cross(v_minus, t)
    v_plus = v_minus + np.cross(v_prime, s)
    v = v_plus + q / (m ) * E_vec * 0.5 * dt
    r += v * dt 
    return r,v


@njit
def boris_pusher_Fext(q,m,E_vec,B_vec,r,v,dt,Fext):
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
    #check E_vec not zero
    v += 0.5*dt*Fext/m
    t = q / m * B_vec * 0.5 * dt
    
    s = 2.0 * t / (1.0 + np.dot(t, t))
    v_minus = v + q / m * E_vec * 0.5 * dt
    v_prime = v_minus + np.cross(v_minus, t)
    v_plus = v_minus + np.cross(v_prime, s)
    v = v_plus + q / (m ) * E_vec * 0.5 * dt
    v += 0.5*dt*Fext/m
    r += v * dt 
    return r,v