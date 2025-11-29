from numba import njit
import numpy as np

kb = 1.380649e-23  # Boltzmann constant in J/K
q_elementary = 1.602176634e-19  # Elementary charge in C

@njit
def ion_drag_force(ni, Ti, a, mi, v_rel, q, phi):
    """Calculate the ion drag force on a dust particle.

    Args:
        ni (float): Ion number density (m^-3)
        Ti (float): Ion temperature (K)
        a (float): Dust particle radius (m)
        v_rel (float): Relative velocity between dust and ions (m/s)

    Returns:
        float: Ion drag force (N)
    """
    # Placeholder implementation; replace with actual physics-based calculation
    v_rel_mag = np.linalg.norm(v_rel)
    #if v_rel_mag == 0:
    #    return np.zeros(3)
    #else:
    v_unit = v_rel / v_rel_mag
    v_th_i= v_thermal(Ti, mi)
    sigma = np.pi * a**2 
    v_s = np.sqrt(v_rel_mag**2 + v_th_i**2)
    bc = impact_parameter(a, q, phi, mi, v_s)

    F_drag_magnitude = ni * mi * v_s * v_rel_mag * np.pi * bc**2
    drag_coefficient = 1e0  # Example coefficient
    #F_drag = drag_coefficient * mi * ni * sigma * v_unit * v_th_i**2
    #print("F_ion_drag:", F_drag)
    return F_drag_magnitude * v_unit
    


    
@njit
def v_min_b(q,phi,m):
    return np.sqrt((2 * q * phi) / m)



@njit
def v_thermal(T, m):
    """Calculate the thermal velocity.

    Args:
        T (float): Temperature (K)
        m (float): Mass (kg)

    Returns:
        float: Thermal velocity (m/s)
    """
     
    return np.sqrt(2 * kb * T / m)

@njit
def impact_parameter(a, q,phi, m, v):
    v_min = v_min_b(q,phi,m)
    if v < v_min:
        return 0.0
    else:
        return a * np.sqrt(1 - ((2*q * phi)/(m * v**2)) )  # Placeholder formula

def K_collision():
    return None



@njit
def neutral_drag_force(beta, m, v):
    """Calculate the neutral drag force.

    Args:
        beta (float): Drag coefficient
        m (float): Mass of the dust particle (kg)
        v (3D vector): Velocity vector of the dust particle (m/s)

    Returns:
        3D vector: Neutral drag force (N)
    """

    return -beta * m * v
