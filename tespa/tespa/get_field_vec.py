from .interpolate_cubic import cubic_interp_3d_jit
from numba import njit


@njit()
def get_field_vec3(r,F,Fx,Fy,Fz,dr):
    """Get tricubic interpolated 3D vector of 3-component vector fields in 3D space

    Args:
        r (3D vector): position vector
        F (3D vector): field vector 
        Fx (3D array): x-component of the field in 3D space
        Fy (3D array): y-component of the field in 3D space
        Fz (3D array): z-component of the field in 3D space
        dr (float): gridspace of the field

    Returns:
        3D vector: interpolated 3-component field vector
    """

    F[0] = cubic_interp_3d_jit(Fx,r[0],r[1],r[2],dr,dr,dr)
    F[1] = cubic_interp_3d_jit(Fy,r[0],r[1],r[2],dr,dr,dr)
    F[2] = cubic_interp_3d_jit(Fz,r[0],r[1],r[2],dr,dr,dr)

    return F