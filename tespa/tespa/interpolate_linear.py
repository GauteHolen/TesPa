import numpy as np
from numba import njit
@njit


def get_cube_values(data, x, y, z, dx ,dy, dz):
    x0 = int(np.floor(x / dx))
    x1 = x0 + 1
    y0 = int(np.floor(y / dy))
    y1 = y0 + 1
    z0 = int(np.floor(z / dz))
    z1 = z0 + 1
    P000 = data[x0, y0, z0]
    P100 = data[x1, y0, z0]
    P010 = data[x0, y1, z0]
    P110 = data[x1, y1, z0]
    P001 = data[x0, y0, z1]
    P101 = data[x1, y0, z1]
    P011 = data[x0, y1, z1]
    P111 = data[x1, y1, z1]
    return [P000, P100, P010, P110, P001, P101, P011, P111]

@njit
def trilinear_interp(data, x,y,z, dx,dy,dz):
    """
    Perform trilinear interpolation on a cube of values in 3D.

    Parameters:
    cube_values : array-like
        The values at the 8 corners of the cube, ordered as:
        [P000, P100, P010, P110, P001, P101, P011, P111].
    tx, ty, tz : float
        The fractional distances along the x, y, and z axes (0 <= t <= 1).
    
    Returns:
    float
        The interpolated value at the point (tx, ty, tz).
    """
    

    cube_values = get_cube_values(data, x, y, z, dx, dy, dz)
    tx = x - np.floor(x)
    ty = y - np.floor(y)
    tz = z - np.floor(z)

    # Reshape cube values for easy access to corner values
    cube_values = np.array(cube_values).reshape(2, 2, 2)
    
    # Interpolate along the x-axis
    c00 = np.interp(tx, [0, 1], [cube_values[0, 0, 0], cube_values[1, 0, 0]])
    c10 = np.interp(tx, [0, 1], [cube_values[0, 1, 0], cube_values[1, 1, 0]])
    c01 = np.interp(tx, [0, 1], [cube_values[0, 0, 1], cube_values[1, 0, 1]])
    c11 = np.interp(tx, [0, 1], [cube_values[0, 1, 1], cube_values[1, 1, 1]])
    
    # Interpolate along the y-axis
    c0 = np.interp(ty, [0, 1], [c00, c10])
    c1 = np.interp(ty, [0, 1], [c01, c11])
    
    # Interpolate along the z-axis
    c = np.interp(tz, [0, 1], [c0, c1])
    
    return c
