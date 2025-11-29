import numpy as np
from numba import njit

@njit
def lagrange_basis(x, xi, x_points):
    """
    Compute the Lagrange basis polynomial L(x) for a given set of x_points.
    
    Parameters:
    x : float
        The x-coordinate at which to evaluate the basis polynomial.
    xi : int
        The index of the current basis polynomial to compute.
    x_points : list of floats
        The x-coordinates of the data points.
        
    Returns:
    L : float
        The value of the Lagrange basis polynomial at x.
    """
    L = 1.0
    for j in range(len(x_points)):
        if j != xi:
            L *= (x - x_points[j]) / (x_points[xi] - x_points[j])
    return L

@njit
def cubic_interp_1d(x_points, y_points, x_new):
    """
    Perform cubic interpolation (Lagrange) for given x, y data points manually.
    
    Parameters:
    x_points : list of floats
        x-coordinates of the data points (4 values).
    y_points : list of floats
        y-coordinates of the data points (values to interpolate, 4 values).
    x_new : float
        The x-coordinate at which to interpolate.
        
    Returns:
    interpolated_value : float
        Interpolated value at x_new using cubic interpolation.
    """
    # Ensure we have exactly 4 points for cubic interpolation
    if len(x_points) != 4 or len(y_points) != 4:
        raise ValueError("Exactly 4 data points are required for cubic interpolation.")
    
    # Compute the interpolation using Lagrange polynomials
    interpolated_value = 0.0
    for i in range(4):
        L = lagrange_basis(x_new, i, x_points)
        interpolated_value += y_points[i] * L
    
    return interpolated_value


@njit
def cubic_interp_3d_jit(data, x, y, z, dx=1.0, dy=1.0, dz=1.0):
    """
    Perform cubic interpolation in 3D for the given data array at position (x, y, z)
    with spacing dx, dy, dz.
    
    Parameters:
    data : ndarray
        3D array with the data to interpolate.
    x, y, z : float
        The coordinates where interpolation is desired.
    dx, dy, dz : float
        The grid spacing along the x, y, z axes respectively.
    
    Returns:
    interpolated_value : float
        Interpolated value at (x, y, z).
    """
    # Normalize the coordinates to grid indices
    x_idx = x / dx
    y_idx = y / dy
    z_idx = z / dz
    
    # Get the integer indices around the point for 4x4x4 interpolation
    x0 = max(min(int(np.floor(x_idx)) - 1, data.shape[0] - 4), 0)
    y0 = max(min(int(np.floor(y_idx)) - 1, data.shape[1] - 4), 0)
    z0 = max(min(int(np.floor(z_idx)) - 1, data.shape[2] - 4), 0)
    
    # Create the grid of surrounding points (4x4x4 cube)
    x_points = np.zeros(4)
    y_points = np.zeros(4)
    z_points = np.zeros(4)
    
    for i in range(4):
        x_points[i] = (x0 + i) * dx
        y_points[i] = (y0 + i) * dy
        z_points[i] = (z0 + i) * dz
    
    # First, interpolate along the x-axis for each of the y-z planes
    interpolated_along_x = np.zeros((4, 4))
    for j in range(4):
        for k in range(4):
            interpolated_along_x[j, k] = cubic_interp_1d(x_points, data[x0:x0+4, y0+j, z0+k], x)
    
    # Then, interpolate along the y-axis for each of the z rows
    interpolated_along_y = np.zeros(4)
    for k in range(4):
        interpolated_along_y[k] = cubic_interp_1d(y_points, interpolated_along_x[:, k], y)
    
    # Finally, interpolate along the z-axis
    interpolated_value = cubic_interp_1d(z_points, interpolated_along_y, z)
    
    return interpolated_value