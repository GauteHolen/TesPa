import numpy as np

def inside_cyl(R,Z,r,z1,z2,dr=0):
    d = 0.5*dr
    valid_r = R <= r+d
    valid_z = (Z >= z1-d) & (Z <= z2+d)
    return valid_r & valid_z

def inside_box(X,Y,Z,x1,x2,y1,y2,z1,z2,dr):
    d = 0.5*dr
    valid_x = (X >= x1-d) & (X <= x2+d)
    valid_y = (Y >= y1-d) & (Y <= y2+d)
    valid_z = (Z >= z1-d) & (Z <= z2+d)
    return valid_x & valid_y & valid_z


def random_points_in_circle(N, r, c, r0=0):
    """
    Generate random points within a circle of radius r centered at point c in 2D.

    Parameters:
    N : int
        Number of random points to generate.
    r : float
        Radius of the circle.
    c : array-like
        2D coordinates of the center point (shape: 2,).

    Returns:
    points : ndarray
        Array of random points with shape (N, 2).
    """
    # Step 1: Generate random polar coordinates
    # Radial distances (uniform in area, hence square root)
    radial_distances = np.sqrt(np.random.uniform(r0**2, r**2, N))
    # Azimuthal angles (theta) uniformly distributed between 0 and 2pi
    theta = np.random.uniform(0, 2 * np.pi, N)
    
    # Step 2: Convert polar coordinates to Cartesian coordinates
    x = radial_distances * np.cos(theta)
    y = radial_distances * np.sin(theta)
    z = np.zeros(N)
    
    # Combine the x, y coordinates into an (N, 2) array
    points = np.vstack((x, y, z)).T
    
    # Step 3: Shift the points by the center point c
    points += np.array(c)
    
    return points