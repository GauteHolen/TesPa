import numpy as np

geometries = ["sphere", "disc"]

def sample_uniform_sphere(Np, xc, yc, zc, Rs):
    # Random directions
    vec = np.random.normal(size=(3, Np))
    vec /= np.linalg.norm(vec, axis=0)

    # Uniform radius
    r = Rs * np.cbrt(np.random.rand(Np))

    pos = np.empty((3, Np))
    pos[0] = xc + r * vec[0]
    pos[1] = yc + r * vec[1]
    pos[2] = zc + r * vec[2]

    return pos

def sample_disk_xyzN(N, xc, yc, zc, R):
    theta = np.random.uniform(0.0, 2*np.pi, N)
    r = R * np.sqrt(np.random.uniform(0.0, 1.0, N))

    x = xc + r * np.cos(theta)
    y = yc + r * np.sin(theta)
    z = np.full(N, zc)

    return np.array([x, y, z])   # shape (3, N)


class DustParams():
    def __init__(self, Nd : int, 
                 density : float,
                 init_geometry : str = "sphere",
                 r0 : tuple = (0.1,0.1,0.1),
                 Rs : float = 0.1,
                 r0s : np.ndarray = None, 
                 v0s : np.ndarray = None, 
                 rdust=1e-6, 
                 r_dust_array=None, 
                 q : float = 0.0, 
                 qs : np.ndarray = None, 
                 ms : np.ndarray = None,
                 verbose: bool = False) -> None:
        """Data class holding 

        Args:
            Nd (int): Number of dust particles
            density (float): density of dust particles in kg/m3
            init_geometry (str, optional): shape of dust particles. Defaults to "sphere".
            r0 (tuple, optional): center position of dust particle initialization structure in m. Defaults to (0.1,0.1,0.1).
            Rs (float, optional): characteristic size of dust particle initialization structure in m. Defaults to 0.1.
            r0s (np.ndarray): inital positions of dust particles in m
            v0s (np.ndarray, optional): initial velocities of dust particles in m/s. Defaults to None.
            rdust (float, optional): radius of dust particles in m. Defaults to 1e-6.
            r_dust_array (np.ndarray, optional): array of dust particle radii in m. Defaults to None.
            q (float, optional): charge of dust particles in Coulombs. Defaults to 0.0.
            qs (np.ndarray, optional): array of dust particle charges in Coulombs. Defaults to None.
            ms (np.ndarray, optional): array of dust particle masses in kg. Defaults to None.
        """
        self.Nd = Nd
        self.density = density


        if init_geometry not in geometries:
            raise ValueError(f"init_geometry {init_geometry} not recognized. Available geometries: {geometries}")
        elif r0s is not None:
            self.init_geometry = "custom"
        else:
            self.init_geometry = init_geometry

        if r0s is not None:
            self.r0s = r0s
        else:
            if init_geometry == "sphere":
                self.r0s = sample_uniform_sphere(Nd, r0[0], r0[1], r0[2], Rs)  
            elif init_geometry == "disc":
                self.r0s = sample_disk_xyzN(Nd, r0[0], r0[1], r0[2], Rs)  

        if r_dust_array is None:
            self.rdust = rdust
            self.r_dust_array = np.ones(Nd) * rdust
        else:
            self.rdust = rdust
            self.r_dust_array = r_dust_array

        if v0s is None:
            self.v0s = np.zeros_like(r0s)
        
        if ms is None:
            self.ms = (4/3)*np.pi*(self.r_dust_array**3)*density
        else:
            self.ms = ms
        
        if qs is None:
            self.qs = np.ones(Nd) * q
        else:
            self.qs = qs

        if verbose:
            print(self)

    def __str__(self):

        s = f"Dust Params: "
        s += f"N particles: {self.Nd} \n"
        s += f"density={self.density} kg/m3 \n"
        s += f"average rdust={np.mean(self.r_dust_array):.2e} m \n"
        s += f"average mass={np.mean(self.ms):.2e} kg \n"
        s += f"Initial geometry: {self.init_geometry} \n"
        s += f"average initial velocity: {np.mean(self.v0s)} m/s \n"


        return s