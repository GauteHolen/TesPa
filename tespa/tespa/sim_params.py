


class SimParams:
    def __init__(self, Nx : int,
                       Ny : int,
                       Nz : int,
                       dx : float,
                       dt : float,
                       Nt : int,
                       x_boundary = "periodic",
                       y_boundary = "periodic",
                       z_boundary = "periodic",
                       sub_boundary = None,
                       verbose : bool = False) -> None:
        """Initialize simulation parameters.

        Args:
            Nx (int): Number of grid points in x-direction.
            Ny (int): Number of grid points in y-direction.
            Nz (int): Number of grid points in z-direction.
            dx (float): Grid spacing in meters.
            dt (float): Time step in seconds.
            total_time (float): Total simulation time in seconds.
        """
        self.Nx = Nx
        self.Ny = Ny
        self.Nz = Nz
        self.dx = dx
        self.dt = dt
        self.Nt = Nt
        self.x_boundary = x_boundary
        self.y_boundary = y_boundary
        self.z_boundary = z_boundary
        self.sub_boundary = sub_boundary

        if verbose:
            print(self)

    def __str__(self):
        s = f"SimParams: \n"
        s += f"Nx={self.Nx} \n"
        s += f"Ny={self.Ny} \n"
        s += f"Nz={self.Nz} \n"
        s += f"dx={self.dx} m \n"
        s += f"dt={self.dt} s \n"
        s += f"Nt={self.Nt} \n"
        s += f"x_boundary={self.x_boundary} \n"
        s += f"y_boundary={self.y_boundary} \n"
        s += f"z_boundary={self.z_boundary} \n"
        s += f"sub_boundary={self.sub_boundary} \n"
        return s