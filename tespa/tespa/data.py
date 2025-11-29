
import numpy as np


class DataLoader():
    def __init__(self, dx, Nx, Ny, Nz):
        self.dx = dx
        self.x = np.linspace(0, dx*Nx, Nx)
        self.y = np.linspace(0, dx*Ny, Ny)
        self.z = np.linspace(0, dx*Nz, Nz)

        self.phisp = np.zeros((Nx,Ny,Nz))
    
        self.Ex = np.zeros((Nx,Ny,Nz))
        self.Ey = np.zeros((Nx,Ny,Nz))
        self.Ez = np.zeros((Nx,Ny,Nz))

        self.Bx = np.zeros((Nx,Ny,Nz))
        self.By = np.zeros((Nx,Ny,Nz))
        self.Bz = np.zeros((Nx,Ny,Nz))

        self.jx = np.zeros((Nx,Ny,Nz))
        self.jy = np.zeros((Nx,Ny,Nz))
        self.jz = np.zeros((Nx,Ny,Nz))

        self.jex = np.zeros((Nx,Ny,Nz))
        self.jey = np.zeros((Nx,Ny,Nz))
        self.jez = np.zeros((Nx,Ny,Nz))

        self.jix = np.zeros((Nx,Ny,Nz))
        self.jiy = np.zeros((Nx,Ny,Nz))
        self.jiz = np.zeros((Nx,Ny,Nz))

        self.nde = np.zeros((Nx,Ny,Nz))
        self.ndi = np.zeros((Nx,Ny,Nz))

        self.ndph = np.zeros((Nx,Ny,Nz))

        print("Initialized box with size ", Nx*dx, Ny*dx, Nz*dx, "meters")

    def load_phisp_compute_E(self, phisp):
        self.phisp = phisp
        dphidx, dphidy, dphidz = np.gradient(self.phisp, self.dx, self.dx, self.dx, edge_order=2)
        self.Ex = -dphidx
        self.Ey = -dphidy
        self.Ez = -dphidz
        #self.Ex, self.Ey, self.Ez = np.gradient(-self.phisp, self.dx)

    
    def load_from_npz(self, filepath, verbose: bool = False):
        data = np.load(filepath)

        self.phisp = data['phisp']

        self.Ex = data['Ex']
        self.Ey = data['Ey']
        self.Ez = data['Ez']

        self.Bx = data['Bx']
        self.By = data['By']
        self.Bz = data['Bz']

        self.jx = data['jx']
        self.jy = data['jy']
        self.jz = data['jz']

        self.jex = data['jex']
        self.jey = data['jey']
        self.jez = data['jez']

        self.jix = data['jix']
        self.jiy = data['jiy']
        self.jiz = data['jiz']

        self.nde = data['nde']
        self.ndi = data['ndi']

        self.ndph = data['ndph']

        if verbose:
            print(f"Successfully loaded data from {filepath}")