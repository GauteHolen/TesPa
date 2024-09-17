
import numpy as np
import h5py

coeff_freq = 8.339102E-08


class LoadEMSES():
    def __init__(self, path, dx, entry = '0001'):
        """
        Initializes an instance of the IVdata class.

        Args:
            name (str): The name of the IVdata object.
            path (str): The path to the IV data file.
            Ap (float, optional): The surface area of the probe. Defaults to None.
            Vbias (float, optional): The value of Vbias. Defaults to None.
            Ab (float, optional): The surface area of the body. Defaults to None.
            epcol (int, optional): The column index for the electron current data. Defaults to None.
            ipcol (int, optional): The column index for the ion current data. Defaults to None.
            ebcol (int, optional): The column index for the electron body current data. Defaults to None.
            ibcol (int, optional): The column index for the ion body current data. Defaults to None.
            bodycol (int, optional): The column index for the body potential data. Defaults to None.
            probecol (int, optional): The column index for the probe potential data. Defaults to None.
            vdrift (float, optional): The drift velocity value. Defaults to None.
            B (float, optional): The magnetic field value. Defaults to None.
            potential (float, optional): The potential value. Defaults to None.
            vdir (str, optional): The direction of the drift velocity. Defaults to None.
            Bdir (str, optional): The direction of the magnetic field. Defaults to None.
            flowBorient (str, optional): The orientation of the flow and magnetic field. Defaults to None.
        """

        print("Loading data from ", path)

        self.path = path
        self.entry = entry
        self.dx = dx
        try:
            self.load_E()
            self.load_dim()
        except Exception as e:
            print("Error loading E data", e)

        try:
            self.load_B()
        except Exception as e:
            print("Error loading B data", e)


    def load_E(self, entry="0001"):
        """Load electric field data from a file.

        Args:
            entry (str, optional): The entry to load. Defaults to "0001".
        """
        Ex = h5py.File(self.path + "/ex00_0000.h5", 'r')["ex"][self.entry]
        Ey = h5py.File(self.path + "/ey00_0000.h5", 'r')["ey"][self.entry]
        Ez = h5py.File(self.path + "/ez00_0000.h5", 'r')["ez"][self.entry]

        self.Ex = Ex[:, :, :]
        self.Ey = Ey[:, :, :]
        self.Ez = Ez[:, :, :]
    
    def load_B(self, entry="0001"):
        """Load magnetic field data from a file.

        Args:
            entry (str, optional): The entry to load. Defaults to "0001".
        """
        Bx = h5py.File(self.path + "/bx00_0000.h5", 'r')["bx"][self.entry]
        By = h5py.File(self.path + "/by00_0000.h5", 'r')["by"][self.entry]
        Bz = h5py.File(self.path + "/bz00_0000.h5", 'r')["bz"][self.entry]

        self.Bx = Bx[:, :, :]
        self.By = By[:, :, :]
        self.Bz = Bz[:, :, :]


    def load_dim(self):
        """
        Load the dimensions of the grid.

        Args:
            dx (float): The grid spacing.
        """
        self.x = np.arange(0, self.Ex.shape[0]) * self.dx
        self.y = np.arange(0, self.Ex.shape[1]) * self.dx
        self.z = np.arange(0, self.Ex.shape[2]) * self.dx

    def load_j(self, entry="0001"):
        """Load current density data from a file.

        Args:
            entry (str, optional): The entry to load. Defaults to "0001".
        """
        jx = h5py.File(self.path + "/jx00_0000.h5", 'r')["jx"][self.entry]
        jy = h5py.File(self.path + "/jy00_0000.h5", 'r')["jy"][self.entry]
        jz = h5py.File(self.path + "/jz00_0000.h5", 'r')["jz"][self.entry]

        self.jx = jx[:, :, :]
        self.jy = jy[:, :, :]
        self.jz = jz[:, :, :]

        jex = h5py.File(self.path + "/j1x00_0000.h5", 'r')["j1x"][self.entry]
        jey = h5py.File(self.path + "/j1y00_0000.h5", 'r')["j1y"][self.entry]
        jez = h5py.File(self.path + "/j1z00_0000.h5", 'r')["j1z"][self.entry]

        self.jex = jex[:, :, :]
        self.jey = jey[:, :, :]
        self.jez = jez[:, :, :]

        jix = h5py.File(self.path + "/j2x00_0000.h5", 'r')["j2x"][self.entry]
        jiy = h5py.File(self.path + "/j2y00_0000.h5", 'r')["j2y"][self.entry]
        jiz = h5py.File(self.path + "/j2z00_0000.h5", 'r')["j2z"][self.entry]

        self.jix = jix[:, :, :]
        self.jiy = jiy[:, :, :]
        self.jiz = jiz[:, :, :]