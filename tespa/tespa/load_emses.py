
import numpy as np
import h5py
from scipy.interpolate import interpn, RegularGridInterpolator

coeff_freq = 8.339102E-08
V_coeff = 5.109989e-03
E_coeff = 5.109989e-01


class LoadEMSES():
    def __init__(self, path, dx, entry = '0001', offset=True, reshape=None, V_coeff=V_coeff, E_coeff=E_coeff, coeff_freq=coeff_freq, nd_coeff=1.0, B_coeff=1.0, j_coeff=1.0):
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
        
        self.nd_coeff = nd_coeff if nd_coeff is not None else 1.0
        self.V_coeff = V_coeff if V_coeff is not None else 1.0
        self.E_coeff = E_coeff if E_coeff is not None else 1.0
        self.coeff_freq = coeff_freq if coeff_freq is not None else 1.0
        self.B_coeff = B_coeff if B_coeff is not None else 1.0
        self.j_coeff = j_coeff if j_coeff is not None else 1.0

        if reshape is None:
            self.reshape = (0, 1, 2)
        else:
            self.reshape = reshape
            print("Reshaping data to ", self.reshape)

        self.path = path
        self.entry = entry
        self.dx = dx
        self.load_all_data()
        if offset:
            self.offset_fields(dx*0.5)
        

    def load_all_data(self):
        try:
            self.load_E()
            self.load_dim()
            self.load_phisp()
        except Exception as e:
            print("Error loading E data", e)
        try:
            self.load_nd()
        except Exception as e:
            print("Error loading density data", e)

        try:
            self.load_j()
        except Exception as e:
            print("Error loading current density data", e)
        try:
            self.load_B()
        except Exception as e:
            print("Error loading B data", e)

    def offset_fields(self, offset, E=True, B=False, phisp=True):
        
        if E:
            print("Offsetting E fields by ", offset)
            try:
                self.Ex = self.offset(self.Ex.copy(), offset)
                self.Ey = self.offset(self.Ey.copy(), offset)
                self.Ez = self.offset(self.Ez.copy(), offset)
            except Exception as e:
                print("Error offsetting E data", e)
        if B:
            print("Offsetting B fields by ", offset)
            try:
                self.Bx = self.offset(self.Bx, offset)
                self.By = self.offset(self.By, offset)
                self.Bz = self.offset(self.Bz, offset)
            except Exception as e:
                print("Error offsetting B data", e)
        if phisp:
            print("Offsetting phisp fields by ", offset)
            try:
                self.phisp = self.offset(self.phisp.copy(), offset)
            except Exception as e:
                print("Error offsetting phisp data", e)

    def offset(self, data, offset, method="linear"):
        # Define grid spacing
        dx, dy, dz = offset, offset, offset  # Grid spacings in each dimension

        # Original grid with offsets in each dimension
        x_original = self.x + dx  # Offset along x-axis
        y_original = self.y + dy  # Offset along y-axis
        z_original = self.z + dz  # Offset along z-axis

        # Create the interpolator with RegularGridInterpolator
        interp_func = RegularGridInterpolator((self.x, self.y, self.z), data, method=method, bounds_error=False, fill_value=None)

        # Define the points where we want to evaluate the interpolated data
        X_corr, Y_corr, Z_corr = np.meshgrid(x_original, y_original, z_original, indexing='ij')
        points = np.array([X_corr.flatten(), Y_corr.flatten(), Z_corr.flatten()]).T

        # Interpolate data to the corrected grid
        data_corrected = interp_func(points)

        # Reshape the corrected data back to the original grid shape
        data_corrected = data_corrected.reshape(len(self.x), len(self.y), len(self.z))

        return data_corrected
    

    def offset_interpn(self, data, offset, method = "linear"):
        # Define grid spacing
        dx, dy, dz = offset, offset, offset  # Grid spacings in each dimension

        # Original grid with offsets in each dimension
        x_original = self.x + dx  # Offset along x-axis
        y_original = self.y + dy  # Offset along y-axis
        z_original = self.z + dz  # Offset along z-axis

        # Create a meshgrid for the original coordinates
        grid_points = np.meshgrid(x_original, y_original, z_original, indexing='ij')

        # Define the points where we want to evaluate the interpolated data
        points = np.array([grid.flatten() for grid in grid_points]).T

        # Interpolate data to the corrected grid using interpn
        data_corrected = interpn((self.x, self.y, self.z), data, points, method=method, bounds_error=False, fill_value=None)

        # Reshape the corrected data back to the original grid shape
        data_corrected = data_corrected.reshape(len(self.x), len(self.y), len(self.z))

        return data_corrected


    def load_phisp(self, entry="0001"):
        """Load phisp data from a file.

        Args:
            entry (str, optional): The entry to load. Defaults to "0001".
        """
        Z = h5py.File(self.path + "/phisp00_0000.h5", 'r')["phisp"][self.entry]
        Z = Z[:, :, :] * self.V_coeff
        self.phisp = np.transpose(Z, self.reshape)


    


    def load_E(self, entry="0001"):
        """Load electric field data from a file.

        Args:
            entry (str, optional): The entry to load. Defaults to "0001".
        """
        Ex = h5py.File(self.path + "/ex00_0000.h5", 'r')["ex"][self.entry]
        Ey = h5py.File(self.path + "/ey00_0000.h5", 'r')["ey"][self.entry]
        Ez = h5py.File(self.path + "/ez00_0000.h5", 'r')["ez"][self.entry]

        self.Ex = np.transpose(Ex[:, :, :], self.reshape) * self.E_coeff
        self.Ey = np.transpose(Ey[:, :, :], self.reshape) * self.E_coeff
        self.Ez = np.transpose(Ez[:, :, :], self.reshape) * self.E_coeff

    def load_nd(self,entry = "0001"):
        #For electrons
        if entry is None:
            entry = self.entry
        
        nde_path = self.path + "/nd1p00_0000.h5"
        ndi_path = self.path + "/nd2p00_0000.h5"

        try:
            nde = h5py.File(nde_path, 'r')['nd1p'][self.entry]
            ndi = h5py.File(ndi_path, 'r')['nd2p'][self.entry]
        
        except OSError as e:
            print(e, "filename: ", self.path)

        self.nde = np.transpose(nde[:, :, :], self.reshape) * self.nd_coeff
        self.ndi = np.transpose(ndi[:, :, :], self.reshape) * self.nd_coeff


        self.nd = self.nde-self.ndi
    
    def load_nd_phe(self, filename, entry="0001"):
        ndph_path = self.path + "/" + filename
        try:
            ndph = h5py.File(ndph_path, 'r')['nd3p'][self.entry]
            print("Successfully loaded photoelectron density from ", ndph_path)
        except OSError as e:
            print(e, "filename: ", self.path)
        self.ndph = np.transpose(ndph[:, :, :], self.reshape) * self.nd_coeff


    def load_B(self, entry="0001"):
        """Load magnetic field data from a file.

        Args:
            entry (str, optional): The entry to load. Defaults to "0001".
        """
        Bx = h5py.File(self.path + "/bx00_0000.h5", 'r')["bx"][self.entry]
        By = h5py.File(self.path + "/by00_0000.h5", 'r')["by"][self.entry]
        Bz = h5py.File(self.path + "/bz00_0000.h5", 'r')["bz"][self.entry]

        self.Bx = np.transpose(Bx[:, :, :], self.reshape) * self.B_coeff
        self.By = np.transpose(By[:, :, :], self.reshape) * self.B_coeff
        self.Bz = np.transpose(Bz[:, :, :], self.reshape) * self.B_coeff


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

        jx = np.transpose(jx[:, :, :], self.reshape)
        jy = np.transpose(jy[:, :, :], self.reshape)
        jz = np.transpose(jz[:, :, :], self.reshape)

        self.jx = jx[:, :, :] * self.j_coeff
        self.jy = jy[:, :, :] * self.j_coeff
        self.jz = jz[:, :, :] * self.j_coeff

        jex = h5py.File(self.path + "/j1x00_0000.h5", 'r')["j1x"][self.entry]
        jey = h5py.File(self.path + "/j1y00_0000.h5", 'r')["j1y"][self.entry]
        jez = h5py.File(self.path + "/j1z00_0000.h5", 'r')["j1z"][self.entry]

        jex = np.transpose(jex[:,:,:], self.reshape)
        jey = np.transpose(jey[:,:,:], self.reshape)
        jez = np.transpose(jez[:,:,:], self.reshape)


        self.jex = jex[:, :, :] * self.j_coeff
        self.jey = jey[:, :, :] * self.j_coeff
        self.jez = jez[:, :, :] * self.j_coeff

        jix = h5py.File(self.path + "/j2x00_0000.h5", 'r')["j2x"][self.entry]
        jiy = h5py.File(self.path + "/j2y00_0000.h5", 'r')["j2y"][self.entry]
        jiz = h5py.File(self.path + "/j2z00_0000.h5", 'r')["j2z"][self.entry]

        jix = np.transpose(jix[:,:,:], self.reshape)
        jiy = np.transpose(jiy[:,:,:], self.reshape)
        jiz = np.transpose(jiz[:,:,:], self.reshape)

        self.jix = jix[:, :, :] * self.j_coeff
        self.jiy = jiy[:, :, :] * self.j_coeff
        self.jiz = jiz[:, :, :] * self.j_coeff