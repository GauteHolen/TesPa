"""
This module contains functions for calculating electron and ion currents in a cylinder and a sphere.

Author: Gaute Holen
Date: 27.06.2024
"""

import numpy as np
from numba import njit, prange, jit
from .load_emses import LoadEMSES
import h5py
from .interpolate_cubic import cubic_interp_3d_jit
from .run_boris_dust import _sim_boris_dust
from.get_velocities import get_velocity_field
from .thermo_utils import debyelength_e
from .data import DataLoader
from .read_config import read_dust_config

q = 1.602176634e-19
qe = -q



class TesPa():
    def __init__(self, dx=1, Nx=100, Ny=100, Nz=100):
        self.data = DataLoader(dx, Nx, Ny, Nz)
        self.emses = False
        
    
    def load_emses(self,dirname,dx, offset=True, entry='0001', reshape = None,V_coeff=None, E_coeff=None, coeff_freq=None, nd_coeff=None, B_coeff=None, j_coeff=None):
        """Load emses data from a given directory. Uses the LoadEMSES class.

        Args:
            dirname (str): path to directory
            dx (float): gridsize in meters
        """

        self.dir = dirname
        self.data = LoadEMSES(self.dir,dx,offset=offset,entry=entry, reshape=reshape, V_coeff=V_coeff, E_coeff=E_coeff, coeff_freq=coeff_freq, nd_coeff=nd_coeff, B_coeff=B_coeff, j_coeff=j_coeff)
        self.collision_arr = np.full(self.data.Ex.shape, False)
        self.emses = True



    def __str__(self):
        """Print some info about the simulation

        Returns:
            str: some info
        """
        str = f"Data loaded from {self.dir} \n"
        str += f"Dimensions in meters (x,y,z): {self.data.x[-1], self.data.y[-1], self.data.z[-1]} with dx = {self.data.dx}"
        return str
    
    def set_B(self,B_mag,dir='z'):
        """Set the magnetic field manually for the simulation

        Args:
            B_mag (float): Magnetic field strength in Tesla
            dir (str, optional): Direction of the magnetic field. Defaults to 'z'. 'x' and 'y' are also valid directions.
        """
        self.Bdir = dir
        B0 = np.zeros(self.data.Ex.shape)
        B = np.ones(self.data.Ex.shape) * B_mag

        self.data.Bx = B0.copy()
        self.data.By = B0.copy()
        self.data.Bz = B0.copy()

        if dir == 'z':
            self.data.Bz = B.copy()
            
        elif dir == 'x':
            self.data.Bx = B.copy()
            
        elif dir == 'y':
            self.data.By = B.copy()

    def set_object_boundary(self,collision_arr):
        self.collision_arr = collision_arr

    def set_E_arrays(self,Ex : np.ndarray, Ey : np.ndarray, Ez : np.ndarray):
        """Set the electric field manually for the simulation

        Args:
            E (np.array): Electric field array with same shape as the simulation grid
        """
        self.data.Ex = Ex
        self.data.Ey = Ey
        self.data.Ez = Ez

    def set_relative_density_arrays(self,nde : np.ndarray, ndi : np.ndarray):
        self.data.nde = nde
        self.data.ndi = ndi

            
    def set_E(self,E_mag : float,dir='x',gradient=(1,1), gradient_extent=1.0):
        """Set the electric field manually for the simulation

        Args:
            E_mag (float): Electric field strength in V/m
            dir (str, optional): Direction of the magnetic field. Defaults to 'z'. 'x' and 'y' are also valid directions.
        """
        self.Bdir = dir
        E0 = np.zeros(self.data.Ex.shape)
        E = np.ones(self.data.Ex.shape) * E_mag

        grad_mag = np.linspace(gradient[0],gradient[1],int(self.data.Ex.shape[0]*gradient_extent))
        grad = np.zeros(self.data.Ex.shape[0])
        for i in range(len(grad_mag)):
            grad[i] = grad_mag[i]

        for i in range(len(grad)):
            if dir == 'x':
                E[i,:,:] = E[i,:,:] * grad[i]
            elif dir == 'y':
                E[:,i,:] = E[:,i,:] * grad[i]
            elif dir == 'z':
                E[:,:,i] = E[:,:,i] * grad[i]

        self.data.Ex = E0.copy()
        self.data.Ey = E0.copy()
        self.data.Ez = E0.copy()

        if dir == 'z':
            self.data.Ez = E.copy()
            
        elif dir == 'x':
            self.data.Ex = E.copy()
            
        elif dir == 'y':
            self.data.Ey = E.copy()

    def run(self, r0,v0,m,q,dt0=1e-10, Nt=1e5):
        """Run the simulation with the given parameters

        Args:
            r0 (np.array([r_x,r_y,r_z])): initial position
            v0 (np.array([v_x,v_y,v_z])): Initial velocity
            m (float): particle mass
            q (float): particle charge
            dt0 (float, optional): Initial timestep for adaptive timestep. Defaults to 1e-10.
            Nt (int, optional): Number of time steps. Defaults to 1e5.
        """
        self.m = m
        self.q = q
        Ex = self.data.Ex
        Ey = self.data.Ey
        Ez = self.data.Ez
        Bx = self.data.Bx
        By = self.data.By
        Bz = self.data.Bz
        x = self.data.x
        y = self.data.y
        z = self.data.z
        collision_arr = self.collision_arr
        dr = self.data.dx
        


        self.r,self.v,self.e,self.t,self.emags,self.vmags,self.r_avg = _sim_boris(r0,x,y,z,m,q,Ex,Ey,Ez,Bx,By,Bz,v0,dr,dt0,int(Nt),collision_arr)

    def load_dust_config(self, path: str, verbose: bool = False):
        self.dust_params, self.plasma_params, self.sim_params = read_dust_config(path, verbose=verbose)

    def run_dust(self, 
                 r0s : np.ndarray,
                 v0s : np.ndarray,
                 ms : np.ndarray,
                 qs : np.ndarray,
                 Te : float,
                 Ti : float,
                 ne0 : float,
                 ni0 : float,
                 mass_ratio : float =1000,
                 dt0 : float =1e-3,
                 Nt : int =1e4,
                 aG : float =0.0,
                 rdust : float =0.003, 
                 dt_precompute : float =1e-5, 
                 precompute_charge : int =1000, 
                 recompute_charge : int =100, 
                 sub_boundary : tuple = None, 
                 photoemission_current_density : float = None, 
                 r_dust_array : np.ndarray = None, 
                 nphe0 : float = None,
                 Teph : float = None,
                 beta_epstein : float = 0.0):
        """Run the dust simulation

        Args:
            r0s (np.ndarray): initial positions
            v0s (np.ndarray): initial velocities
            ms (np.ndarray): masses
            qs (np.ndarray): initial charges if static charge, else does nothing
            Te (float): electron temperature
            Ti (float): ion temperature
            ne0 (float): electron density
            ni0 (float): ion density
            dt0 (float, optional): timestep. Defaults to 1e-3.
            Nt (int, optional): number of time steps. Defaults to 1e4.
            aG (float, optional): Acceleration due to gravity in z direction. Defaults to 0.0.
            rdust (float, optional): dust radius if mono-disperse. Defaults to 0.003.
            dt_precompute (float, optional): precompute timestep. Defaults to 1e-5.
            precompute_charge (int, optional): precompute charge interval. Defaults to 1000.
            recompute_charge (int, optional): recompute charge interval. Defaults to 100.
            sub_boundary (tuple, optional): sub-boundary conditions. Defaults to None.
            photoemission_current_density (float, optional): photoemission current density. Defaults to None.
            r_dust_array (np.ndarray, optional): dust radius array if poly-disperse. Defaults to None.
            nphe0 (float, optional): photoemission electron density. Defaults to None.
            Teph (float, optional): photoemission electron temperature. Defaults to None.
        """
        Nt = int(Nt)
        Ex = self.data.Ex.copy()
        Ey = self.data.Ey.copy()
        Ez = self.data.Ez.copy()
        Bx = self.data.Bx.copy()
        By = self.data.By.copy()
        Bz = self.data.Bz.copy()
        Jex = self.data.jex.copy()
        Jey = self.data.jey.copy()
        Jez = self.data.jez.copy()
        Jix = self.data.jix.copy()
        Jiy = self.data.jiy.copy()
        Jiz = self.data.jiz.copy()
        x = self.data.x.copy()
        y = self.data.y.copy()
        z = self.data.z.copy()
        dr = self.data.dx
        #vex,vey,vez = get_velocity_field(Jex,Jey,Jez,self.data.nde,-q)
        #vix,viy,viz = get_velocity_field(Jix,Jiy,Jiz,self.data.ndi,q)

        Np = ms.shape[0]


        if self.emses:
            nde = self.data.nde.copy() * ne0
            ndi = self.data.ndi.copy() * ni0
        else:
            nde = self.data.nde.copy()
            ndi = self.data.ndi.copy()

        if nphe0 is None or Teph is None:
            print("No photoemission parameters given, disabling photoemission")
            ndphe = nde * 0
            nphe0 = 0.0
            Teph = 0.0
        elif self.emses:
            ndphe = self.data.ndph.copy() * nphe0
        else:
            Teph = 0.0
            

        lambda_D = debyelength_e(Te,ne0,qe)
        kd = 1.0 / lambda_D
        print("Debye length: ", lambda_D, " m, kd = ", kd)

        if r_dust_array is not None:
            a = r_dust_array
            
        else:
            a = np.ones(Np) * rdust
        self.a = a


        self.r,self.v,self.t,self.Zd,self.phi = _sim_boris_dust(r0s,
                                               x,y,z,
                                               ms,qs,a,
                                               Ex,Ey,Ez,
                                               Bx,By,Bz,
                                               nde,ndi,ndphe,
                                               Te,Ti,Teph,
                                               Jex,Jey,Jez,
                                               Jix,Jiy,Jiz,
                                               #vex,vey,vez,
                                               #vix,viy,viz,
                                               v0s,
                                               dr,dt0,Nt,
                                               kd = kd,
                                               aG=aG,
                                               dt_precompute=dt_precompute,
                                               precompute_charge=precompute_charge, recompute_charge=recompute_charge,
                                               sub_boundary=sub_boundary,
                                               photoemission_current_density=photoemission_current_density, 
                                               beta_epstein = beta_epstein,
                                               massratio = mass_ratio)


    def run_simple(self, r0,v0,m,q,dt0=1e-10, Nt=1e5, interval = None):
        """Run the simulation with the given parameters

        Args:
            r0 (np.array([r_x,r_y,r_z])): initial position
            v0 (np.array([v_x,v_y,v_z])): Initial velocity
            m (float): particle mass
            q (float): particle charge
            dt0 (float, optional): Initial timestep for adaptive timestep. Defaults to 1e-10.
            Nt (int, optional): Number of time steps. Defaults to 1e5.
        """

        

        Ex = self.data.Ex
        Ey = self.data.Ey
        Ez = self.data.Ez
        Bx = self.data.Bx
        By = self.data.By
        Bz = self.data.Bz
        x = self.data.x
        y = self.data.y
        z = self.data.z
        collision_arr = self.collision_arr
        dr = self.data.dx

        if interval == None:
            v_mag = np.sqrt(v0[0]**2 + v0[1]**2 + v0[2]**2)
            if v_mag > 0:
                interval = int(dr / dt0 / v_mag) # How many timesteps we need to move one gridcell
            else:
                interval = 1
            interval = max(1,interval) 
            print("Timestep index of data storage: ", interval)        

        r0 = r0.reshape(3,1)
        v0 = v0.reshape(3,1)

        print("Compiling...")
        r,v,t,finalstate,finalindex = _sim_boris_simple(r0,x,y,z,m,q,Ex,Ey,Ez,Bx,By,Bz,v0,collision_arr,dr,dt0,int(Nt))
        self.r = r[:,::interval,0]
        self.v = v[:,::interval,0]
        self.t = t[::interval,0]
        self.finalstate = finalstate[0] 
        self.finalindex = finalindex[0]

    def run_many(self, r0,v0,m,q,dt0=1e-10, Nt=1e5, interval = 1, backwards = False):
        """Run the simulation with the given parameters

        Args:
            r0 (np.array([r_x,r_y,r_z])): initial position
            v0 (np.array([v_x,v_y,v_z])): Initial velocity
            m (float): particle mass
            q (float): particle charge
            dt0 (float, optional): Initial timestep for adaptive timestep. Defaults to 1e-10.
            Nt (int, optional): Number of time steps. Defaults to 1e5.
        """
        Ex = self.data.Ex
        Ey = self.data.Ey
        Ez = self.data.Ez
        Bx = self.data.Bx
        By = self.data.By
        Bz = self.data.Bz
        self.dt = dt0
        x = self.data.x
        y = self.data.y
        z = self.data.z
        collision_arr = self.collision_arr
        dr = self.data.dx
        self.q = q
        self.m = m
        self.dump_period = interval


        print("Compiling...")
        rs,vs,ts, finalstate, finalindex = _sim_boris_simple(r0,x,y,z,m,q,Ex,Ey,Ez,Bx,By,Bz,v0,collision_arr,dr,dt0,int(Nt), backwards=backwards)

        """# Trim the data
        trimmed_data = []

        for p in range(Np):
            valid_steps = valid_steps_per_particle[p]
            trimmed_data.append(simulation_data[:, :valid_steps, p])  # Trim the time axis for each particle
            # Convert the list of trimmed arrays back into a NumPy array (or keep as list if particles have varying Nt)
            trimmed_data = np.array(trimmed_data, dtype=object)

            # Example usage: Accessing the time steps for particle 0
            print(trimmed_data[0].shape)  # Should be (3, 800)
        """

        # Remove positions after particle escapes or is collected
        for p in range(rs.shape[2]):
            if finalstate[p] != 0.0:
                print("Particle", p, "escaped or was collected")
                rs[:,finalindex[p]+1:,p] = np.nan
                vs[:,finalindex[p]+1:,p] = np.nan
                ts[finalindex[p]+1:,p] = np.nan


        
        self.r = rs[:,::interval,:]
        self.v = vs[:,::interval,:]
        self.t = ts[::interval,:]
        self.finalstate = finalstate
        self.finalindex = np.floor(finalindex/interval).astype(np.int64)


    def store_fields_npz(self, path: str):
        """Store the field data in a npz file for later use.

        Args:
            path (str): Path to the npz file.
        """
        if not path.endswith('.npz'):
            path = path + '.npz'
        np.savez_compressed(path,
                            Ex=self.data.Ex,
                            Ey=self.data.Ey,
                            Ez=self.data.Ez,
                            Bx=self.data.Bx,
                            By=self.data.By,
                            Bz=self.data.Bz,
                            jex=self.data.jex,
                            jey=self.data.jey,
                            jez=self.data.jez,
                            jix=self.data.jix,
                            jiy=self.data.jiy,
                            jiz=self.data.jiz,
                            jx=self.data.jx,
                            jy=self.data.jy,
                            jz=self.data.jz,
                            nde=self.data.nde,
                            ndi=self.data.ndi,
                            ndph=self.data.ndph,
                            x=self.data.x,
                            y=self.data.y,
                            z=self.data.z,
                            phisp=self.data.phisp,
                            )
    
    



    def write_h5(self, path: str):
        """Write a particle to a file.

        Args:
            file (file): The file to write to.
            r (np.array): The position of the particle.
            v (np.array): The velocity of the particle.
            q (float): The charge of the particle.
            m (float): The mass of the particle.
        """

        if not path.endswith('.h5'):
            path = path + '.h5'

        print("Writing to file: ", path)


        with h5py.File(path, "w") as file:

            file.attrs['description'] = 'Test particle simulation data using TesPa package. Contains position, velocity, time, electric field, magnetic field, number density, and potential data.'
        
            file.create_dataset("positions", data=self.r)
            file['positions'].attrs['description'] = 'Particle positions data: [dim, t, i] where dim is the dimension of the simulation, t is the time step, and i is the particle index.'
            file['positions'].attrs['units'] = 'm'
            file['positions'].attrs['dim'] = "x, y, z"
            
            file.create_dataset("finalstate", data=self.finalstate)
            file['finalstate'].attrs['description'] = 'Final state of the simulation: [i] where i is the particle index. 0 if the particle is still in the simulation, 1 if the particle is collected and -1 if the particle has left the simulation.'

            file.create_dataset("finalindex", data=self.finalindex)
            file['finalindex'].attrs['description'] = 'Final time index where the particle is in the simulation: [i] where i is the particle index.'


            file.create_dataset("velocities", data=self.v)
            file['velocities'].attrs['description'] = 'Particle velocities data: [dim, t, i] where dim is the dimension of the simulation, t is the time step, and i is the particle index.'
            file['velocities'].attrs['units'] = 'm/s'
            file['velocities'].attrs['dim'] = "vx, vy, vz"

            file.create_dataset("time", data=self.t)
            file['time'].attrs['description'] = 'Time data: [t,i] where t is the time step and i is the particle index.'

            file.attrs['dt'] = self.dt
            file.attrs['dr'] = self.data.dx
            file.attrs['q'] = self.q
            file.attrs['m'] = self.m
            file.attrs['Nt'] = self.r.shape[1]
            file.attrs['Np'] = self.r.shape[2]

            file.create_dataset("object", data=self.collision_arr)
            file['object'].attrs['description'] = 'Collision object array: True is object is present, False if object is not present.'
            file['object'].attrs['dim'] = "x, y, z"


            file.create_dataset("phisp", data=self.data.phisp)
            file['phisp'].attrs['description'] = 'Scalar potential data: [x, y, z] where x, y, z are the spatial dimensions of the simulation.'
            file['phisp'].attrs['units'] = 'V'
            file['phisp'].attrs['dim'] = "x, y, z"

            file.create_dataset("Ex", data=self.data.Ex)
            file['Ex'].attrs['description'] = 'Electric field data: [x, y, z] where x, y, z are the spatial dimensions of the simulation.'
            file['Ex'].attrs['units'] = 'V/m'
            file['Ex'].attrs['dim'] = "x, y, z"

            file.create_dataset("Ey", data=self.data.Ey)
            file['Ey'].attrs['description'] = 'Electric field data: [x, y, z] where x, y, z are the spatial dimensions of the simulation.'
            file['Ey'].attrs['units'] = 'V/m'
            file['Ey'].attrs['dim'] = "x, y, z"

            file.create_dataset("Ez", data=self.data.Ez)
            file['Ez'].attrs['description'] = 'Electric field data: [x, y, z] where x, y, z are the spatial dimensions of the simulation.'
            file['Ez'].attrs['units'] = 'V/m'
            file['Ez'].attrs['dim'] = "x, y, z"

            file.create_dataset("Bx", data=self.data.Bx)
            file['Bx'].attrs['description'] = 'Magnetic field data: [x, y, z] where x, y, z are the spatial dimensions of the simulation.'
            file['Bx'].attrs['units'] = 'T'
            file['Bx'].attrs['dim'] = "x, y, z"

            file.create_dataset("By", data=self.data.By)
            file['By'].attrs['description'] = 'Magnetic field data: [x, y, z] where x, y, z are the spatial dimensions of the simulation.'
            file['By'].attrs['units'] = 'T'
            file['By'].attrs['dim'] = "x, y, z"

            file.create_dataset("Bz", data=self.data.Bz)
            file['Bz'].attrs['description'] = 'Magnetic field data: [x, y, z] where x, y, z are the spatial dimensions of the simulation.'
            file['Bz'].attrs['units'] = 'T'
            file['Bz'].attrs['dim'] = "x, y, z"

            file.create_dataset("nd", data=self.data.nd)
            file['nd'].attrs['description'] = 'Number density data normalized to nd0: [x, y, z] where x, y, z are the spatial dimensions of the simulation.'
            file['nd'].attrs['units'] = 'Unitless'
            file['nd'].attrs['dim'] = "x, y, z"

            file.create_dataset("nde", data=self.data.nde)
            file['nde'].attrs['description'] = 'Electron number density data normalized to nd0: [x, y, z] where x, y, z are the spatial dimensions of the simulation.'
            file['nde'].attrs['units'] = 'Unitless'
            file['nde'].attrs['dim'] = "x, y, z"

            file.create_dataset("ndi", data=self.data.ndi)
            file['ndi'].attrs['description'] = 'Ion number density data normalized to nd0: [x, y, z] where x, y, z are the spatial dimensions of the simulation.'
            file['ndi'].attrs['units'] = 'Unitless'
            file['ndi'].attrs['dim'] = "x, y, z"

            file.attrs['DumpPeriod'] = self.dump_period

            #file.attrs["Lx"] = self.data.Lx
            #file.attrs["Ly"] = self.data.Ly
            #file.attrs["Lz"] = self.data.Lz
            file.attrs["x"] = self.data.x
            file.attrs["y"] = self.data.y
            file.attrs["z"] = self.data.z

        return None





@jit(nopython=True)
def _sim_boris_simple(r0, x, y, z, m, q, Ex, Ey, Ez, Bx, By, Bz, v0, coll_arr, dr=0.25, dt0=1e-10, Nt=1000, backwards = False):  
    """Run a simple particle simulation"""

    
    
    Nr = r0.shape[1]

    print('Running simulation with ', Nr, " initial conditions")
    
    dir = 1.0
    if backwards:
        dir = -1.0
        v0 *= dir

    drhalf = dr*0.5

    Rs = np.zeros((3, Nt, Nr))
    Vs = np.zeros((3, Nt, Nr))
    Ts = np.zeros((Nt, Nr))
    Finalstates = np.zeros(Nr)
    Finalindex = np.ones(Nr, dtype=np.int64)*(Nt-1)
    
    
    Bi = np.zeros(3)
    for j in prange(Nr):

        R = np.zeros((3, Nt))
        V = np.zeros((3, Nt))
        T = np.zeros(Nt)


        time = 0
        T[0] = time
        
        
        r = r0[:,j].copy()
        #Used for collisions
        ix = int(max((r[0]+drhalf)/dr,0))
        iy = int(max((r[1]+drhalf)/dr,0))
        iz = int(max((r[2]+drhalf)/dr,0))

        R[:, 0] = r.copy()

        v = v0[:,j].copy()
        V[:, 0] = v.copy()
        # Ensure the data type is float64

        dt = dt0 * dir
        Ei = np.zeros(3)
        Ei[0] = interp3(r,x, y, z, Ex, dr)
        Ei[1] = interp3(r,x, y, z, Ey, dr)
        Ei[2] = interp3(r,x, y, z, Ez, dr)

        for i in range(1, Nt):

            Ei[0] = interp3(r,x, y, z, Ex, dr)
            Ei[1] = interp3(r,x, y, z, Ey, dr)
            Ei[2] = interp3(r,x, y, z, Ez, dr)
            

            Bi[0] = interp3(r,x, y, z, Bx, dr)
            Bi[1] = interp3(r,x, y, z, By, dr)
            Bi[2] = interp3(r,x, y, z, Bz, dr)

            v_mag = np.sqrt(v[0]**2 + v[1]**2 + v[2]**2)
            t = q / m * Bi * 0.5 * dt
            
            s = 2.0 * t / (1.0 + np.dot(t, t))
            v_minus = v + q / (m * v_mag) * Ei * 0.5 * dt
            v_prime = v_minus + np.cross(v_minus, t)
            v_plus = v_minus + np.cross(v_prime, s)
            v = v_plus + q / (m ) * Ei * 0.5 * dt
            r += v * dt 
            R[:, i] = r
            V[:, i] = v
            time += dt 
            T[i] = time

            #Used for collisions
            ix = int(max((r[0]+drhalf)/dr,0))
            iy = int(max((r[1]+drhalf)/dr,0))
            iz = int(max((r[2]+drhalf)/dr,0))

            if r[0] < x[0] or r[1] < y[0] or r[2] < z[0]:
                print('Particle escaped at', r)
                Finalstates[j] = -1.0
                Finalindex[j] = i
                #return R[:, :i], V[:, :i], T[:i]
               
                break
            elif r[0] > x[-1] or r[1] > y[-1] or r[2] > z[-1]:
                print('Particle escaped at', r)
                Finalstates[j] = -1.0
                Finalindex[j] = i
                
                #return R[:, :i], V[:, :i], T[:i]
                break
            
            elif coll_arr[ix,iy,iz] == True:
                print('Particle collided with object at', r)
                Finalstates[j] = 1.0
                Finalindex[j] = i

                #return R[:, :i], V[:, :i], T[:i]
                break
            
            #if i % 100 == 0:
            #    print(Ei)
            
            """
            Probe collision, maybe unnecessary
            elif np.sqrt((r[0] - probexy[0])**2 + (r[1] - probexy[1])**2 ) < prober and r[2]<probez[1] and r[2]>probez[0]:
                print('Probe hit at', r)
                return R[:, :i], V[:, :i], E[:, :i], T[:i], Emags[:i], vmags[:i], r_avg[:,:ravgis]"""
        if Finalstates[j] == 0:
            print("Particle still in domain after final timestep at",r)
        
        Rs[:,:,j] = R[:,:]
        Vs[:,:,j] = V[:,:]
        Ts[:,j] = T[:]
        
        
        
    print("\nSimulation finished")
    
    return Rs, Vs, Ts, Finalstates, Finalindex



@jit(nopython=True)
def _sim_boris(r0, x, y, z, m, q, Ex, Ey, Ez, Bx, By, Bz, v0, dr=0.25, dt0=1e-10, Nt=1000, probexy=(41,45), prober=0.0025, probez = np.array([40,50]), backwards = False):  
    """Run a Boris particle simulation"""

    print('Running simulation')
    
    dir = 1.0
    if backwards:
        dir = -1.0
    R = np.zeros((3, Nt))
    V = np.zeros((3, Nt))
    E = np.zeros((3, Nt))
    T = np.zeros(Nt)
    time = 0
    T[0] = time
    r_avg = np.zeros((3,Nt))
    vmags = np.zeros(Nt)
    Emags = np.zeros(Nt)
    
    R[:, 0] = r0
    

    v = v0
    V[:, 0] = v0
    v0_mag = np.sqrt(v0[0]**2 + v0[1]**2 + v0[2]**2)
    vmags[0] = v0_mag
    rhalf = x[-1] * 0.5
    #rmax = mag3(rhalf, rhalf, rhalf)
    
    r = r0.copy()  # Ensure the data type is float64

    dt = dt0
    Ei = np.zeros(3)
    Ei[0] = interp3(r,x, y, z, Ex, dr)
    Ei[1] = interp3(r,x, y, z, Ey, dr)
    Ei[2] = interp3(r,x, y, z, Ez, dr)
    Emags[0] = mag3(Ei[0], Ei[1], Ei[2])
    Bi = np.zeros(3)
    r_rg = np.zeros(3)
    angle_ref = get_angle(v, r)
    #print('Initial angle =', angle_ref)
    angle0 = angle_ref
    iavg0 = 0
    ravgis = 0
    

    for i in range(1, Nt):

        v_mag = np.sqrt(v[0]**2 + v[1]**2 + v[2]**2)
        #dt = dt0 * v0_mag / v_mag

        #Ei[0] = interpn((x, y, z), Ex, r, bounds_error=False)
        #Ei[1] = interpn((x, y, z), Ey, r, bounds_error=False)
        #Ei[2] = interpn((x, y, z), Ez, r, bounds_error=False)


        Ei[0] = interp3(r,x, y, z, Ex, dr)
        Ei[1] = interp3(r,x, y, z, Ey, dr)
        Ei[2] = interp3(r,x, y, z, Ez, dr)
    
        Emags[i] = mag3(Ei[0], Ei[1], Ei[2])
        vmags[i] = v_mag
        E[0,i]  = Ei[0]
        E[1,i]  = Ei[1]
        E[2,i]  = Ei[2]
        
        #Bi[0] = interpn((x, y, z), Bx, r,  bounds_error=False)
        #Bi[1] = interpn((x, y, z), By, r,  bounds_error=False)
        #Bi[2] = interpn((x, y, z), Bz, r,  bounds_error=False)

        Bi[0] = interp3(r,x, y, z, Bx, dr)
        Bi[1] = interp3(r,x, y, z, By, dr)
        Bi[2] = interp3(r,x, y, z, Bz, dr)
    

        t = q / m * Bi * 0.5 * dt
        s = 2.0 * t / (1.0 + np.dot(t, t))
        v_minus = v + q / (m * v_mag) * Ei * 0.5 * dt
        v_prime = v_minus + np.cross(v_minus, t)
        v_plus = v_minus + np.cross(v_prime, s)
        v = v_plus + q / (m * v_mag) * Ei * 0.5 * dt
        r += v * dt * dir
        R[:, i] = r
        V[:, i] = v
        time += dt * dir
        T[i] = time
        


        angle = get_angle(v, r)
        if abs(angle - angle0) - angle_ref > np.pi:
            #print('Angle jump detected at i =', i)
            r_rg[0] = np.mean(R[0,iavg0:i])
            r_rg[1] = np.mean(R[1,iavg0:i])
            r_rg[2] = np.mean(R[2,iavg0:i])
            #print('r_rg =', r_rg)
            r_avg[:,ravgis] = r_rg
            iavg0 = i
            ravgis += 1
        angle0 = angle

        if r[0] < x[0] or r[1] < y[0] or r[2] < z[0]:
            print('Particle escaped at', r)
            return R[:, :i], V[:, :i], E[:, :i], T[:i], Emags[:i], vmags[:i], r_avg[:,:ravgis] 
        elif r[0] > x[-1] or r[1] > y[-1] or r[2] > z[-1]:
            print('Particle escaped at', r)
            return R[:, :i], V[:, :i], E[:, :i], T[:i], Emags[:i], vmags[:i], r_avg[:,:ravgis]
        
        
        """
        Probe collision, maybe unnecessary
        elif np.sqrt((r[0] - probexy[0])**2 + (r[1] - probexy[1])**2 ) < prober and r[2]<probez[1] and r[2]>probez[0]:
            print('Probe hit at', r)
            return R[:, :i], V[:, :i], E[:, :i], T[:i], Emags[:i], vmags[:i], r_avg[:,:ravgis]"""
    print("\nSimulation finished")
    
    return R, V, E, T, Emags, vmags, r_avg[:,:ravgis]




@jit(nopython=True)
def mag3(v1,v2,v3):
    return np.sqrt(v1**2 + v2**2 + v3**2)


@jit(nopython=True)
def mag(v):
    return np.sqrt(v[0]**2 + v[1]**2 + v[2]**2)


"""@jit(nopython=True)
def interp3(r,x,y,z,field,dr):
    ix = int(r[0]/dr)
    iy = int(r[1]/dr)
    iz = int(r[2]/dr)

    # Only correct with -0.5*dr for emses
    Fx = np.interp(r[0]-0.5*dr,x,field[:,iy,iz])
    Fy = np.interp(r[1]-0.5*dr,y,field[ix,:,iz])
    Fz = np.interp(r[2]-0.5*dr,z,field[ix,iy,:])
    return (Fx+ Fy + Fz) / 3.0"""

"""@jit(nopython=True)
def interp3(r,x,y,z,field,dr):
    ix = int(r[0]/dr)
    iy = int(r[1]/dr)
    iz = int(r[2]/dr)

    # Only correct with -0.5*dr for emses
    Fx = cubic_spline_interpolate(r[0]-0.5*dr,x,field[:,iy,iz])
    Fy = cubic_spline_interpolate(r[1]-0.5*dr,y,field[ix,:,iz])
    Fz = cubic_spline_interpolate(r[2]-0.5*dr,z,field[ix,iy,:])
    return (Fx+ Fy + Fz) / 3.0"""

@njit
def interp3(r,x,y,z,field,dr):
    return cubic_interp_3d_jit(field, r[0], r[1], r[2], dr, dr, dr)
    #return trilinear_interp(field, r[0], r[1], r[2], dr, dr, dr)



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




@njit
def solve_tridiagonal(a, b, c, d):
    """Solve the tridiagonal system using the Thomas algorithm."""
    n = len(d)
    # Create copies of b and d since we will modify them
    c_ = np.zeros(n-1)
    d_ = np.zeros(n)
    b_ = np.copy(b)

    # Forward sweep
    c_[0] = c[0] / b_[0]
    d_[0] = d[0] / b_[0]
    for i in range(1, n-1):
        temp = b_[i] - a[i] * c_[i-1]
        c_[i] = c[i] / temp
        d_[i] = (d[i] - a[i] * d_[i-1]) / temp
    d_[n-1] = (d[n-1] - a[n-1] * d_[n-2]) / (b_[n-1] - a[n-1] * c_[n-2])

    # Backward substitution
    x = np.zeros(n)
    x[n-1] = d_[n-1]
    for i in range(n-2, -1, -1):
        x[i] = d_[i] - c_[i] * x[i+1]
    
    return x

@njit
def numba_diff(arr):
    """Numba-compatible version of np.diff."""
    n = len(arr) - 1
    result = np.empty(n)
    for i in range(n):
        result[i] = arr[i+1] - arr[i]
    return result


@njit
def cubic_spline_coefficients(x, y):
    """Compute the coefficients of the natural cubic spline."""
    n = len(x) - 1
    h = x[1]-x[0]
    
    # Set up the system of equations to solve for m (the second derivatives)
    a = h[:-1]
    b = 2 * (h[:-1] + h[1:])
    c = h[1:]
    d = 6 * (np.diff(y[1:]) / h[1:] - np.diff(y[:-1]) / h[:-1])
    
    # Solve for m using a tridiagonal solver
    m = np.zeros(n+1)
    m[1:n] = solve_tridiagonal(a, b, c, d)

    # Return the coefficients for each interval
    coeffs = np.zeros((n, 4))  # Each interval will have 4 coefficients: a, b, c, d
    for i in range(n):
        coeffs[i, 0] = (m[i+1] - m[i]) / (6 * h[i])  # Coefficient a
        coeffs[i, 1] = m[i] / 2  # Coefficient b
        coeffs[i, 2] = (y[i+1] - y[i]) / h[i] - (2*m[i] + m[i+1]) * h[i] / 6  # Coefficient c
        coeffs[i, 3] = y[i]  # Coefficient d
    
    return coeffs

@njit
def cubic_spline_evaluate(x, coeffs, x_new, x_points):
    """Evaluate the cubic spline at new x values."""
    n = len(x_points) - 1
    y_new = np.zeros(len(x_new))
    
    for i in range(len(x_new)):
        # Find the interval for x_new[i]
        for j in range(n):
            if x_points[j] <= x_new[i] <= x_points[j+1]:
                dx = x_new[i] - x_points[j]
                # Evaluate the spline polynomial using the coefficients
                y_new[i] = (
                    coeffs[j, 0] * dx**3 +
                    coeffs[j, 1] * dx**2 +
                    coeffs[j, 2] * dx +
                    coeffs[j, 3]
                )
                break
    
    return y_new

# Wrapper function to fit and evaluate the cubic spline
@njit
def cubic_spline_interpolate(x_new, x ,y):
    coeffs = cubic_spline_coefficients(x, y)
    y_new = cubic_spline_evaluate(x, coeffs, x_new, x)
    return y_new


@jit(nopython=True)
def get_angle(v,r):
    return np.math.atan2(v[0] * r[1] - v[1] * r[0], np.dot(v, r))