"""
This module contains functions for calculating electron and ion currents in a cylinder and a sphere.

Author: Gaute Holen
Date: 27.06.2024
"""

import numpy as np
from numba import njit, prange, jit
from .load_emses import LoadEMSES



class TesPa():
    def __init__(self):
        pass
    
    def load_emses(self,dirname,dx):
        """Load emses data from a given directory. Uses the LoadEMSES class.

        Args:
            dirname (str): path to directory
            dx (float): gridsize in meters
        """
        
        self.dir = dirname
        self.data = LoadEMSES(self.dir,dx)

    
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
            
    def set_E(self,E_mag,dir='x',gradient=(1,1), gradient_extent=1.0):
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
        Ex = self.data.Ex.copy()
        Ey = self.data.Ey.copy()
        Ez = self.data.Ez.copy()
        Bx = self.data.Bx.copy()
        By = self.data.By.copy()
        Bz = self.data.Bz.copy()
        x = self.data.x.copy()
        y = self.data.y.copy()
        z = self.data.z.copy()
        dr = self.data.dx
        


        self.r,self.v,self.e,self.t,self.emags,self.vmags,self.r_avg = _sim_boris(r0,x,y,z,m,q,Ex,Ey,Ez,Bx,By,Bz,v0,dr,dt0,int(Nt))
    

    def run_dust(self, r0s,v0s,ms,qs,dt0=1e-3,Nt=1e4,aG=0.0,rdust=0.003):
        Nt = int(Nt)
        Ex = self.data.Ex.copy()
        Ey = self.data.Ey.copy()
        Ez = self.data.Ez.copy()
        Bx = self.data.Bx.copy()
        By = self.data.By.copy()
        Bz = self.data.Bz.copy()
        x = self.data.x.copy()
        y = self.data.y.copy()
        z = self.data.z.copy()
        dr = self.data.dx

        self.r,self.v,self.t = _sim_boris_dust(r0s,x,y,z,ms,qs,Ex,Ey,Ez,Bx,By,Bz,v0s,dr,dt0,Nt,aG=aG, rdust = rdust)


    def run_simple(self, r0,v0,m,q,dt0=1e-10, Nt=1e5):
        """Run the simulation with the given parameters

        Args:
            r0 (np.array([r_x,r_y,r_z])): initial position
            v0 (np.array([v_x,v_y,v_z])): Initial velocity
            m (float): particle mass
            q (float): particle charge
            dt0 (float, optional): Initial timestep for adaptive timestep. Defaults to 1e-10.
            Nt (int, optional): Number of time steps. Defaults to 1e5.
        """
        Ex = self.data.Ex.copy()
        Ey = self.data.Ey.copy()
        Ez = self.data.Ez.copy()
        Bx = self.data.Bx.copy()
        By = self.data.By.copy()
        Bz = self.data.Bz.copy()
        x = self.data.x.copy()
        y = self.data.y.copy()
        z = self.data.z.copy()
        dr = self.data.dx
        

        self.r,self.v,self.t = _sim_boris_simple(r0,x,y,z,m,q,Ex,Ey,Ez,Bx,By,Bz,v0,dr,dt0,int(Nt))


@njit()
def _sim_boris_dust(r0s, x, y, z, ms, qs, Ex, Ey, Ez, Bx, By, Bz, v0s, dr=0.25, dt0=1e-10, Nt=1000,kd = 1.0, backwards = False, aG = 0.0, rdust = 0.003):  
    """Run a simple particle simulation"""

    print('Running simulation')
    rmin = 5*rdust
    Np = r0s.shape[1]
    dir = 1.0
    if backwards:
        dir = -1.0
    R = np.zeros((3, Np, Nt))
    V = np.zeros((3, Np, Nt))
    T = np.zeros(Nt)

    time = 0
    T[0] = time
    R[:,:, 0] = r0s.copy()
    vs = v0s.copy()
    V[:,:, 0] = v0s.copy()

    
    rs = r0s.copy()  # Ensure the data type is float64

    dt = dt0
    Ei = np.zeros(3)
    """    Ei[0] = interp3(r,x, y, z, Ex, dr)
    Ei[1] = interp3(r,x, y, z, Ey, dr)
    Ei[2] = interp3(r,x, y, z, Ez, dr)"""
    Bi = np.zeros(3)

    r = np.zeros(3)
    v = np.zeros(3)
    
    for i in range(1, Nt):

        for j in range(Np):
            r = rs[:,j]
            #q = qs[j]
            #m = ms[j]
            v = vs[:,j]

            


            #contribution to v from Boris method
            Ei[0] = interp3(r,x, y, z, Ex, dr)
            Ei[1] = interp3(r,x, y, z, Ey, dr)
            Ei[2] = interp3(r,x, y, z, Ez, dr)
            

            Bi[0] = interp3(r,x, y, z, Bx, dr)
            Bi[1] = interp3(r,x, y, z, By, dr)
            Bi[2] = interp3(r,x, y, z, Bz, dr)

            v_mag = max(np.sqrt(v[0]**2 + v[1]**2 + v[2]**2),1e-12)

            

            """t = q / m * Bi * 0.5 * dt
            s = 2.0 * t / (1.0 + np.dot(t, t))
            v_minus = v + q / (m * v_mag) * Ei * 0.5 * dt
            v_prime = v_minus + np.cross(v_minus, t)
            v_plus = v_minus + np.cross(v_prime, s)
            v += v_plus + q / (m * v_mag) * Ei * 0.5 * dt"""



            #print("v before pp interact",v)

            #Contribution to v from interparticle forces

            #fx = 0.0
            #fy = 0.0
            #fz = 0.0
            
            ax = 0.0
            ay = 0.0
            az = 0.0

            fx = qs[j]*(Ei[0] + v[1]*Bi[2] - v[2]*Bi[1])
            fy = qs[j]*(Ei[1] + v[2]*Bi[0] - v[0]*Bi[2])
            fz = qs[j]*(Ei[2] + v[0]*Bi[1] - v[1]*Bi[0])


            for k in range(Np):
                if (k != j):

                    xdiff = ( rs[0,j]-rs[0,k] ) #- round((x[i]-x[j])/(2.0*Lx)) * 2.0*Lx
                    ydiff = ( rs[1,j]-rs[1,k] ) #- round((y[i]-y[j])/(2.0*Ly)) * 2.0*Ly
                    zdiff = ( rs[2,j]-rs[2,k] ) #- round((z[i]-z[j])/(2.0*Lz)) * 2.0*Lz
                    rmag = np.sqrt(xdiff*xdiff + ydiff*ydiff + zdiff*zdiff)
                    rdiff = max(rmag,rmin)
                    r_inv = 1/(rdiff*rdiff*rdiff)
                    QQr3 = (qs[i]*qs[j])*r_inv     #r or r^3 ?
                    fx += xdiff*(1.0+kd*rdiff)*np.exp(-kd*rdiff)*QQr3   # xdiff/(r*r*r)
                    fy += ydiff*(1.0+kd*rdiff)*np.exp(-kd*rdiff)*QQr3    # ydiff/(r*r*r)
                    fz += zdiff*(1.0+kd*rdiff)*np.exp(-kd*rdiff)*QQr3 #+ zdiff*g + Lz*g # zdiff/(r*r*r)

                    """if rmag <= rmin:
                        print(rmag)
                        print("Collision at",rs[:,j],rs[:,k])
                        print(fx/ms[j],fy/ms[j],fz/ms[j])"""
                         
                    if abs(fx/ms[j])>1e1 or abs(fy/ms[j])>1e1 or abs(fz/ms[j])>1e1:
                        print("f too high",fx,fy,fz)
                        print("r_inv = ",r_inv, "q1 = ",qs[i], "q2 = ",qs[j])
                        print("qs = ",qs)
                        print("QQr3 = ",QQr3)
                        print("r1 = ",r, "r2 = ",rs[:,k], "rdiff = ",rdiff, "rmin = ",rmin)

                #Can implement drag etc if we want here


                ax += fx/ms[j]
                ay += fy/ms[j]
                az += fz/ms[j] 


                #print("ax,ay,az",ax,ay,az)
            az += aG #1.62 on the moon, 9.81 on earth

            
            v[0] +=  0.5 * ax * dt
            v[1] +=  0.5 * ay * dt
            v[2] +=  0.5 * az * dt
            #print("v after pp interact",v)
            
            """if mag3(v[0],v[1],v[2]) > 1e1:
                print("v too high",v)
                print("r = ",r)"""


            # Update position
            r += v * dt * dir
            R[:,j, i] = r
            V[:,j, i] = v
            time += dt * dir
            T[i] = time
            

            if r[0] < x[0] or r[1] < y[0] or r[2] < z[0]:
                print('Particle escaped at', r)
                return R[:,:, :i], V[:,:, :i], T[:i]
            elif r[0] > x[-1] or r[1] > y[-1] or r[2] > z[-1]:
                print('Particle escaped at', r)
                return R[:,:, :i], V[:,:, :i], T[:i], 
        #print("r = ",r)
            
            
    print("\nSimulation finished")
    #print(R.shape,V.shape,T.shape)
    
    return R, V, T


@jit(nopython=True)
def _sim_boris_simple(r0, x, y, z, m, q, Ex, Ey, Ez, Bx, By, Bz, v0, dr=0.25, dt0=1e-10, Nt=1000, backwards = False):  
    """Run a simple particle simulation"""

    print('Running simulation')
    
    dir = 1.0
    if backwards:
        dir = -1.0
    R = np.zeros((3, Nt))
    V = np.zeros((3, Nt))
    T = np.zeros(Nt)


    time = 0
    T[0] = time
    R[:, 0] = r0.copy()
    v = v0.copy()
    V[:, 0] = v0

    
    r = r0.copy()  # Ensure the data type is float64

    dt = dt0
    Ei = np.zeros(3)
    Ei[0] = interp3(r,x, y, z, Ex, dr)
    Ei[1] = interp3(r,x, y, z, Ey, dr)
    Ei[2] = interp3(r,x, y, z, Ez, dr)
    Bi = np.zeros(3)
    
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
        v = v_plus + q / (m * v_mag) * Ei * 0.5 * dt
        r += v * dt * dir
        R[:, i] = r
        V[:, i] = v
        time += dt * dir
        T[i] = time
        

        if r[0] < x[0] or r[1] < y[0] or r[2] < z[0]:
            print('Particle escaped at', r)
            return R[:, :i], V[:, :i], T[:i]
        elif r[0] > x[-1] or r[1] > y[-1] or r[2] > z[-1]:
            print('Particle escaped at', r)
            return R[:, :i], V[:, :i], T[:i], 
        
        
        """
        Probe collision, maybe unnecessary
        elif np.sqrt((r[0] - probexy[0])**2 + (r[1] - probexy[1])**2 ) < prober and r[2]<probez[1] and r[2]>probez[0]:
            print('Probe hit at', r)
            return R[:, :i], V[:, :i], E[:, :i], T[:i], Emags[:i], vmags[:i], r_avg[:,:ravgis]"""
    print("\nSimulation finished")
    
    return R, V, T



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


@jit(nopython=True)
def interp3(r,x,y,z,field,dr):
    ix = int(r[0]/dr)
    iy = int(r[1]/dr)
    iz = int(r[2]/dr)

    Fx = np.interp(r[0],x,field[:,iy,iz])
    Fy = np.interp(r[1],y,field[ix,:,iz])
    Fz = np.interp(r[2],z,field[ix,iy,:])
    return (Fx+ Fy + Fz) / 3.0


@jit(nopython=True)
def get_angle(v,r):
    return np.math.atan2(v[0] * r[1] - v[1] * r[0], np.dot(v, r))