from numba import njit, prange
import numpy as np
from .get_field_vec import get_field_vec3
from .boris_pusher import boris_pusher_Fext
from .dust_charge import Q0_precompute,delta_dust_charge_density,compute_potential, photoemission_charge_density, converge_charge
from .interpolate_cubic import cubic_interp_3d_jit
from .drag import ion_drag_force, neutral_drag_force
from .boundary import periodic, reflective
from .verlet_pusher import verlet_pusher_Fext


q_elementary = 1.602176634e-19
me = 9.10938356e-31

@njit()
def _sim_boris_dust(r0s, 
                    x, y, z, 
                    ms, qs, a,
                    Ex, Ey, Ez, 
                    Bx, By, Bz,
                    nde,ndi,ndphe,
                    Te,Ti,Teph,
                    Jex, Jey, Jez, 
                    Jix, Jiy, Jiz,
                    v0s, 
                    dr=0.25, dt0=1e-10, Nt=1000,
                    kd = 1.0, 
                    backwards = False, 
                    aG = 0.0, 
                    massratio = 1000,
                    precompute_charge = 1000,
                    dt_precompute = 1e-7,
                    recompute_charge = 100, 
                    boundary_conditions_side = 'periodic',
                    sub_boundary = None,
                    photoemission_current_density = None,
                    beta_epstein = 0.0):  
    """Run a simple particle simulation"""
    print("TE, TI, Tph =", Te, Ti, Teph)
    print("ne_avg", np.mean(nde), "ni_avg", np.mean(ndi), "nph_avg", np.mean(ndphe))
    print("Drag coefficient beta_epstein =", beta_epstein)
    


    print('Running simulation for ', Nt, " steps")
    if sub_boundary is None:
        boundary_x = (x[0], x[-1])
        boundary_y = (y[0], y[-1])
        boundary_z = (z[0], z[-1])
    else:
        boundary_x = (sub_boundary[0],sub_boundary[1])
        boundary_y = (sub_boundary[2],sub_boundary[3])
        boundary_z = (sub_boundary[4], sub_boundary[5])
    print("Simulation boundaries : \n x ", boundary_x, "\n y ",boundary_y, "\n z ", boundary_z)
    

    maxA = 1e5

    if photoemission_current_density is None:
        photoemission_current_density = 0.0

    mi = me * massratio

    #boundary condition function selection
    if boundary_conditions_side == 'periodic':
        boundary_func_x = periodic
        boundary_func_y = periodic
    
    boundary_func_z = reflective 

    
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
    Qs = np.zeros((Np,Nt))
    Zdst = np.zeros((Np,Nt))
    Qs[:,0] = qs
    phist = np.zeros((Np,Nt))

    
    rs = r0s.copy()  # Ensure the data type is float64

    dt = dt0
    Ei = np.zeros(3)
    """    Ei[0] = interp3(r,x, y, z, Ex, dr)
    Ei[1] = interp3(r,x, y, z, Ey, dr)
    Ei[2] = interp3(r,x, y, z, Ez, dr)"""
    Bi = np.zeros(3)

    r = np.zeros(3)
    v = np.zeros(3)
    Fext = np.zeros(3)
    Fdrag = np.zeros(3)
    Fdrag_neutrals = np.zeros(3)
    Je = np.zeros(3)
    Ji = np.zeros(3)
    ve = np.zeros(3)
    vi = np.zeros(3)

    # We want to pre-compute I0/n
    Q0ne_precompute = np.zeros(Np, dtype=np.float64)
    Q0ni_precompute = np.zeros(Np, dtype=np.float64)
    Q0neph_precompute = np.zeros(Np, dtype=np.float64)
        

    Vs = np.ones(Np, dtype=np.float64) #potentials
    Zds = np.zeros(Np, dtype=np.float64) #number of charges on dust

    for j in range(Np):
        Q0ne_precompute[j] = Q0_precompute(Te,me,a[j])
        Q0ni_precompute[j] = Q0_precompute(Ti,mi,a[j])
    if Teph> 1e-12 and np.mean(ndphe) > 1e-12:
        for j in range(Np):
            Q0neph_precompute[j] = Q0_precompute(Teph,me,a[j])

    print("Precomputing initial charge")
    for j in range(Np):
        r = rs[:,j]
        #print("interpolating for particle ", j, " at position ", r)
        ne = cubic_interp_3d_jit(nde,r[0],r[1],r[2],dr,dr,dr)
        Ji = get_field_vec3(r,Ji,Jix,Jiy,Jiz,dr)
        ni = cubic_interp_3d_jit(ndi,r[0],r[1],r[2],dr,dr,dr)
        nphe = cubic_interp_3d_jit(ndphe,r[0],r[1],r[2],dr,dr,dr)
        #print("computing vi for particle ", j)
        vi = Ji / (ni * q_elementary) - v0s[:,j]
        
        #print("converging charge for particle ", j)
        Zds[j], Vs[j] = converge_charge(Zds[j], 
                                        Q0ne_precompute[j], Q0ni_precompute[j],Q0neph_precompute[j],
                                        Vs[j],  
                                        ne, ni, nphe, 
                                        Te, Ti,Teph, 
                                        photoemission_current_density, precompute_charge, a[j], vi, mi, tol=1e-3)
        qs[j] = Zds[j] * q_elementary
    #print("Initial charge precomputation done, Q0ne =", Q0ne_precompute, " Q0ni =", Q0ni_precompute, " Q0neph =", Q0neph_precompute)

    phist[:,0] = Vs.copy()
    Zdst[:,0] = Zds.copy()

    print("Starting main simulation loop")
    for i in range(1, Nt):
        if i%10==0:
            print("Timestep ", i, " of ", Nt, "\r")


        for j in prange(Np):
            r = rs[:,j]
            v = vs[:,j]

            Ei = get_field_vec3(r,Ei,Ex,Ey,Ez,dr)
            Bi = get_field_vec3(r,Bi,Bx,By,Bz,dr)
            Je = get_field_vec3(r,Je,Jex,Jey,Jez,dr)
            Ji = get_field_vec3(r,Ji,Jix,Jiy,Jiz,dr)

            ne = cubic_interp_3d_jit(nde,r[0],r[1],r[2],dr,dr,dr)
            ni = cubic_interp_3d_jit(ndi,r[0],r[1],r[2],dr,dr,dr)
            nphe =   cubic_interp_3d_jit(ndphe,r[0],r[1],r[2],dr,dr,dr)

            #print("position (x,y,z)=", r[0], r[1], r[2])
            #print("ne, ni, nphe =", ne, ni, nphe)
            #print("Ei =", Ei)
            #print("Bi =", Bi)

            vi = Ji / (ni * q_elementary) - v
            #print("vi =", vi)

            Zds[j], Vs[j] = converge_charge(Zds[j], 
                                            Q0ne_precompute[j], Q0ni_precompute[j], Q0neph_precompute[j],
                                            Vs[j], 
                                            ne, ni, nphe,
                                            Te, Ti, Teph,
                                            photoemission_current_density, recompute_charge, a[j], vi, mi, tol=1e-3)
            qs[j] = Zds[j] * q_elementary

            ax = 0.0
            ay = 0.0
            az = 0.0

            fx = 0.0
            fy = 0.0
            fz = 0.0

            #fx = qs[j]*(Ei[0] + v[1]*Bi[2] - v[2]*Bi[1])
            #fy = qs[j]*(Ei[1] + v[2]*Bi[0] - v[0]*Bi[2])
            #fz = qs[j]*(Ei[2] + v[0]*Bi[1] - v[1]*Bi[0])

            for k in range(Np):
                if (k != j):
                    rmin = a[j] + a[k]

                    xdiff = ( rs[0,j]-rs[0,k] ) #- round((x[i]-x[j])/(2.0*Lx)) * 2.0*Lx
                    ydiff = ( rs[1,j]-rs[1,k] ) #- round((y[i]-y[j])/(2.0*Ly)) * 2.0*Ly
                    zdiff = ( rs[2,j]-rs[2,k] ) #- round((z[i]-z[j])/(2.0*Lz)) * 2.0*Lz
                    rmag = np.sqrt(xdiff*xdiff + ydiff*ydiff + zdiff*zdiff)
                    rdiff = max(rmag,rmin)
                    r_inv = 1/(rdiff*rdiff*rdiff)
                    QQr3 = (qs[j]*qs[k])*r_inv     #r or r^3 ?
                    fx += xdiff*(1.0+kd*rdiff)*np.exp(-kd*rdiff)*QQr3   # xdiff/(r*r*r)
                    fy += ydiff*(1.0+kd*rdiff)*np.exp(-kd*rdiff)*QQr3    # ydiff/(r*r*r)
                    fz += zdiff*(1.0+kd*rdiff)*np.exp(-kd*rdiff)*QQr3 #+ zdiff*g + Lz*g # zdiff/(r*r*r)
                         
                    if abs(fx/ms[j])>maxA or abs(fy/ms[j])>maxA or abs(fz/ms[j])>maxA:
                        print("--- ERROR TIMESTEP ",i," PARTICLE ",j," INTERACTING WITH PARTICLE ",k," ---")
                        print("f too high",fx/ms[j],fy/ms[j],fz/ms[j])
                        print("r_inv = ",r_inv, "Zd1 = ",Zds[j], "Zd2 = ",Zds[k])
                        print("QQr3 = ",QQr3)
                        print("r1 = ",r, "r2 = ",rs[:,k], "rdiff = ",rdiff, "rmin = ",rmin)
                        
                        return R[:,:, :i], V[:,:, :i], T[:i], Zdst[:,:i], Qs[:,:i]
                        
                ax += fx/ms[j]
                ay += fy/ms[j]
                az += fz/ms[j] 


                #print("ax,ay,az",ax,ay,az)
            az += aG #1.62 on the moon, 9.81 on earth

            
            if np.linalg.norm(vi) > 1e-12:
                Fdrag = ion_drag_force(ni, Ti, a[j], mi, vi, qs[j], Vs[j])
            else:
                Fdrag = np.zeros(3)

            

            Fext[0] = ax*ms[j] + Fdrag[0] #+ Fdrag_neutrals[0]
            Fext[1] = ay*ms[j] + Fdrag[1] #+ Fdrag_neutrals[1]
            Fext[2] = az*ms[j] + Fdrag[2] #+ Fdrag_neutrals[2]

            Fdrag_neutrals = neutral_drag_force(beta_epstein, ms[j], v)

            Fext[0] += Fdrag_neutrals[0]
            Fext[1] += Fdrag_neutrals[1]
            Fext[2] += Fdrag_neutrals[2]

            #print("F_ext, m, q, v, r =", Fext, ms[j], qs[j], v, r)

            #print("position before push:", r)
            #r,v = boris_pusher_Fext(qs[j],ms[j],Ei,Bi,r,v,dt*dir,Fext)
            r,v = verlet_pusher_Fext(qs[j],ms[j],Ei,Bi,r,v,dt*dir,Fext)
            #print("position after push:", r)
            # Apply boundary conditions
            
            r[0],v[0] = boundary_func_x(r[0],v[0],boundary_x[0], boundary_x[1])
            r[1],v[1] = boundary_func_y(r[1],v[1],boundary_y[0],boundary_y[1])
            r[2],v[2] = boundary_func_z(r[2],v[2],boundary_z[0], boundary_z[1])
            

            rs[:,j] = r
            vs[:,j] = v

            R[:,j, i] = r
            V[:,j, i] = v
            time += dt * dir
            T[i] = time
        
            


        #print("r = ",r)
        Qs[:,i] = qs
        Zdst[:,i] = Zds
        phist[:,i] = Vs

            
            
    print("\nSimulation finished at step ",i)
    #print(R.shape,V.shape,T.shape)
    
    return R, V, T, Zdst, phist