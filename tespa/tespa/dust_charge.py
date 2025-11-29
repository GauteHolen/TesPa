import numpy as np
from numba import njit
epsilon0 = 8.854187817e-12  # Vacuum permittivity in F/m
q_elementary = 1.602176634e-19      # Elementary charge in C
kb = 1.380649e-23

@njit
def delta_dust_charge_drift(Je,Ji, a, dt):
    """Calculate the dust charge density

    Args:
        Je (np.array): Electron current density
        Ji (np.array): Ion current density
        a (float): Dust radius
        dt (float): Time step

    Returns:
        np.array: Dust charge density
    """
    C = 4*np.pi*epsilon0*a
    je = np.sqrt(Je[0]**2 + Je[1]**2 + Je[2]**2)
    ji = np.sqrt(Ji[0]**2 + Ji[1]**2 + Ji[2]**2)
    I = (ji-je) * a**2
    Q = q_elementary * I * dt
    return Q / C

@njit
def compute_potential(Zd,a):
    C = 4*np.pi*epsilon0*a # can be precomputed
    return Zd * q_elementary / C

@njit
def compute_charge_potential(V,a):
    C = 4*np.pi*epsilon0*a # can be precomputed
    return V * C / q_elementary


@njit
def get_chi(qp,phi,T):
    return qp*phi / (kb*T)

@njit
def delta_dust_charge_density(Q0_precompute : float, n : float, phi : float, q : float, T : float):
    chi = get_chi(q,phi,T)
    if chi > 0: # repelling
        return Q0_precompute * n * np.exp(-chi)
    else: #attracting
        return Q0_precompute * n * (1-chi)

@njit
def Q0_precompute(T : float, m : float, a : float):
    """_summary_

    Args:
        T (float): _description_
        m (float): _description_
        a (float): _description_

    Returns:
        _type_: _description_
    """
    A = 4*np.pi*a**2
    return A*np.sqrt((kb*T)/(2*np.pi*m))

@njit
def photoemission_charge_density(Jph : float, a : float):
    """_summary_

    Args:
        Jph (float): _description_
        a (float): _description_

    Returns:
        _type_: _description_
    """
    A = np.pi*a**2 # area of projected circle
    return -Jph * A / q_elementary


@njit
def P_escape(phi,T):
    chi = get_chi(-q_elementary,phi,T)
    if chi < 0:
        return np.exp(chi)
    else:
        return 1.0

@njit
def delta_Zd(Q0ne, Q0ni,Q0eph, ne, ni,neph, V, Te, Ti, Teph, Jph, a, vi,mi):
    de = delta_dust_charge_density(Q0ne,ne,V,-q_elementary,Te)
    di_th = delta_dust_charge_density(Q0ni,ni,V,q_elementary,Ti)
    vi_mag = np.linalg.norm(vi)
    if vi_mag > 1:
        di_streaming = delta_streaming(a, ni, vi, mi, V, q_elementary)
        di = max(di_th, di_streaming)
    else:
        di = di_th

    deph = delta_dust_charge_density(Q0eph,neph,V,-q_elementary,Teph)
    dph = photoemission_charge_density(Jph, a)
    Pesc = P_escape(V, Teph)
    dph_Pesc = dph * Pesc

    dZd_dt = -de + di_th #+ dph_Pesc - deph
    return dZd_dt


@njit
def delta_Zd_simple(Q0ne, Q0ni,Q0eph, ne, ni,neph, V, Te, Ti, Teph, Jph, a, vi,mi):
    de = delta_dust_charge_density(Q0ne,ne,V,-q_elementary,Te)
    di = delta_dust_charge_density(Q0ni,ni,V,q_elementary,Ti)
    dZd_dt = -de + di
    return dZd_dt


@njit
def delta_streaming(a, n, v, m, phi, q):
    v_mag = np.linalg.norm(v)
    return n * np.pi * a**2 * v_mag * (1 - ( q * phi / (0.5 * m * v_mag**2)))


@njit
def dFdZ(Z, Q0ne, Q0ni,Q0eph, ne, ni,neph, Te, Ti, Teph, Jph, a, vi,mi):
    dZ = Z*1e-6 + 1e-12  # safe perturbation
    #V  = compute_potential(Z, a)
    #F0 = delta_Zd(Q0ne, Q0ni,Q0eph, ne, ni,neph, V, Te, Ti,Teph, Jph, a)

    Zp = Z + dZ
    Vp = compute_potential(Zp, a)
    Fp = delta_Zd(Q0ne, Q0ni,Q0eph, ne, ni,neph, Vp, Te, Ti,Teph, Jph, a, vi,mi)

    Zm = Z - dZ
    Vm = compute_potential(Zm, a)
    Fm = delta_Zd(Q0ne, Q0ni,Q0eph, ne, ni,neph, Vm, Te, Ti,Teph, Jph, a, vi,mi)
    return (Fp - Fm) / (2*dZ)



@njit
def dFdZ_simple(Z, Q0ne, Q0ni,Q0eph, ne, ni,neph, Te, Ti, Teph, Jph, a, vi,mi):
    dZ = Z*1e-6 + 1e-12  # safe perturbation
    #V  = compute_potential(Z, a)
    #F0 = delta_Zd(Q0ne, Q0ni,Q0eph, ne, ni,neph, V, Te, Ti,Teph, Jph, a)

    Zp = Z + dZ
    Vp = compute_potential(Zp, a)
    Fp = delta_Zd_simple(Q0ne, Q0ni,Q0eph, ne, ni,neph, Vp, Te, Ti,Teph, Jph, a, vi,mi)

    Zm = Z - dZ
    Vm = compute_potential(Zm, a)
    Fm = delta_Zd_simple(Q0ne, Q0ni,Q0eph, ne, ni,neph, Vm, Te, Ti,Teph, Jph, a, vi,mi)
    return (Fp - Fm) / (2*dZ)


@njit
def converge_charge(Z, 
                    Q0ne, Q0ni, Q0eph, V, 
                    ne, ni,neph, 
                    Te, Ti,Teph, 
                    Jph, 
                    Nt, a,vi,mi, tol=1e-3):
    
    if Teph < 1e-12 and Jph < 1e-12:
        
        
        for it in range(Nt):
            # Compute current potential and derivative
            V = compute_potential(Z, a)
            F = delta_Zd_simple(Q0ne, Q0ni, Q0eph, ne, ni,neph, V, Te, Ti, Teph, Jph, a, vi, mi)
            # Check for convergence
            if abs(F) < tol:
                return Z, V
            # Compute stiffness (Jacobian)
            dF = dFdZ_simple(Z, Q0ne, Q0ni,Q0eph, ne, ni,neph, Te, Ti,Teph, Jph, a, vi, mi)

            if dF == 0:
                dtau = 1.0
            else:
                dtau = 0.6 / abs(dF)  # optimal stable step
            Z = Z + F * dtau
    else:
        for it in range(Nt):

            # Compute current potential and derivative
            #print("potential")
            V = compute_potential(Z, a)
            #print("computing delta_Zd")
            F = delta_Zd(Q0ne, Q0ni, Q0eph, ne, ni,neph, V, Te, Ti, Teph, Jph, a, vi, mi)

            # Check for convergence
            if abs(F) < tol:
                return Z, V

            # Compute stiffness (Jacobian)
            dF = dFdZ(Z, Q0ne, Q0ni,Q0eph, ne, ni,neph, Te, Ti,Teph, Jph, a, vi, mi)
            if dF == 0:
                dtau = 1.0
            else:
                dtau = 0.6 / abs(dF)  # optimal stable step

            # Update
            Z = Z + F * dtau

    # return last computed values
    print(" -- WARNING: Charge did not converge --")
    V = compute_potential(Z, a)
    return Z, V


@njit
def converge_charge_array(Z, 
                    Q0ne, Q0ni, ne, ni, Te, Ti, Jph, 
                    Nt, a,vi,mi, tol=1e-3):
    Zds = np.zeros(Nt)
    Vds = np.zeros(Nt)
    Zds[0] = Z
    V = compute_potential(Z, a)
    Vds[0] = V

    converged = False
    for it in range(1,Nt):

        # Compute current potential and derivative
        
        F = delta_Zd(Q0ne, Q0ni, ne, ni, V, Te, Ti, Jph, a, vi, mi)

        # Check for convergence
        if abs(F) < tol:
            converged = True

        # Compute stiffness (Jacobian)
        dF = dFdZ(Z, Q0ne, Q0ni, ne, ni, Te, Ti, Jph, a, vi, mi)
        if dF == 0:
            dtau = 1.0
        else:
            dtau = 0.3 / abs(dF)  # optimal stable step

        # Update
        Z = Z + F * dtau
        V = compute_potential(Z, a)

        Zds[it] = Z
        Vds[it] = V
        if converged:
            Zds = Zds[:it+1]
            Vds = Vds[:it+1]
            print("Converged after ", it, " iterations")
            break

    # return last computed values
    V = compute_potential(Z, a)
    return Zds, Vds