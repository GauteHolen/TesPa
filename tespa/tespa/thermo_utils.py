import numpy as np

kb = 1.380649e-23
epsilon0 = 8.85418782e-12
e = 1.602176634e-19

def maxwell_boltzmann(v, m, T):
    '''
        This is a function that computes the Maxwell-Boltzmann Distribution for: 
            - `v`: np.array of velocities
            - `m`: mass of the gas of interest
            - `T`: temperature of the system of interest
    '''
    r = kb # J / (mol * K)
    return(4.*np.pi * ((m / (2 * np.pi * r * T))**1.5) * (v**2) * np.exp(- (m * v**2)/(2 * r * T)))

def D_e_Bohm(q, B, T):
    return  kb*T/(abs(q)*B)

def D_e_perp(B,T,n):
    return n/(B**2 * np.sqrt(kb*T))

def D_e(T,m,nu):
    return kb*T/(m*nu)

def D_perp(D,wc,nu):
    return D/(1+(wc/nu)**2)

def nu_e(ne,Te):
    lnA = 10
    return 2.91e-6*ne*lnA/(Te**1.5)

def nu_e_perp(ne,Te,wc):
    nu = nu_e(ne,Te)
    return nu/(1+(wc/nu)**2)

def lambda_e(Te,qe,ne):
    kd = debyelength_e(Te,ne,qe)
    return 4*np.pi*ne*kd**3

def D_para(T, n_e, m, ln_Lambda):
    """
    Calculate the diffusion time to fill a cylinder with plasma from both ends along the magnetic field.

    Parameters:
    T : float
        Plasma temperature in Kelvin.
    L : float
        Half the length of the cylinder in meters.
    n_e : float
        Electron or ion number density in m^-3.
    ln_Lambda : float
        Coulomb logarithm.
    particle_type : str
        Type of particle ('electron' or 'ion'). Default is 'electron'.
    
    Returns:
    t_diff : float
        Diffusion time in seconds.
    """
    

    
    # Step 1: Calculate thermal velocity (v_th)
    v_th = np.sqrt(kb * T / m)
    
    # Step 2: Estimate collision frequency (nu) for electron-ion or ion-ion collisions
    # This is a simplified form of the collision frequency:
    nu = (4 * np.pi * n_e * e**4 * ln_Lambda) / ((4 * np.pi * epsilon0)**2 * m**2 * v_th**3)
    
    # Step 3: Calculate the parallel diffusion coefficient (D_parallel)
    D_parallel = v_th**2 / nu

    return D_parallel

def mean_free_path(m_e, T_e, n_e, ln_Lambda):
    """
    Calculate the mean free path for electrons in a plasma.

    Parameters:
    T_e : float
        Electron temperature in Kelvin.
    n_e : float
        Electron number density in m^-3.
    ln_Lambda : float
        Coulomb logarithm.
    
    Returns:
    lambda_mfp : float
        Mean free path in meters.
    """
    
    # Step 1: Calculate the electron thermal velocity (v_th)
    v_th = np.sqrt(2 * kb * T_e / m_e)
    
    # Step 2: Calculate the collision frequency (nu_c)
    nu_c = (4 * np.pi * n_e * e**4 * ln_Lambda) / ((4 * np.pi * epsilon0)**2 * m_e**2 * v_th**3)
    
    # Step 3: Calculate the mean free path (lambda_mfp)
    lambda_mfp = v_th / nu_c
    
    return lambda_mfp


def coulomb_logarithm(ne, Te):
    # Constants
    e = 1.602e-19      # Elementary charge (C)
    epsilon_0 = 8.854e-12  # Vacuum permittivity (F/m)
    k_B = 1.38e-23     # Boltzmann constant (J/K)
    
    # Convert temperature to eV
    Te_eV = Te / 11604  # 1 eV = 11604 K

    # Debye length
    lambda_D = np.sqrt(epsilon_0 * k_B * Te / (ne * e**2))

    # Minimum impact parameter (closest approach)
    b_min = e**2 / (4 * np.pi * epsilon_0 * k_B * Te)

    # Coulomb logarithm
    ln_lambda = np.log(lambda_D / b_min)
    
    return ln_lambda


def maxwell_boltzmann_vx(vx, m, T):
    '''
        This is a function that computes the Maxwell-Boltzmann Distribution for: 
            - `v`: np.array of velocities in x direction
            - `m`: mass of the gas of interest
            - `T`: temperature of the system of interest
    '''
    r = kb # J / (mol * K)
    f = ( ((m / (2 * np.pi * r * T))**0.5) *  np.exp(- (m * vx**2)/(2 * r * T)))
    A = np.trapz(f,vx)
    return f / A

def maxwell_boltzmann_vxy(vxy, m, T):
    '''
        This is a function that computes the Maxwell-Boltzmann Distribution for: 
            - `v`: np.array of velocities in xy plane
            - `m`: mass of the gas of interest
            - `T`: temperature of the system of interest
    '''
    r = kb # J / (mol * K)
    f = (vxy * ((m / (2 * np.pi * r * T))**1) *  np.exp(- (m * vxy**2)/(2 * r * T)))
    A = np.trapz(f,vxy)
    return f / A


def gyroradius(m,v,q,B):
    return np.abs(m*v/(q*B))

def gyrofrequency(m,q,B):
    return np.abs(q)*B/m

def thermal_speed(T,m):
    return np.sqrt(2*kb*T/m)

def debyelength_e(Te,ne,qe):
    
    return np.sqrt(epsilon0*kb*Te/(ne*qe**2))



def maxwellian_velocity_distribution(N, v0, temperature, mass):
    """
    Initialize a Maxwellian velocity distribution centered at v0 in 3D.

    Parameters:
    N : int
        Number of particles (velocities to generate).
    v0 : array-like
        Desired drift velocity of the distribution (shape: 3,).
    temperature : float
        Temperature of the system (related to the velocity spread).
    mass : float
        Mass of the particles.

    Returns:
    velocities : ndarray
        Maxwellian distributed velocities with shape (N, 3).
    """
    # Boltzmann constant (assuming SI units)
    k_B = 1.380649e-23  # J/K
    
    # Standard deviation of the Maxwellian distribution (related to temperature and mass)
    sigma = np.sqrt(k_B * temperature / mass)
    
    # Generate velocities using normal distribution for each component (x, y, z)
    velocities = np.random.normal(0, sigma, size=(N, 3))
    
    # Shift by the desired velocity v0
    velocities += v0
    
    return velocities

