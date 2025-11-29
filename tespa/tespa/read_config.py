from .dust_params import DustParams
from .plasma_params import PlasmaParams
from .sim_params import SimParams
import configparser



def read_dust_config(path: str, verbose: bool = False) -> tuple[DustParams, PlasmaParams, SimParams]:
    config = configparser.ConfigParser(inline_comment_prefixes=("#",))
    config.read(path)

    # Read Dust Parameters
    Nd = config.getint('dust', 'Nd')
    density = config.getfloat('dust', 'density')
    init_geometry = config.get('dust', 'init_geometry', fallback="sphere")
    r0 = tuple(map(float, config.get('dust', 'r0', fallback="0.1,0.1,0.1").split(',')))
    Rs = config.getfloat('dust', 'Rs', fallback=0.1)
    rdust = config.getfloat('dust', 'rdust', fallback=1e-6)
    q = config.getfloat('dust', 'q', fallback=0.0)
    dust_params = DustParams(Nd=Nd, 
                             density=density, 
                             init_geometry=init_geometry, 
                             r0=r0, 
                             Rs=Rs, 
                             rdust=rdust, 
                             q=q,
                             verbose=verbose)

    # Read Plasma Parameters
    ne0 = config.getfloat('plasma', 'ne0')
    ni0 = config.getfloat('plasma', 'ni0')
    Te_K = config.getfloat('plasma', 'Te_K', fallback=None)
    Ti_K = config.getfloat('plasma', 'Ti_K', fallback=None)
    Te_eV = config.getfloat('plasma', 'Te_eV', fallback=None)
    Ti_eV = config.getfloat('plasma', 'Ti_eV', fallback=None)
    mass_ratio = config.getfloat('plasma', 'mass_ratio', fallback=1836.0)
    photoemission_current_density = config.getfloat('plasma', 'photoemission_current_density', fallback=0.0)
    nn0 = config.getfloat('plasma', 'nn0', fallback=0.0)
    nphe0 = config.getfloat('plasma', 'nphe0', fallback=0.0)
    Tn_K = config.getfloat('plasma', 'Tn_K', fallback=300.0)
    Tphe_K = config.getfloat('plasma', 'Tphe_K', fallback=None)
    Tphe_eV = config.getfloat('plasma', 'Tphe_eV', fallback=2)
    neutral_delta = config.getfloat('plasma', 'neutral_delta', fallback=1.44)
    neutral_species = config.get('plasma', 'neutral_species', fallback="Ar")
    neutral_mass_kg = config.getfloat('plasma', 'neutral_mass_kg', fallback=None)
    fields_npz_path = config.get('plasma', 'fields_npz_path', fallback=None)

    plasma_params = PlasmaParams(   ne0=ne0,
                                    ni0=ni0,
                                    Te_K=Te_K,
                                    Ti_K=Ti_K,
                                    Te_eV=Te_eV,
                                    Ti_eV=Ti_eV,
                                    mass_ratio=mass_ratio,
                                    photoemission_current_density=photoemission_current_density,
                                    nn0=nn0,
                                    nphe0=nphe0,
                                    Tn_K=Tn_K,
                                    Tphe_K=Tphe_K,
                                    Tphe_eV=Tphe_eV,
                                    neutral_delta=neutral_delta,
                                    neutral_species=neutral_species,
                                    neutral_mass_kg=neutral_mass_kg,
                                    verbose=verbose,
                                    fields_npz_path=fields_npz_path)
    
    # Read Simulation Parameters

    Nx = config.getint('simbox', 'Nx')
    Ny = config.getint('simbox', 'Ny')
    Nz = config.getint('simbox', 'Nz')
    dx = config.getfloat('simbox', 'dx')
    Nt = config.getint('time', 'Nt')
    dt = config.getfloat('time', 'dt')
    x_boundary = config.get('simbox', 'x_boundary_condition', fallback="periodic")
    y_boundary = config.get('simbox', 'y_boundary_condition', fallback="periodic")
    z_boundary = config.get('simbox', 'z_boundary_condition', fallback="periodic")
    sub_boundary = config.get('simbox', 'sub_boundary', fallback=None)
    if sub_boundary is not None:
        sub_boundary = tuple(map(float, sub_boundary.split(',')))
    sim_params = SimParams(  Nx=Nx,
                             Ny=Ny,
                             Nz=Nz,
                             dx=dx,
                             dt=dt,
                             Nt=Nt,
                             x_boundary=x_boundary,
                             y_boundary=y_boundary,
                             z_boundary=z_boundary,
                             sub_boundary=sub_boundary,
                             verbose=verbose)

    
    return dust_params, plasma_params, sim_params
    



