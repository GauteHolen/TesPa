from .read_config import read_dust_config
from .sim import TesPa


def init_from_file(path: str, 
                   verbose: bool = False) -> TesPa:
    """Initialize a TesPa simulation from a given file path.

    Args:
        path (str): Path to the .ini file.

    Returns:
        TesPa: Initialized TesPa simulation object.
    """
    dust_params, plasma_params, sim_params = read_dust_config(path, verbose=verbose)
    sim = TesPa()
    sim.dust_params = dust_params
    sim.plasma_params = plasma_params
    sim.sim_params = sim_params
    if plasma_params.fields_npz_path is not None:
        if verbose:
            print(f"Loading fields from {plasma_params.fields_npz_path}")
        sim.data.load_from_npz(plasma_params.fields_npz_path, verbose=verbose)
    return sim