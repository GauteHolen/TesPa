import numpy as np


def get_velocity_field(Jx : np.ndarray,
                       Jy : np.ndarray,
                       Jz : np.ndarray,
                       N : np.ndarray,
                       q : float):
    """_summary_

    Args:
        Jx (np.ndarray): _description_
        Jy (np.ndarray): _description_
        Jz (np.ndarray): _description_
        N (np.ndarray): _description_
        q (float): _description_

    Returns:
        _type_: _description_
    """
    vx = Jx /(N*q)
    vy = Jy /(N*q)
    vz = Jz /(N*q)
    return vx,vy,vz