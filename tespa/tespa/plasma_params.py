import numpy as np

eV_to_K = 11604.52500617  # 1 eV in Kelvin
atomic_mass_unit_kg = 1.66053906660e-27  # 1 amu in kg


neutral_mass_dict = {
    "He": 4.002602,
    "Ne": 20.1797,
    "Ar": 39.948,
    "Kr": 83.798,
    "Xe": 131.293,
    "N2": 28.0134,
    "O2": 31.9988,
    "CO2": 44.0095,
}

class PlasmaParams():
    def __init__(self, 
                 ne0 : float,
                 ni0 : float,
                 Te_K : float = None,
                 Ti_K : float = None,
                 Te_eV : float = None,
                 Ti_eV : float = None,
                 mass_ratio : float = 1836.0,
                 photoemission_current_density : float = 0.0,
                 nn0 : float = 0.0,
                 nphe0 : float = 0.0,
                 Tn_K : float = 300.0,
                 Tphe_K : float = None,
                 Tphe_eV : float = 2,
                 neutral_delta : float = 1.44,
                 neutral_species = "Ar",
                 neutral_mass_kg : float = None,
                 verbose: bool = False,
                 fields_npz_path: str = None) -> None:
        

        """Initialize plasma parameters.

        Args:
            ne0 (float): Electron number density in m^-3
            ni0 (float): Ion number density in m^-3
            Te_K (float, optional): Electron temperature in Kelvin. Defaults to None.
            Ti_K (float, optional): Ion temperature in Kelvin. Defaults to None.
            Te_eV (float, optional): Electron temperature in eV. Defaults to None.
            Ti_eV (float, optional): Ion temperature in eV. Defaults to None.
            mass_ratio (float, optional): Ion to electron mass ratio. Defaults to 1836.0.
            photoemission_current_density (float, optional): Photoemission current density. Defaults to 0.0.
            nn0 (float, optional): Neutral number density in m^-3. Defaults to 0.0.
            nphe0 (float, optional): Photoelectron number density in m^-3. Defaults to 0.0.
            Tn_K (float, optional): Neutral temperature in Kelvin. Defaults to 300.0.
            Tphe_K (float, optional): Photoelectron temperature in Kelvin. Defaults to None.
            Tphe_eV (float, optional): Photoelectron temperature in eV. Defaults to 2.
            neutral_delta (float, optional): Neutral parameter delta for Epstein drag. Defaults to 1.44.
            neutral_species (str, optional): Neutral species type. Defaults to "Ar".
            neutral_mass_kg (float, optional): Custom neutral mass in kg. Defaults to None.
        """


        if Te_K is None and Te_eV is None:
            raise ValueError("Either Te_K or Te_eV must be provided")
        if Ti_K is None and Ti_eV is None:
            raise ValueError("Either Ti_K or Ti_eV must be provided")
        
        if neutral_species not in neutral_mass_dict:
            raise ValueError(f"Neutral species {neutral_species} not recognized. Available species: {list(neutral_mass_dict.keys())}")
        else:
            self.m_n_kg = neutral_mass_dict[neutral_species] * atomic_mass_unit_kg  # in kg
            self.neutral_species = neutral_species
        
        if neutral_mass_kg is not None:
            self.m_n_kg = neutral_mass_kg
            self.neutral_species = "Custom"

        if Te_K is None and Te_eV is not None:
            self.Te_K = Te_eV * eV_to_K
        else:
            self.Te_K = Te_K

        if Ti_K is None and Ti_eV is not None:
            self.Ti_K = Ti_eV * eV_to_K
        else:
            self.Ti_K = Ti_K
        
        if Tphe_K is None and Tphe_eV is not None:
            self.Tphe_K = Tphe_eV * eV_to_K
        else:
            self.Tphe_K = Tphe_K

        

        self.ne0 = ne0
        self.ni0 = ni0

        self.mass_ratio = mass_ratio
        self.photoemission_current_density = photoemission_current_density
        self.nn0 = nn0
        self.nphe0 = nphe0
        self.Tn_K = Tn_K
        self.neutral_delta = neutral_delta

        self.fields_npz_path = fields_npz_path

        if verbose:
            print(self)
    

    def __str__(self):
        s = f"Plasma Params: \n"
        s += f"ne0={self.ne0:.2e} m^-3 \n"
        s += f"ni0={self.ni0:.2e} m^-3 \n"
        s += f"Te={self.Te_K:.2f} K \n"
        s += f"Ti={self.Ti_K:.2f} K \n"
        s += f"mass_ratio={self.mass_ratio:.2f} \n"
        s += f"photoemission_current_density={self.photoemission_current_density:.2e} A/m^2 \n"
        s += f"nn0={self.nn0:.2e} m^-3 \n"
        s += f"nphe0={self.nphe0:.2e} m^-3 \n"
        s += f"Tn={self.Tn_K:.2f} K \n"
        s += f"neutral_species={self.neutral_species} \n"
        s += f"neutral_mass={self.m_n_kg:.2e} kg \n"
        s += f"neutral_delta={self.neutral_delta:.2f} \n"
        return s