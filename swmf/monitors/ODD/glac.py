from typing import List, Dict
import numpy as np

# Define the constant to convert from NM to meters
NM_TO_KM = 1.852

# Define the generic landing approach cone (SI units)
GENERIC_LANDING_APPROACH_CONE = {
    'ATD': np.array([0.08, 3]) * NM_TO_KM * 1000,  # meters
    'VPA': np.deg2rad(np.array([-2.2, -3.8])),     # radians
    'LPA': np.deg2rad(np.array([-4, 4])),          # radians
    'phi': np.deg2rad(np.array([-10, 10])),        # radians
    'theta': np.deg2rad(np.array([-8, 0])),        # radians
    'psi': np.deg2rad(np.array([-10, 10])),        # radians
}

# Metadata conversion
METADATA_NAMES = ['rwy_id', 'date', 'time', 'ATD', 'VPA', 'LPA', 'phi', 'theta', 'psi']
METADATA_TYPES = {
    'rwy_id': str,
    'date': int,
    'time': int,
    'ATD': float,
    'VPA': float,
    'LPA': float,
    'phi': float,
    'theta': float,
    'psi': float,
}


def format_meta_data(m: List[np.ndarray], m_names: List[str] = METADATA_NAMES, m_types: Dict[str, type] = METADATA_TYPES):
    """
    Format the metadata for processing.

    Args:
        m (List[np.ndarray(9)]): The metadata.
        m_names (List[str]): The metadata names.
        m_types (Dict[str, type]): The metadata types.
    """
    return {k: v.astype(m_types[k]) for k,v in zip(m_names, np.vstack(m).T)}


def comply_with_GLAC(data, glac=GENERIC_LANDING_APPROACH_CONE):
    """
    Verifies if the given metadata comply with the generic landing approach cone (GLAC).

    Args:
        metadata (Dict[str, np.ndarray]): The metadata to check
        glac     (Dict[str, np.ndarray]): The generic landing approach cone

    Returns:
        (np.ndarray) Binary array; 1 = OK, 0 = NOT OK
    """
    return np.logical_and.reduce(
        [glac[k].min() <= data[k] for k in glac.keys()] + [data[k] <= glac[k].max() for k in glac.keys()], axis=0
    )


class ODD_GLAC_checker:
    """
    Monitor for ODD based on the Generic Landing Approach Cone (only).
    """

    def __init__(self, glac: Dict[str, np.ndarray] = GENERIC_LANDING_APPROACH_CONE):
        self.glac = glac

    def __call__(self, data: Dict[str, np.ndarray]):
        return self.predict(data)

    def predict(self, data):
        """
        Check the given metadata's compliance with the ODD (here, GLAC).

        Args:
            metadata (Dict[str, np.ndarray]): The metadata to check
            glac     (Dict[str, np.ndarray]): The generic landing approach cone

        Returns:
            (bool) Binary array; 1 = out-of-ODD, 0 = in-ODD.
        """
        return np.logical_not(comply_with_GLAC(data=data, glac=self.glac))

        # return np.logical_or.reduce([
        #     glac['ATD'].min()   > metadata['ATD']  , metadata['ATD']   > glac['ATD'].max(),
        #     glac['VPA'].min()   > metadata['VPA']  , metadata['VPA']   > glac['VPA'].max(),
        #     glac['LPA'].min()   > metadata['LPA']  , metadata['LPA']   > glac['LPA'].max(),
        #     glac['phi'].min()   > metadata['phi']  , metadata['phi']   > glac['phi'].max(),
        #     glac['theta'].min() > metadata['theta'], metadata['theta'] > glac['theta'].max(),
        #     # glac['psi'].min()   >= metadata['psi']  , metadata['psi']   >= glac['psi'].max(),  # yaw angle, not used for now...
        # ], axis=0)
