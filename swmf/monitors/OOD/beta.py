from pathlib import Path
import h5py
import numpy as np
import scipy.stats as stats
import torch
import tqdm
from torch.utils.data import DataLoader

# Import images properties from utils.py
from .utils import IMAGE_PROPERTIES
# Import the path to save monitor from data_utils.py
from ...data_utils import PATH_TO_SAVED_MONITORS


# OOD interval bounds
DEFAULT_LOWER_QUANTILE = 0.025  # Lower quantile for alpha=0.05 
DEFAULT_UPPER_QUANTILE = 0.975  # Upper quantile for alpha=0.05


class OOD_BetaQuantile:
    """
    """

    def __init__(self):
        self._img_properties = IMAGE_PROPERTIES

        self._a = dict()  # Dict {img_prop -> Beta distrib 'a' param}
        self._b = dict()  # Dict {img_prop -> Beta distrib 'b' param}
        self._loc = dict()
        self._scale = dict()

        self.monitor_id = "OOD_BetaQuantile"
        self.dataset_id = None

    def __repr__(self):
        s = "OOD_BetaQuantile(\n"
        for k in self._a.keys():
            s += f" - {k:<12}: BetaDist(a={self._a[k]:6.3f}; b={self._b[k]:6.3f}; loc={self._loc[k]:4.2f}; scale={self._scale[k]:4.2f})\n"
        s += ")"
        return s
    
    def __call__(self, x: torch.Tensor):
        return self.predict(x)

    @staticmethod
    def _save_params(filename: Path, a, b, loc, scale):
        """
        Saves OOD monitoring params to file.
        """
        hf = h5py.File(filename, "w")
        hf.create_dataset('a', data=a)
        hf.create_dataset('b', data=b)
        hf.create_dataset('loc', data=loc)
        hf.create_dataset('scale', data=scale)
        hf.close()

    @staticmethod
    def _load_params(filename: Path):
        """
        Loads OOD monitoring params from file.
        """
        hf = h5py.File(filename, "r")
        _a = np.array(hf.get('a'))
        _b = np.array(hf.get('b'))
        _loc = np.array(hf.get('loc'))
        _scale = np.array(hf.get('scale'))
        hf.close()
        return _a, _b, _loc, _scale

    def fit(
            self,
            dataset: DataLoader,
            verbose: bool = False,
    ):
        """
        """
        # Extract dataset id
        if self.dataset_id is None and hasattr(dataset.dataset, "dataset_id"):
            self.dataset_id = dataset.dataset.dataset_id
        monitor_id = self.monitor_id
        dataset_id = self.dataset_id

        for p_id, p_fn in self._img_properties.items():
            # Try and load params if property already fitted
            filename = PATH_TO_SAVED_MONITORS / f"{monitor_id}__{dataset_id}__{p_id}.h5"
            if dataset_id is not None and filename.exists():
                # Load params
                _a, _b, _loc, _scale = self._load_params(filename=filename)
            else:
                # Compute the image property
                pv = torch.Tensor([], device='cpu')
                for X, _ in tqdm.tqdm(dataset, desc=f"Fitting Beta distribution on {p_id:<12}", disable=not verbose):
                    pv = torch.cat([pv, p_fn(X)])
                pv = pv.numpy()

                # Compute params of beta distribution
                _a, _b, _loc, _scale = stats.beta.fit(pv, floc=0, fscale=1)

            # Save params into monitor
            self._a[p_id] = _a
            self._b[p_id] = _b
            self._loc[p_id] = _loc
            self._scale[p_id] = _scale

            # Try and save params once property is fitted
            if dataset_id is not None:
                # Make sure the directory exists
                PATH_TO_SAVED_MONITORS.mkdir(exist_ok=True, parents=True)
                # Save params
                self._save_params(
                    filename=filename, 
                    a=self._a[p_id], 
                    b=self._b[p_id], 
                    loc=self._loc[p_id], 
                    scale=self._scale[p_id]
                )

    def predict(self, x, alpha=None, lower_q=None, upper_q=None, return_info=False):
        """
        """
        # Make sure quantiles are set
        if lower_q is None or upper_q is None:
            if alpha is None:
                lower_q = DEFAULT_LOWER_QUANTILE
                upper_q = DEFAULT_UPPER_QUANTILE
            else:
                lower_q = alpha / 2.0         # Take balanced cut between min and max tails
                upper_q = 1.0 - alpha / 2.0

        ood_props = {}
        is_ood_1d = {}
        
        for p_id, p_fn in self._img_properties.items():
            # Compute the image properties
            value = p_fn(x).numpy()

            # Compute the Beta quantile thresholds
            lower_t = stats.beta.ppf(lower_q, self._a[p_id], self._b[p_id], self._loc[p_id], self._scale[p_id])
            upper_t = stats.beta.ppf(upper_q, self._a[p_id], self._b[p_id], self._loc[p_id], self._scale[p_id])

            # Store results
            ood_props[p_id] = value
            is_ood_1d[p_id] = np.logical_or.reduce([lower_t > value, value > upper_t], axis=0)

        is_ood = np.vstack([z for _,z in is_ood_1d.items()])
        is_ood = np.logical_or.reduce(is_ood, axis=0)

        if return_info:
            return is_ood, ood_props, is_ood_1d
        else:
            return is_ood
    
    # def predict_on_dataset(
    #         self,
    #         dataset,
    #         verbose: bool = False,
    #         pct_to_cut = None,
    #         lb = None, 
    #         ub = None,
    #         return_info: bool = False,
    # ):
    #     """
    #     """
    #     if lb is None or ub is None:
    #         if pct_to_cut is None:
    #             lb = DEFAULT_LOWER_BOUND
    #             ub = DEFAULT_UPPER_BOUND
    #         else:
    #             lb = pct_to_cut / 2.0
    #             ub = 100.0 - lb

    #     pass


class OOD_BetaQuantile_Brightness:
    
    def __init__(self):
        self._img_properties = IMAGE_PROPERTIES

        self._a = None  # float  (Beta distrib 'a' param)
        self._b = None  # float  (Beta distrib 'b' param)
        self._loc = None
        self._scale = None

        self.monitor_id = "OOD_Beta_Brightness_Monitor"
        self.dataset_id = None

    def __repr__(self):
        s = f"OOD_Beta_Brightness_Monitor(a={self._a:6.3f}; b={self._b:6.3f}; loc={self._loc:4.2f}; scale={self._scale:4.2f})"
        return s


class OOD_BetaQuantile_Saturation:
    pass


class OOD_BetaQuantile_Entropy:
    pass


class OOD_BetaQuantile_EdgeAmount:
    pass