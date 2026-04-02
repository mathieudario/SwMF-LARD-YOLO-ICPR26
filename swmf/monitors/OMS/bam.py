from pathlib import Path
from typing import (
    Callable,
    List,
    Union,
)

# Modules to manage files I/O
import h5py
import pickle

from collections import OrderedDict
import tqdm
import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader
from sklearn.cluster import KMeans
from sklearn.cluster import MiniBatchKMeans
from kneed import KneeLocator

from ...data import get_dataset_id_from_dataset
from ...data import get_image_from_batch, get_label_from_batch
from ...data_utils import PATH_TO_SAVED_MONITORS
from ...data_utils import PATH_TO_SAVED_FEATURES
from ...metrics import match_predictions
from ...models.yolo_utils import get_model_id_from_model
from ...models.yolo_utils import postproc_yolo_outputs, postproc_yolo_targets


class OMS_BAM_YOLO_logits:
    """
    Implemented from: https://dumas.ccsd.cnrs.fr/IMAG/hal-04749451v1
    """

    def __init__(self, model, density = "auto", n_boxes = "auto"):
        
        if model is None:
            raise ValueError("Missing required argument 'model' to initialize.")
        
        ### GENERAL ATTRIBUTES ###
        self.monitor_id = "OMS_BAM_YOLO_logits"
        self.model_id = get_model_id_from_model(model)
        
        self.model = model.eval();  # Keep given model in eval mode !
        ##########################

        ### HYPERPARAMETERS ###
        self.decision_thresh = None
        self.boxes = dict()
        self.classes = None

        self._density = "auto" if (not np.issubdtype(type(density), np.integer) and density != "auto") else density
        self._n_boxes = "auto" if (not np.issubdtype(type(n_boxes), np.integer) and n_boxes != "auto") else n_boxes
        #######################

        # Add specification to monitor's name
        if self._density != "auto":
            self.monitor_id += f"_d{self._density}"
        else:
            if self._n_boxes != "auto":
                self.monitor_id += f"_n{self._n_boxes}"
            else:
                self.monitor_id += f"_auto" 

        ### FEATURE TRACKER ###
        self._hooked_logits_ = [None]

        self.prepare_extract()
        #######################

    def clean_hooks(self):
        """"""
        def _clean_hooks(m: nn.Module):
            for _, child in m._modules.items():
                if child is not None:
                    if hasattr(child, "_forward_hooks"):
                        child._forward_hooks = OrderedDict()
                    _clean_hooks(child)
        
        return _clean_hooks(self.model)

    @staticmethod
    def get_model_layer(model, layer_id):
        if isinstance(layer_id,int):
            if isinstance(model, nn.Sequential):
                layer = model[layer_id]
            else:
                layer = list(model.named_modules())[layer_id][1]
        else:
            layer_id = list(dict(model.named_modules()).keys()).index(layer_id)
            layer = list(model.named_modules())[layer_id][1]
        
        return layer

    def prepare_extract(self):
        """"""
        # Clean previous hooks
        self.clean_hooks()

        def hook(m, i, o):
            self._hooked_logits_[0] = o
        
        # Set new logits hook
        model_layer_to_hook = dict(self.model.named_modules())["model.model.24"]  # Fixed for now (see YOLO architecture)
        model_layer_to_hook.register_forward_hook(hook)
        
    @staticmethod
    def _load_features(filename: str):
        hf = h5py.File(filename, 'r')
        ff = torch.tensor(np.array(hf.get("features")))
        hf.close()
        return ff
    
    @staticmethod
    def _save_features(filename: str, data: torch.Tensor):
        hf = h5py.File(filename, 'w')
        hf.create_dataset("features", data=data)
        hf.close()

    def _extract_aligned_feature(self, y_pred, return_info=True):
        """
        """
        if not self._hooked_logits_:
            raise ValueError("No logits have been extracted. Run model first.")

        logits = self._hooked_logits_[0]
        # Logits:
        # - logits[0]. shape: (bs, 5, 5376)
        # - logits[1].
        #   - logits[1][0]. shape: (bs, 65, 64, 64)
        #   - logits[1][1]. shape: (bs, 65, 32, 32)
        #   - logits[1][2]. shape: (bs, 65, 16, 16)

        y_pred_bboxes = y_pred[:, 2:]
        logits_bboxes = np.zeros((y_pred.shape[0], logits[0].shape[1]))
        logits_values = np.zeros((y_pred.shape[0], logits[1][0].shape[1]))

        for i in range(logits[0].shape[0]):
            j = torch.eq(y_pred[:, 0], i).flatten().to(bool)
            
            if j.sum() == 0:
                continue # No pred in image 'i'

            # Extract bboxes of final predictions
            y_pred_bboxes_i = y_pred[j, 2:] # shape: (num_preds_for_img_i, 5) often (1, 5)
            # Extract bboxes out-of logits layer
            logits_bboxes_i = logits[0][i, :5, :].permute(1, 0)
            # Extract values out-of logits layer
            logits_values_i = torch.cat([lv[i].view(65, -1) for lv in logits[1]], dim=1).permute(1, 0)
            
            # Transform to numpy
            y_pred_bboxes_i = y_pred_bboxes_i.detach().cpu().numpy() # shape: (N, 5)
            logits_bboxes_i = logits_bboxes_i.detach().cpu().numpy() # shape: (M, 5)
            logits_values_i = logits_values_i.detach().cpu().numpy() # shape: (M, 65)

            # Temporary reshape (for distance calculation)
            y_pred_bboxes_reshaped = y_pred_bboxes_i.reshape(-1, 1, 5) # shape: (N, 1, 5)
            logits_bboxes_reshaped = logits_bboxes_i.reshape( 1,-1, 5) # shape: (1, M, 5)

            # Compute distances
            distances = np.sqrt(np.sum((y_pred_bboxes_reshaped - logits_bboxes_reshaped)**2, axis=2)) # shape: (N, M)
            d_indices = np.argmin(distances, axis=1) # shape: (N,)

            # Extract logits values according to distance indices
            logits_bboxes_filtered = logits_bboxes_i[d_indices, :] # shape: (N, 5)
            logits_values_filtered = logits_values_i[d_indices, :] # shape: (N, 65)
            
            # Store inside common arrays
            logits_bboxes[j.numpy()] = logits_bboxes_filtered
            logits_values[j.numpy()] = logits_values_filtered

        # All returned values are numpy arrays
        if return_info:
            return logits_values, logits_bboxes, y_pred_bboxes
        else:
            return logits_values

    def extract_logits_tensor(self, x, verbose=False, return_info=True, **ml_kwargs): 
        """
        """
        # Make sure to work on same device (CPU, CUDA)
        x = x.to(self.model.device)

        # Execute model inference
        y_pred = postproc_yolo_outputs(self.model(x, verbose=False, **ml_kwargs))

        # Release GPU memory
        torch.cuda.empty_cache()

        # Extract logits
        logits = self._extract_aligned_feature(y_pred, return_info=False)
        # logits = torch.tensor(logits)

        # Checkup that logits and y_pred have same dim
        assert logits.shape[0] == y_pred.shape[0]

        if return_info:
            return logits, y_pred
        else:
            return logits

    def extract_logits(self, dataset, verbose=False, override=False, return_info=True, **ml_kwargs): 
        """
        """
        # Init the research flags
        has_to_get_logits = True
        has_to_get_y_pred = True
        has_to_get_y_true = True

        # Check if the logits for this very case were not extracted before
        if self.dataset_id is not None and self.model_id is not None:
            # Define the filepath to features
            logits_fpath = PATH_TO_SAVED_FEATURES / f"{self.dataset_id}__{self.model_id}__logits.h5"
            y_pred_fpath = PATH_TO_SAVED_FEATURES / f"{self.dataset_id}__{self.model_id}__preds.h5"
            y_true_fpath = PATH_TO_SAVED_FEATURES / f"{self.dataset_id}__{self.model_id}__trues.h5"

            if not override and logits_fpath.exists():
                logits = self._load_features(filename=logits_fpath)
                has_to_get_logits = False
            else:
                logits = None

            if not override and y_pred_fpath.exists():
                y_pred = self._load_features(filename=y_pred_fpath)
                has_to_get_y_pred = False
            else:
                y_pred = None

            if not override and y_true_fpath.exists():
                y_true = self._load_features(filename=y_true_fpath)
                has_to_get_y_true = False
            else:
                y_true = None

        # If need to extract missing features
        if has_to_get_logits or has_to_get_y_pred or has_to_get_y_true:
            # Prepare the logits extractor
            self.prepare_extract()

            # Init target flag
            has_targets = False

            if isinstance(dataset, DataLoader):
                n_img = 0
                batch = next(iter(dataset))
                has_targets = (isinstance(batch, (list, tuple)) and len(batch) > 1)

                for batch in tqdm.tqdm(dataset, desc="Extracting logits...", disable=not verbose):
                    x = get_image_from_batch(batch)

                    # Extract the logits and predictions
                    if has_to_get_logits or has_to_get_y_pred:
                        logits_batch, y_pred_batch = self.extract_logits_tensor(x, verbose=verbose, **ml_kwargs)
                    # Extract the ground truths
                    if has_to_get_y_true and has_targets:
                        y_true_batch = postproc_yolo_targets(get_label_from_batch(batch, return_bbox=True))

                    # Concatenate to existing output tensors
                    if has_to_get_logits:
                        logits_batch = torch.tensor(logits_batch)
                        logits = (logits_batch if logits is None else torch.cat((logits, logits_batch), dim=0))
                    if has_to_get_y_pred:
                        y_pred_batch[:, 0] += n_img
                        y_pred = (y_pred_batch if y_pred is None else torch.cat((y_pred, y_pred_batch), dim=0))
                    if has_to_get_y_true and has_targets:
                        y_true_batch[:, 0] += n_img
                        y_true = (y_true_batch if y_true is None else torch.cat((y_true, y_true_batch), dim=0))
                    
                    # HOTFIX: artificially add the total num of img treated so far
                    n_img += len(batch)
            else:
                raise NotImplementedError("Wait. Not yet.")
            
            # Save features if possible
            if self.dataset_id is not None and self.model_id is not None:
                PATH_TO_SAVED_FEATURES.mkdir(exist_ok=True, parents=True)

                if has_to_get_logits:
                    self._save_features(filename=logits_fpath, data=logits)
                if has_to_get_y_pred:
                    self._save_features(filename=y_pred_fpath, data=y_pred)
                if has_to_get_y_true and has_targets:
                    self._save_features(filename=y_true_fpath, data=y_true)

        if return_info:
            return logits, y_pred, y_true
        else:
            return logits

    def fit(self, dataset, verbose=True, override=False, keep_only_good=True, iou_thresh=0.7, iou_method="CIOU", box_format="xyxy", **ml_kwargs):
        """
        """
        self.dataset_id = get_dataset_id_from_dataset(dataset=dataset)

        # Init train flag
        has_to_train = True

        # Check if the monitor has not been trained on this very scenario before
        if self.dataset_id is not None and self.model_id is not None:
            filename = PATH_TO_SAVED_MONITORS / f"{self.monitor_id}__{self.dataset_id}__{self.model_id}.pkl"

            if not override and filename.exists():
                # Extract saved monitor params
                self.boxes = self._load_params(filename=filename)
                # Compute classes ids (if needed)
                self.classes = torch.arange(len(self.boxes)) if self.classes is None else self.classes
                has_to_train = False
        
        # Otherwise, needs to train
        if has_to_train:
            # Extract the logits, predictions and ground truths
            logits, y_pred, y_true = self.extract_logits(dataset=dataset, verbose=verbose, return_info=True, **ml_kwargs)
            # logits --> [num_preds, 65]
            # y_pred --> [num_preds,  7]
            # y_true --> [num_trues,  6]

            # If only correct outputs are kept
            if keep_only_good:
                tpfp, _ = match_predictions(y_pred, y_true, iou_thresh=iou_thresh, iou_method=iou_method, box_format=box_format)
                indx_ok = (tpfp != -1)
                
                logits = logits[indx_ok]
                y_pred = y_pred[indx_ok]

            # Compute classes ids (if needed)
            self.classes = torch.unique(y_true[:, 1].flatten()) if self.classes is None else self.classes

            # Compute box-abstraction
            if verbose:
                print(f"Fitting monitor parameters...")
            self.boxes = [Boxes() for _ in self.classes]

            for c in self.classes:
                # Compute corresponding class index
                i = torch.eq(y_pred[:, 1], int(c)).flatten()

                # Compute nb of clusters and clusters themselves
                n_clusters = self._find_hyperparameters(logits[i], verbose=verbose)
                j_clusters = KMeans(n_clusters=n_clusters, random_state=0).fit_predict(logits[i])

                if verbose:
                    print(f"n_clusters = {n_clusters}")
                
                # Compute box-abstractions for class 'c'
                for j in range(n_clusters):
                    k = (j_clusters == j)

                    min_logits = np.min(logits[i][k].numpy(), axis=0)
                    max_logits = np.max(logits[i][k].numpy(), axis=0)
                    self.boxes[int(c)].add_box(Box(min_logits, max_logits))

            # Save the computed monitor's parameters
            if self.dataset_id is not None and self.model_id is not None:
                self._save_params(filename=filename, boxes=self.boxes)

    def score_tensor(self, x, verbose=False, **ml_kwargs):
        """
        """
        # Reset hooks on tracker
        self.prepare_extract()

        # Extract logits
        logits, y_pred = self.extract_logits_tensor(x, verbose=verbose, return_info=True, **ml_kwargs)
        # logits. numpy array.  shape: (N, 65)
        # y_pred. torch tensor. shape: (N,  5)

        # Compute OMS scores
        box_scores = np.zeros(y_pred.shape[0])

        for c in self.classes:
            i = torch.eq(y_pred[:, 1], int(c)).flatten().numpy()
            box_scores[i] = self.boxes[int(c)].predict(logits[i, :])
        
        return box_scores

    # def score(self, dataset): pass

    # def predict_tensor(self, x): pass

    # def predict(self, dataset): pass

    def _find_hyperparameters(self, features, desc="Finding hyperparameters...", verbose=False):
        """
        """
        n = len(features)

        # If hyperparameters were already init
        if np.issubdtype(type(self._density), np.integer):
            return int(n / self._density)
        if np.issubdtype(type(self._n_boxes), np.integer):
            return self._n_boxes

        # Else, compute inertias for potential k-values
        k_values = range(2, int(np.sqrt(n)) + 1)
        inertias = []
        for k in tqdm.tqdm(k_values, desc=desc, disable=not verbose):
            inertias.append(MiniBatchKMeans(n_clusters=k, random_state=0).fit(features).inertia_)

        # Find the best k
        kneedle = KneeLocator(k_values, inertias, curve='convex', direction='decreasing')
        optimal = kneedle.elbow
        return optimal

    @staticmethod
    def _load_params(filename):
        pf = open(filename, "rb")
        boxes = pickle.load(pf)
        pf.close()
        return boxes

    @staticmethod
    def _save_params(filename, boxes):
        pf = open(filename, 'wb')
        pickle.dump(boxes, pf)
        pf.close()
    

class Box:

    def __init__(self, lb: np.ndarray, ub: np.ndarray):
        self.lower_bounds = lb
        self.upper_bounds = ub
        self.centers = (self.lower_bounds + self.upper_bounds) / 2
        self.distance_box = ((self.upper_bounds - self.lower_bounds) / 2) + 1e-20

    def contain(self, pts: np.ndarray):
        is_above_min = (np.min(pts - self.lower_bounds, axis=1) >= 0)
        is_below_max = (np.min(self.upper_bounds - pts, axis=1) >= 0)
        return is_above_min & is_below_max

    def predict(self, pts: np.ndarray):
        distances = np.abs(pts - self.centers)
        return np.max(distances / self.distance_box, axis=1)


class Boxes:

    def __init__(self):
        self.boxes: List[Box] = []

    def add_box(self, box: Box):
        self.boxes.append(box)

    def contain(self, pts: np.ndarray):
        return np.any([b.contain(pts) for b in self.boxes], axis=0)

    def predict(self, pts: np.ndarray):
        return np.min([b.predict(pts) for b in self.boxes], axis=0)