from pathlib import Path
from typing import (
    Callable,
    List,
    Tuple,
    Union,
)
import os
import numpy as np
import pandas as pd
import torch
import torchvision
import yaml
from torch.utils.data import (
    Dataset,
    DataLoader,
)

# Import helper to convert bbox format
from .metrics_utils import bbox_convert


SUPPORTED_TASKS = ["detect", "segment"]
SUPPORTED_MODES = ["train", "test"]
SUPPORTED_SPLIT_STRATEGIES = {
    "trainval": ["train", "valid"],
    "trainval_per_runway": ["train", "valid"],
    "trainval_per_airprt": ["train", "valid"],
}


class CustomDataset(Dataset):
    """
    """

    def __init__(
            self,
            image_dpath: Path,
            label_dpath: Path,
            image_transform: Callable = None,
            label_transform: Callable = None,
            ids: Union[List, np.ndarray] = None,
            using_metadata: bool = False,
            metadata_dpath: Path = None,
            metadata_transform: Callable = None,
            dataset_id: str = None,
    ):
        """
        Initialize the custom dataset.

        Args:
            image_dpath (Path): Path to the directory containing the samples images.
            label_dpath (Path): Path to the directory containing the samples labels.
            image_transform (Callable): Transform to apply to the images in the getter.
            label_transform (Callable): Transform to apply to the labels in the getter.
            ids (Union[List, np.ndarray]): Array-like containing the available (reachable) samples in the dataset.
            using_metadata (bool): Flag whether to load and deal with metadatas.
            metadata_dpath (Path): Path to the directory containing the samples metadata. 
            metadata_transform (Callable): Transform to apply to the metadatas in the getter.

        Note:

        """
        self.dataset_id = dataset_id

        self._image_dpath = image_dpath
        self._label_dpath = label_dpath
        self._image_transform = image_transform
        self._label_transform = label_transform

        if ids is None:
            self._ids = np.array([f.stem for f in image_dpath.glob("*")])  #  only jpg is supported now...
        else:
            self._ids = ids

        # Setup metadata if using_metadata = True
        if using_metadata:
            self._using_metadata = True
            self._metadata_dpath = metadata_dpath
            self._metadata_transform = metadata_transform
        else:
            self._using_metadata = False
    
    def __repr__(self):
        return "CustomDataset({})".format({
            'path': None,
            'name': None,
            'size': self.__len__(),
        })
    
    def __str__(self):
        return self.__repr__()

    def __len__(self):
        return len(self._ids)
    
    def __getitem__(self, idx: int):
        """
        Method to get (image, label) item from dataset at given index.
        
        Args:
            idx (int): index of the sample in the `self.ids` attribute array.
        """
        img_id = self._ids[idx]
        image_path = self._image_dpath / f"{img_id}.jpg"
        label_path = self._label_dpath / f"{img_id}.txt"

        # Load the image
        if not image_path.exists():
            raise ValueError(f"No image found at {image_path.as_posix()}")
        img = torchvision.io.decode_image(image_path.as_posix())

        # Load the label(s)
        if not label_path.exists():
            lab = []
        else:
            lab = np.genfromtxt(label_path.as_posix(), delimiter=' ', dtype=str) # [L, 5] - (class_id, cx, cy, w, h)
            lab = np.atleast_2d(lab).astype(np.float32)
        lab = torch.Tensor(lab)

        # Transform image or label if needed
        if self._image_transform is not None:
            img = self._image_transform(img)
        if self._label_transform is not None:
            lab = self._label_transform(lab)

        # If using metadata #################################
        if self._using_metadata:
            metadata_path = self._metadata_dpath / f"{img_id}.txt"

            ### TODO: move elsewhere ###
            def __convert_date(date: str) -> int:
                """Convert date (YYYY-MM-DD) to integer YYYYMMDD."""
                yy, mm, dd = date.split('-')
                return int(yy+mm+dd)
            
            def __convert_time(time: str) -> int:
                """Convert time (hh:mm:ss) to integer (number of seconds)."""
                hh, mm, ss = time.split(':')
                return int(hh)*3600+int(mm)*60+int(ss)
            ############################

            if not metadata_path.exists():
                metadata = np.array([])
            else:
                metadata = np.genfromtxt(metadata_path, delimiter=';', dtype=str)  # [L, 10] - (airport, runway, rnwy_id, time, ATD, VPA, LPA, phi, theta, psi)
                metadata[2] = __convert_date(metadata[2])                          # Convert date to int for tensor conversion
                metadata[3] = __convert_time(metadata[3])                          # Convert time to int for tensor conversion
                metadata = metadata[1:]   # Remove ["airport"], keep in string()

            if self._metadata_transform is not None:
                metadata = self._metadata_transform(metadata)

            return img, lab, metadata
        #####################################################

        return img, lab
    

def create_dataloader(
        dataset_dpath: Path = None,
        image_dpath: Path = None,
        label_dpath: Path = None,
        image_transform: Callable = None,
        label_transform: Callable = None,
        yolo_task: str = 'detect',
        data_mode: str = 'train',
        ids: Union[List, np.ndarray] = None,
        using_split: bool = False,
        split_fpath: Path = None,
        split: str = None,
        using_metadata: bool = False,
        metadata_dpath: Path = None,
        metadata_transform: Callable = None,
        **kwargs,
) -> DataLoader:
    """
    Create a torch.utils.data.DataLoader object to specifically load data from LARD custom dataset.

    Args:
        dataset_dpath (Path): The path to the root dataset directory.
        image_dpath (Path): The path to the directory containing the sample images. Either `dataset_dpath` or (`image_dpath` and `label_dpath`) should be specified. 
        label_dpath (Path): The path to the directory containing the sample labels. Either `dataset_dpath` or (`image_dpath` and `label_dpath`) should be specified. 
        yolo_task (str): The ML task. In ['detect', 'segment'].
        data_mode (str): Either 'train' or 'test'.
        ids (Union[List, np.ndarray]): List-like of the sample ids to use as dataset.
        image_transform (Callable): The transform to apply to the image in the dataset getter.
        label_transform (Callable): The transform to apply to the label in the dataset getter.
        split (str): The name of the split to use (from split file)
        split_fpath (Path): The path to the csv file that describes the split.

    Returns:
        A `torch.utils.data.DataLoader` object to iterate over the desired dataset.
    """
    # Check that at least a path is given
    if dataset_dpath is None and (image_dpath is None or label_dpath is None):
        raise ValueError("Either `dataset_dpath` or (`image_dpath` and `label_dpath`) should be specified.")
    
    # Check that task and mode are correct
    if not yolo_task in SUPPORTED_TASKS:
        raise ValueError(f"Unknown specified task {yolo_task}. Expecting one from {SUPPORTED_TASKS}")
    if not data_mode in SUPPORTED_MODES:
        raise ValueError(f"Unknown specified mode {data_mode}. Expecting one from {SUPPORTED_MODES}")

    # Ensure image_dpath and label_dpath are set
    if image_dpath is None or label_dpath is None:
        image_dpath = dataset_dpath / f"task_{yolo_task}" / "images" / data_mode
        label_dpath = dataset_dpath / f"task_{yolo_task}" / "labels" / data_mode
        
    # Ensure ids are set (only if not set...)
    if ids is None and (split is not None and split_fpath is not None):
        if data_mode == "train":
            # TODO: add checks on split and split_fpath before use
            csv = pd.read_csv(split_fpath, sep=';', dtype=str)
            ids = csv[csv['split']==split]['index'].to_numpy()
        else:
            print("Cannot split test data. Ignored.")

    # If metadata are active [TODO: Warning, think of alternative if dataset_dpath no provided]
    if using_metadata and metadata_dpath is None:
        print("Path to metadata not provided, trying to infer based on dataset path...")
        metadata_dpath = label_dpath.parent.parent / "metadatas" / data_mode
        
        # Watchdog ...
        if metadata_dpath.exists():
            print(f"Found existing metadata at {metadata_dpath.as_posix()}")
        else:
            print(f"No metadata found at {metadata_dpath.as_posix()}, fallback to no metadata.")
            using_metadata = False

    # Define the dataset object
    _d = dataset_dpath.stem if dataset_dpath is not None else label_dpath.parent.parent.parent.stem
    if data_mode == "train" and split_fpath is not None:
        _s = split_fpath.stem.split('_')[-1][0].upper()
        _s = "" if _s == "T" else _s
        dataset_id = f"{_d}_{split}_split_{_s}"
    else:
        dataset_id = f"{_d}_{data_mode}"

    dataset = CustomDataset(
        image_dpath=image_dpath,
        label_dpath=label_dpath,
        image_transform=image_transform,
        label_transform=label_transform,
        ids=ids,
        using_metadata=using_metadata,
        metadata_dpath=metadata_dpath,
        metadata_transform=metadata_transform,
        dataset_id=dataset_id
    )

    def collate_fn(batch):
        """
        Collate samples from a batch, used for variable-length labels...
        """
        images = torch.stack([elmt[0] for elmt in batch], dim=0)  # TODO: think about returning torch.Tensor instead of list, for images.
        labels = torch.stack([elmt[1] for elmt in batch], dim=0)
        return images, labels
    
    def collate_fn_metadata(batch):
        """
        Collate samples from a batch when option metadata is active.
        """
        images = torch.stack([elmt[0] for elmt in batch], dim=0)
        labels = torch.stack([elmt[1] for elmt in batch], dim=0)
        metadatas = [elmt[2] for elmt in batch]
        return images, labels, metadatas

    if using_metadata:
        dataloader = DataLoader(dataset, collate_fn=collate_fn_metadata, **kwargs)
    else:
        dataloader = DataLoader(dataset, collate_fn=collate_fn, **kwargs)
    
    return dataloader


def create_yolo_datayaml(
        dataset_dpath: Path,
        yolo_task: str,
        split_fpath: Path,
        split_dname: str,
) -> Path:
    """
    Create a 'data.yaml' file and the associated paths to train, val and test
    files that are required by YOLO ultralytics routines.

    Args:
        dataset_dpath (Path): The path to the dataset.
        yolo_task (str): The task of yolo, in ['detect', 'segment'].
        split_fpath (Path): The path to the csv file describing the split.
        split_dname (str): The name of the split folder to create (in which to save all generated files).
    
        TODO: add option to override / not override
    Returns:
        (Path) The path to generated split directory.
    """
    # Get images directory
    origin_dpath = dataset_dpath / f"task_{yolo_task}"
    if not origin_dpath.exists():
        raise ValueError(f"No dataset found at {origin_dpath.as_posix()}")

    # Set output directory
    output_dpath = dataset_dpath / f"task_{yolo_task}" / f"{split_dname}"
    output_dpath.mkdir(exist_ok=True)

    df = pd.read_csv(split_fpath, sep=';', dtype=str)
    train_ids = df[df['split']=='train']['index']
    valid_ids = df[df['split']!='train']['index']

    train_txt_fpath = output_dpath / "train.txt"
    valid_txt_fpath = output_dpath / "valid.txt"

    # Create the valid/ symlink directory
    img_train_dpath = origin_dpath / "images" / "train"
    img_valid_dpath = img_train_dpath.parent / "valid"
    if img_valid_dpath.exists() or img_valid_dpath.is_symlink():
        img_valid_dpath.unlink()
    os.symlink(img_train_dpath, img_valid_dpath)

    lab_train_dpath = origin_dpath / "labels" / "train"
    lab_valid_dpath = lab_train_dpath.parent / "valid"
    if lab_valid_dpath.exists() or lab_valid_dpath.is_symlink():
        lab_valid_dpath.unlink()
    os.symlink(lab_train_dpath, lab_valid_dpath)

    # Create the train.txt and valid.txt files
    train_ids.map(lambda x: (img_train_dpath / f"{x}.jpg")).to_csv(train_txt_fpath, index=False, header=False)
    valid_ids.map(lambda x: (img_valid_dpath / f"{x}.jpg")).to_csv(valid_txt_fpath, index=False, header=False)

    # Create the data.yaml files
    d = {
        'path': origin_dpath.as_posix(),
        'train': train_txt_fpath.as_posix(),
        'val': valid_txt_fpath.as_posix(),
        'test': "images/test",
        'nc': 1,
        'names': {0: "runway"}
    }
    with open(output_dpath / "data.yaml", "w") as f:
        yaml.dump(d, f, sort_keys=False)

    d['val'] = "images/test"
    with open(output_dpath / "data_test.yaml", "w") as f:
        yaml.dump(d, f, sort_keys=False)

    return output_dpath


def default_lab_transform(imgsz: Tuple[int, int]):
    """
    DEFAULT label transform, make bouding box to x1y1x2y2 format
    """
    h, w = imgsz

    def __transform(x: torch.Tensor):
        x[:, 1] *= w
        x[:, 2] *= h
        x[:, 3] *= w
        x[:, 4] *= h
        x[:, 1:] = bbox_convert(x[:, 1:], i_fmt="cxcywh", o_fmt="xyxy")
        return x
    
    return __transform


def default_img_transform(imgsz: Tuple[int, int]):
    """
    DEFAULT image transform, make image into float (0, 1) values
    """
    return torchvision.transforms.Compose([
        torchvision.transforms.Lambda(lambda x: torch.clamp(x.to(torch.float32) / 255.0, min=0.0+torch.finfo(torch.float32).eps, max=1.0-torch.finfo(torch.float32).eps)),
        torchvision.transforms.Resize(imgsz),                                                                 
    ])


def default_metadata_transform():
    pass


def get_dataset_id_from_dataset(dataset) -> str:
    """
    """
    dataset_id = None
    if isinstance(dataset, DataLoader) and hasattr(dataset.dataset, 'dataset_id'):
        dataset_id = dataset.dataset.dataset_id
    if isinstance(dataset, DataLoader) and hasattr(dataset.dataset, 'id'):
        dataset_id = dataset.dataset.id
    return dataset_id


def get_image_from_batch(dataset_elem) -> torch.Tensor:
    """
    """
    if isinstance(dataset_elem, tuple) or isinstance(dataset_elem, list):
        return dataset_elem[0]
    if isinstance(dataset_elem, dict) and "x" in dataset_elem:
        return dataset_elem['x']
    if isinstance(dataset_elem, dict) and "X" in dataset_elem:
        return dataset_elem['X']
    if isinstance(dataset_elem, dict) and "image" in dataset_elem:
        return dataset_elem['image']
    # Default to torch.Tensor
    return dataset_elem


def get_label_from_batch(dataset_elem, return_bbox=False) -> torch.Tensor:
    """
    """
    if return_bbox:
        return dataset_elem[1][:, :, :]
    else:
        return dataset_elem[1][:, :, 0]