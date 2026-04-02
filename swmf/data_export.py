import os

from pathlib import Path
from typing import (
    List,
    Tuple,
    Union,
)

import cv2
import numpy as np
import pandas as pd
import math
import tqdm
import yaml
import shutil

from .data_utils import (
    DEFAULT_PATH_LARD_DATASETS,
    DEFAULT_PATH_LARD_EXPORT,
    DEFAULT_TRAIN_ARCHIVES,
    DEFAULT_TESTS_ARCHIVES,
    DEFAULT_LARD_EXPORT_RESOLUTION,
)


#######################################
##
##          DATA EXPORT
##
#######################################

def convert_xyxy_to_xywh(
        bbox_xs: Union[np.ndarray, List],
        bbox_ys: Union[np.ndarray, List],
        img_w: int,
        img_h: int,
) -> np.ndarray: 
    """
    Convert the bounding box form xyxy (LARD format) to xywh (YOLO format).

    Note:
        The xyxy LARD bbox is in pixels while the xywh YOLO bbox is normalized to the image shape.

    Args:
        bbox_xs (Union[np.ndarray, list]): an array or list containing x's positions of the 4 bbox vertices.
        bbox_ys (Union[np.ndarray, list]): an array or list containing y's positions of the 4 bbox vertices.
        img_w (int): the original image's width
        img_h (int): the original image's height

    Returns:
        (np.ndarray) [4,] the YOLO-formated bbox.
    """
    xs = np.clip(bbox_xs, 0., img_w) / img_w  # Clip and normalize the box X coordinates
    ys = np.clip(bbox_ys, 0., img_h) / img_h  # Clip and normalize the box Y coordinates

    x_min = float(xs.min())
    x_max = float(xs.max())
    y_min = float(ys.min())
    y_max = float(ys.max())

    w  = x_max - x_min
    h  = y_max - y_min
    cx = x_min + w / 2.
    cy = y_min + h / 2.

    bbox = np.array([cx, cy, w, h])
    return bbox


def export_one_sample(
        img_sample: Tuple[int, pd.Series],
        dst_dpath: Path,
        new_shape: Tuple[int, int],
        tasks: List[str],
) -> None:
    """
    Export one sample (image + label + metadata).

    Args:
        img_sample (Tuple[int, pd.Series]): the sample index + info.
        dst_dpath (Path): the directory path to save the sample into.
        new_shape (Tuple[int, int]): the new shape of exported sample.
        tasks (List[str]): the list of ML tasks to export the sample for.
    """
    im_indx, im_info = img_sample

    im_fpath = im_info["image_dpath"] / im_info["image"].replace('\\', '/')
    im = np.array(cv2.cvtColor(cv2.imread(im_fpath), cv2.COLOR_BGR2RGB))  # [H, W, C]
    h = im.shape[0]
    w = im.shape[1]

    # Crop watermark if necessary (top and bottom)
    watermark = im_info["watermark_height"]
    if not math.isnan(watermark):
        watermark = int(watermark)
        im = im[watermark: -watermark, :, :]

    # Size up the image and save it
    new_img_fpath = dst_dpath / 'images' / im_info['split'] / f"{im_indx:06d}.jpg"
    if not new_img_fpath.exists():
        im = cv2.resize(im, new_shape, interpolation=cv2.INTER_NEAREST)
        os.makedirs(new_img_fpath.parent, exist_ok=True)
        cv2.imwrite(new_img_fpath, cv2.cvtColor(im, cv2.COLOR_RGB2BGR))

    # Save the metadata (independent of the ML task)
    new_met_fpath = dst_dpath / "metadatas" / im_info['split'] / f"{new_img_fpath.stem}.txt"
    if not new_met_fpath.exists():
        os.makedirs(new_met_fpath.parent, exist_ok=True)
        with open(new_met_fpath, "w") as f:
            f.write(";".join(im_info.loc[['airport','rnwy_id','date','time','ATD','VPA','LPA','phi','theta','psi']].astype(str).to_list()))

    # Compute labels
    x = np.array([im_info[f"x_{k}"] for k in "ABCD"], dtype=np.float32)
    y = np.array([im_info[f"y_{k}"] for k in "ABCD"], dtype=np.float32)

    if not math.isnan(watermark):
        y -= watermark
        h -= watermark * 2
    bbox = convert_xyxy_to_xywh(x, y, w, h)

    # For object detection
    if "detect" in tasks:
        tmp_image_fpath = dst_dpath / "task_detect" / "images" / im_info['split'] / new_img_fpath.name
        tmp_label_fpath = dst_dpath / "task_detect" / "labels" / im_info['split'] / f"{new_img_fpath.stem}.txt"
        tmp_mdata_fpath = dst_dpath / "task_detect" / "metadatas" / im_info['split'] / f"{new_img_fpath.stem}.txt"

        if not tmp_label_fpath.exists():
            os.makedirs(tmp_image_fpath.parent, exist_ok=True)
            os.makedirs(tmp_label_fpath.parent, exist_ok=True)
            os.makedirs(tmp_mdata_fpath.parent, exist_ok=True)
            # Handle image
            os.symlink(new_img_fpath, tmp_image_fpath, target_is_directory=False)
            # Handle metadata    
            os.symlink(new_met_fpath, tmp_mdata_fpath, target_is_directory=False)
            # Handle label
            with open(tmp_label_fpath, "w") as f:
                f.write("%g %.6f %.6f %.6f %.6f\n" % (0, *bbox))

    # For images segmentation
    if "segment" in tasks:
        kpts = np.stack((x/w, y/h), axis=-1).reshape(-1).tolist()  # [x1,x2] & [y1,y2] => [x1,y1,x2,y2]
        tmp_image_fpath = dst_dpath / "task_segment" / "images" / im_info['split'] / new_img_fpath.name
        tmp_label_fpath = dst_dpath / "task_segment" / "labels" / im_info['split'] / f"{new_img_fpath.stem}.txt"
        tmp_mdata_fpath = dst_dpath / "task_segment" / "metadatas" / im_info['split'] / f"{new_img_fpath.stem}.txt"

        if not tmp_label_fpath.exists():
            os.makedirs(tmp_image_fpath.parent, exist_ok=True)
            os.makedirs(tmp_label_fpath.parent, exist_ok=True)
            os.makedirs(tmp_mdata_fpath.parent, exist_ok=True)
            # Handle images symlink
            os.symlink(new_img_fpath, tmp_image_fpath, target_is_directory=False)
            # Handle metadata
            os.symlink(new_met_fpath, tmp_mdata_fpath, target_is_directory=False)
            # Handle labels
            with open(tmp_label_fpath, "w") as f:
                f.write("0 " + " ".join([f'{p:.6f}' for p in kpts]) + "\n")


def export_one_split_set(
        archives: List[Tuple[str, str]],
        dataset_split: str,
        src_dpath: Path,
        dst_dpath: Path,
        new_image_shape: Tuple[int, int],
        tasks: List[str],
) -> None:
    """
    Export each sample of the given dataset split (train or test).

    Args:
        archives
        dataset_name
        src_dpath
        dst_dpath
        new_image_shape
        tasks
    """
    # Get the csv filepath from unzipped archives
    csv_fpaths = [src_dpath / zip_fname.rpartition('.')[0] / csv_fname for zip_fname, csv_fname in archives]

    # Get the data from csv files
    dfs = []
    for csv_fpath in csv_fpaths:
        dfi = pd.read_csv(csv_fpath, delimiter=";")
        dfi["image_dpath"] = csv_fpath.parent
        dfi["split"] = dataset_split
        dfs.append(dfi)
    df = pd.concat(dfs).reset_index(drop=True).reset_index(drop=False)
    df['index'] = df['index'].map(lambda x: f"{x:06d}")

    #######################################################
    ### Preproc the dataFrames (TODO: put elsewhere...) ###

    def _util_convert_runway_number(rn: str):
        """
        Util function to convert runway number into bearing angle
        """
        if not (isinstance(rn, int) or isinstance(rn, str)):
            raise TypeError("Input 'rn' must be either int or str.")
        
        if isinstance(rn, int):
            return rn
        else:
            s_type = type(rn)
            return int(s_type().join(filter(s_type.isdigit, rn)))
    
    df.rename(columns={'time': 'datetime'}, inplace=True)

    df['watermark_height'] = df['watermark_height'].fillna(0.0)
    df['rnwy_id'] = df['airport'].astype(str) + '|' + df['runway'].map(lambda x: f"{x:0>3}")
    df['date'] = df['datetime'].map(lambda x: x.split(' ')[0])
    df['time'] = df['datetime'].map(lambda x: x.split(' ')[1])
    df['ATD'] = df['along_track_distance'] * 1852            # Convert from NM to m
    df['LPA'] = np.deg2rad(df['lateral_path_angle'])         # Convert from degrees to radians
    df['VPA'] = np.deg2rad(df['vertical_path_angle']) *(-1)  # Convert from degrees to radians and opposite
    df['slant_distance'] *= 1.852   # Convert from NM to km
    df['phi'] = np.deg2rad(df['roll'])
    df['psi'] = np.deg2rad((df['yaw'] - df['runway'].map(_util_convert_runway_number)*10 + 180) % 360 - 180)
    df['theta'] = np.deg2rad(df['pitch'] - 90)
    df['X'] = -df['ATD']
    df['Y'] = +df['ATD'] * np.tan(df['LPA'])
    df['Z'] = +df['ATD'] * np.tan(df['VPA'])
    # df['u'] = (df['x_C']+df['x_D'])/2.
    # df['v'] = (df['y_C']+df['y_D'])/2.
    #######################################################
    #######################################################

    # Save csv file of exported metadata {img -> idx}
    os.makedirs(dst_dpath / "images", exist_ok=True)
    df.to_csv(dst_dpath / "images" / f"{dataset_split}_metadata.csv", sep=';', index=False, columns=["index", "image", "airport", "rnwy_id"], header=False)

    # Export each sample individually
    for s in tqdm.tqdm(df.iterrows()):
        export_one_sample(s, dst_dpath, new_image_shape, tasks)


def dataset_export(
        dataset_origin_path: Union[Path, str] = DEFAULT_PATH_LARD_DATASETS,
        dataset_export_path: Union[Path, str] = DEFAULT_PATH_LARD_EXPORT,
        train_archives: List[Tuple[str, str]] = DEFAULT_TRAIN_ARCHIVES,
        tests_archives: List[Tuple[str, str]] = DEFAULT_TESTS_ARCHIVES,
        imgsz: Tuple[int, int] = DEFAULT_LARD_EXPORT_RESOLUTION,
        tasks: List[str] = ["detect"],
        override_data: bool = False,
        verbose: bool = False,
) -> Path:
    """
    Launch the export of the datasets. Give a name to it, new image shape et specific to ML task.

    Args:
        dataset_path (Path): the destination path to save the exported datasets.
        dataset_name (str): the name given to the exported datasets.
        imgsz (Tuple[int, int]): the image resolution to export (WxH).
        tasks (List[str]): the ML tasks for which to export the datasets.
        override_data (bool): flag to allow for data override in destination path.

    Returns:
        (Path) the path to exported datasets.
    """
    if not isinstance(dataset_origin_path, Path):
        dataset_origin_path = Path(dataset_origin_path).resolve()
    if not isinstance(dataset_export_path, Path):
        dataset_export_path = Path(dataset_export_path).resolve()

    dataset_export_name = f"lard_{imgsz[0]}x{imgsz[1]}"
    dataset_export_full_path = dataset_export_path / dataset_export_name

    def _make_yaml_file():
        for task in tasks:
            d = {
                'path': (dataset_export_full_path / f"task_{task}").as_posix(),
                'train': "images/train",
                'valid': "",
                'test': "images/test",
                'nc': 1,
                'names': {0: "runway"},
            }
            with open((dataset_export_full_path / f"task_{task}" / "data.yaml").as_posix(), "w") as f:
                yaml.dump(d, f, sort_keys=False)

    if dataset_export_full_path.exists() and not override_data:
        print(f"Destination path already exists ({dataset_export_full_path.as_posix()}). No action done.")
        return dataset_export_full_path
    else:
        shutil.rmtree(dataset_export_full_path)
    
    os.makedirs(dataset_export_full_path, exist_ok=True)

    if verbose:
        print("Exporting train set.", flush=True, end=" ")
    export_one_split_set(train_archives, "train", dataset_origin_path / "LARD_train", dataset_export_full_path, imgsz, tasks)
    if verbose:
        print("Done.", flush=True)

    if verbose:
        print("Exporting test set.", flush=True, end=" ")
    export_one_split_set(tests_archives, "test" , dataset_origin_path / "LARD_test" , dataset_export_full_path, imgsz, tasks)
    if verbose:
        print("Done.", flush=True)

    if verbose:
        print("Creating dataset YAML file.", flush=True, end=" ")
    _make_yaml_file()
    if verbose:
        print("Done.")

    return dataset_export_full_path
