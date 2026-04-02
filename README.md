# SwMF for LARD monitoring

This repository presents all the necessary code to reproduce the experiments for the ICPR 2026 paper: **Unifying Runtime Monitoring Approaches for Safety-Critical Machine Learning: Application to Vision-Based Landing**

<div align="center">
<img src="yolo-pred.jpg" 
     alt="Predictions results - YOLOv5 - demo"
     align="center"
     width="40%"/>
<br>

<div style="text-align: center; width:80%">

*FIGURE: Prediction of a YOLOv5 model on an image from the LARDv1 dataset (test split)*
</div>

</div>

## Directory structure

```
.
├── data/                       # Directory that contains datasets, models, ...
|   ├── runways_database_all.json   # Database of all runway points (from LARD)
|   └── ...
├── notebooks/                  # Jupyter notebooks for experiments and pipeline
│   ├── data-download.ipynb         # Download datasets
│   ├── data-exploration.ipynb      # Explore datasets
│   ├── data-export.ipynb           # Export datasets (train/valid/test)
│   ├── ml-yolo.ipynb               # Model training notebook
│   ├── ODD.ipynb                   # Exploration of ODD verification
│   ├── OOD.ipynb                   # Exploration of OOD verification
│   └── threats-generation.ipynb    # Corrupted data notebook
├── results/                    # Results, logs, and figures produced by experiments
|   ├── lard_512x512                # Original dataset (download + export)
|   |   └── data_analysis               # ODD and OOD analyses
|   └── lard_512x512_ICPR2026       # Filtered dataset (original + filter out-of-ODD)
|       └── data_analysis               # ODD and OOD analyses
├── swmf/                       # SwMF (Swiss cheese Monitoring Framework) core code
│   ├── models/                     # Model architectures and utils
│   ├── monitors/                   # Monitoring methods
|   |   ├── ODD/                        # ODD monitors folder
|   |   ├── OMS/                        # OMS monitors folder
|   |   └── OOD/                        # OOD monitors folder
│   └── ...                         # Additional SwMF modules
├── icpr2026_experiment.ipynb   # ICPR main experiment notebook
├── requirements.txt            # (Optional) Python dependencies
├── README.md                   # This file
└── ...
```

## Requirements

- `Python 3.10` (recommended)
- `uv` package manager (recommended)

*Note, the project has been developed using the `uv` package manager. Yet, using `pip` or `conda` is also possible.*

### Installation (`uv`)

#### 0. Install `uv`.

First, make sure to have the `uv` package manager installed. You can refer to [this](https://docs.astral.sh/uv/getting-started/installation/#installation-methods) page for details about download or run the commands below.

- On Linux/Mac (recommended):
```sh
curl -LsSf https://astral.sh/uv/install.sh | sh
```
- On Windows:
```powershell
powershell -c "irm https://astral.sh/uv/install.ps1 | iex"
```
- With `pip`:
```sh
pip install uv
```

#### 1. Create a virtual environment and activate it.

```sh
uv venv (--python 3.10)
source .venv/bin/activate
```

#### 2. Install the dependencies.

```sh
uv pip install -r pyproject.toml
```

#### 3. Verify the installation.

```sh
uv pip list
```

### Alternative installation

Create a virtual environment and install the dependencies using the `requirements.txt` file and your favourite package manager (pip, poetry, conda...).

## Usage

The experiment presented in the ICPR 2026 paper can be run in the [`icpr2026_experiment.ipynb`](./icpr2026_experiment.ipynb) notebook. Follow the steps below to properly setup and run the experiment.

### 1. Data preparation

- **Dataset download**.
    
    Open and run [`notebooks/data-download.ipynb`](./notebooks/data-download.ipynb) to download the [LARD](https://github.com/deel-ai/LARD/tree/LARD_V1) (v1) dataset. 
    
    ***Warning**: the automated download does not work anymore, manually download is necessary (see instructions in the notebook).*

- **Dataset export**. 
    
    Open and run [`notebooks/data-export.ipynb`](./notebooks/data-export.ipynb) to export the dataset to the proper format for [YOLO](https://docs.ultralytics.com/fr/models/yolov5/). Also filter data to only keep the images compliant with the ODD.

- **Model training**.
    
    Open and run [`notebooks/ml-yolo.ipynb`](./notebooks/ml-yolo.ipynb) to train the YOLO object detection model that will be monitored at runtime. 

    *Note that all steps are deterministic, with a fixed random seed set in each notebook where necessary. For different settings, modify the seed inside the notebooks.*

### 2. Experiment

- **Main experiment**. 
    
    Run the [`icpr2026_experiment.ipynb`](./icpr2026_experiment.ipynb) notebook. This notebook integrates data loading from exported datasets, model inference, monitors training and testing, results computation and saving.

    *Note that all steps are deterministic, with a fixed random seed set in each notebook where necessary. For different settings, modify the seed inside the notebooks.*

### 3. Results

- Results (standard ML object detection metrics and safety-oriented metrics) will be found in the [`results/`](./results/) directory upon completion.

## License

This project is licensed under the MIT license. See the [LICENSE](./LICENSE) file for more details.

## Citation

TODO

---
For any questions, please contact <mathieu.dario@laas.fr>