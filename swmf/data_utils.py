from pathlib import Path


### LARD download ###
DEFAULT_PATH_LARD_ARCHIVES = "../LARD_dataset"
DEFAULT_PATH_LARD_DATASETS = "../LARD_dataset"

### LARD train datasets (for download and export) ###
DEFAULT_TRAIN_ARCHIVES = [
    ("LARD_train_BIRK_LFST.zip", "LARD_train_BIRK_LFST.csv"),
    ("LARD_train_DAAG_DIAP.zip", "LARD_train_DAAG_DIAP.csv"),
    ("LARD_train_KMSY.zip", "LARD_train_KMSY.csv"),
    ("LARD_train_LFMP_LFPO.zip", "LARD_train_LFMP_LFPO.csv"),
    ("LARD_train_LFQQ.zip", "LARD_train_LFQQ.csv"),
    ("LARD_train_LPPT_SRLI.zip", "LARD_train_LPPT_SRLI.csv"),
    ("LARD_train_VABB.zip", "LARD_train_VABB.csv"),
]

### LARD tests datasets (for download and export) ###
DEFAULT_TESTS_ARCHIVES = [
    ("LARD_test_synth.zip", "LARD_test_synth.csv"),
]

### URL to download LARD
DEFAULT_URL_LARD = "https://share.deel.ai/s/H4iLKRmLkdBWqSt/download?path=%2Flard%2F1.0.0&files="

### LARD export path ###
DEFAULT_PATH_LARD_EXPORT = "data/datasets"

### LARD export resolution ###
DEFAULT_LARD_EXPORT_RESOLUTION = (512, 512)  # WxH

### Path to saved data about models, monitors, features
PATH_TO_SAVED_MODELS = Path("./data/models").resolve()
PATH_TO_SAVED_MONITORS = Path("./data/monitors").resolve()
PATH_TO_SAVED_FEATURES = Path("./data/features").resolve()

### Path to runway database
PATH_TO_RUNWAY_DATABASE = Path("./data/runways_database.json").resolve()