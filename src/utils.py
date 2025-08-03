import configparser
import pandas as pd

def read_seqinfo(path):
    config = configparser.ConfigParser()
    config.read(f"{path}/seqinfo.ini")
    info = {
        "imDir": config.get("Sequence", "imDir"),
        "seqLength": config.getint("Sequence", "seqLength"),
        "frameRate": config.getint("Sequence", "frameRate"),
        "imWidth": config.getint("Sequence", "imWidth"),
        "imHeight": config.getint("Sequence", "imHeight"),
        "seqName": config.get("Sequence", "name"),
        "imExt": config.get("Sequence", "imExt", fallback=".jpg")
    }
    print(f"Read seqinfo from {path}: {info}")
    return info

def load_gt(path):
    try:
        gt = pd.read_csv(path, names=["frame", "id", "x", "y", "w", "h", "conf", "class", "visibility"])
        print(f"Loaded GT from {path} with shape: {gt.shape}")
    except pd.errors.ParserError:
        gt = pd.read_csv(path, names=["frame", "id", "x", "y", "w", "h"])
        gt["conf"], gt["class"], gt["visibility"] = 1.0, -1, -1
        print(f"Loaded GT from {path} with fallback shape: {gt.shape}")
    return gt