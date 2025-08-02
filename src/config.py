import yaml

DEFAULTS = {
    
    "device": "cuda",
    "run_name": "Test",
    
    "feature":{
        "feature_type": "vals",
        "dataset_type": "images",
        "ds_rate": 1,
    },
    
    "io": { 
        "clip_info": None, # must overwrite
        "output_folder": None, # must overwrite
    },
    
    "emoca": {
        "path_to_models": "./assets/EMOCA/models",
        "model_name": "EMOCA_v2_lr_mse_20",
    },
    
    "face_detection":{
        "detect": False,
        "crop_size": 224,
        "threshold": 0.5,
        "iou_threshold": 0.5,
    }
}


def _merge(src, dst):
    for k, v in src.items():
        if k in dst:
            if isinstance(v, dict):
                _merge(src[k], dst[k])
        else:
            dst[k] = v


def load_default_config():
    config = DEFAULTS
    return config


def load_config(config_file, defaults=DEFAULTS):
    with open(config_file, "r") as fd:
        config = yaml.load(fd, Loader=yaml.FullLoader)
    _merge(defaults, config)
    if config['io']['output_folder'] == None:
        raise ValueError('Please specify the output folder and the path to the clip info file.')
    return config