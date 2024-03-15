from pathlib import Path
from omegaconf import OmegaConf
from .DECA import DecaModule
from .utils import locate_checkpoint, hack_paths, replace_asset_dirs


def load_deca( conf, stage, mode, relative_to_path=None, replace_root_path=None):
    if stage is not None:
        cfg = conf[stage]
    else:
        cfg = conf
    if relative_to_path is not None and replace_root_path is not None:
        cfg = hack_paths(cfg, replace_root_path=replace_root_path, relative_to_path=relative_to_path)
    cfg.model.resume_training = False
    checkpoint = locate_checkpoint(cfg, replace_root_path, relative_to_path, mode=mode)

    checkpoint_kwargs = {
        "model_params": cfg.model,
        "learning_params": cfg.learning,
        "inout_params": cfg.inout,
        "stage_name": "testing",
    }
    deca = DecaModule.load_from_checkpoint(checkpoint_path=checkpoint, strict=False, **checkpoint_kwargs)
    return deca


def load_model(path_to_models, run_name, stage, relative_to_path=None, replace_root_path=None, mode='best'):
    run_path = Path(path_to_models) / run_name
    with open(Path(run_path) / "cfg.yaml", "r") as f:
        conf = OmegaConf.load(f)

    conf = replace_asset_dirs(conf, Path(path_to_models) / run_name)
    conf.coarse.checkpoint_dir = str(Path(path_to_models) / run_name / "coarse" / "checkpoints")
    conf.coarse.full_run_dir = str(Path(path_to_models) / run_name / "coarse" )
    conf.coarse.output_dir = str(Path(path_to_models) )
    conf.detail.checkpoint_dir = str(Path(path_to_models) / run_name / "detail" / "checkpoints")
    conf.detail.full_run_dir = str(Path(path_to_models) / run_name / "detail" )
    conf.detail.output_dir = str(Path(path_to_models) )
    deca = load_deca(conf, stage, mode, relative_to_path, replace_root_path)
    return deca, conf