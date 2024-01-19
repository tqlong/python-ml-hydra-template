from omegaconf import DictConfig, OmegaConf
import hydra
import os
import sys
import pyrootutils
import logging
logging.basicConfig(level=logging.INFO)


@hydra.main(version_base=None, config_path="../configs", config_name="infer")
def infer(cfg: DictConfig) -> None:
    print(OmegaConf.to_yaml(cfg))


def setup_path():
    # find absolute root path (searches for directory containing .project-root file)
    # search starts from current file and recursively goes over parent directories
    # returns pathlib object
    path = pyrootutils.find_root(
        search_from=__file__, indicator=".project-root")

    # take advantage of the pathlib syntax
    src_dir = path / "src"
    logging.info(f"root={path} src={src_dir}")
    # assert data_dir.exists(), f"path doesn't exist: {data_dir}"

    # set root directory
    pyrootutils.set_root(
        path=path,  # path to the root directory
        # set the PROJECT_ROOT environment variable to root directory
        project_root_env_var=True,
        dotenv=True,  # load environment variables from .env if exists in root directory
        # add root directory to the PYTHONPATH (helps with imports)
        pythonpath=True,
        # change current working directory to the root directory (helps with filepaths)
        cwd=False,
    )


class Test:
    def __init__(self):
        pass


if __name__ == "__main__":
    # setup_path()
    infer()
