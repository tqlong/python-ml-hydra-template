import os
from typing import Any, Callable, Optional
from lightning.pytorch.utilities.types import EVAL_DATALOADERS, TRAIN_DATALOADERS
import rootutils
import hydra
from omegaconf import DictConfig, OmegaConf
import torch as T
import torchvision as TV
from torch import nn
from torch.nn import functional as F
from torch.utils.data import DataLoader, Dataset, random_split
import lightning as L
from torchvision.datasets.folder import default_loader
from torchvision.transforms.v2 import (
    Compose, ToDtype, Normalize, RandomResizedCrop, RandomHorizontalFlip, ToTensor,
    Resize, ToImage
)


class Transform(Compose):
    def __init__(self, size, horizontal_flip_p, mean, std, train=True):
        main_transforms = [
            ToImage(),
            Resize(size=(int(size[0]*1.2), int(size[1]*1.2)), antialias=True),
            RandomResizedCrop(size=size, antialias=True),
            RandomHorizontalFlip(p=horizontal_flip_p)
        ] if train else [
            ToImage(),
            Resize(size=(size[0], size[1]), antialias=True)
        ]
        super().__init__(main_transforms + [
            ToDtype(dtype=T.float32, scale=True),
            Normalize(mean=mean, std=std)
        ])


class ImageFolderWithPaths(Dataset):
    """Custom dataset that includes image file paths. Extends
    torchvision.datasets.ImageFolder
    """

    def __init__(self, root: str, transform: Callable[..., Any] | None = None):
        self.dataset = TV.datasets.ImageFolder(root, transform=transform)

    def __len__(self):
        return len(self.dataset)

    # override the __getitem__ method. this is the method that dataloader calls
    def __getitem__(self, index):
        # this is what ImageFolder normally returns
        original_tuple = self.dataset[index]
        # the image file path
        path = self.dataset.imgs[index][0]
        # make a new tuple that includes original and the path
        tuple_with_path = (original_tuple + (path,))
        return tuple_with_path


class DogCatDataModule(L.LightningDataModule):
    # , predict_transform):
    def __init__(
            self,
            data_dir, train_dir, test_dir, train_val_split,
            train_transform, predict_transform,
            batch_size):
        super().__init__()
        self.save_hyperparameters(
            ignore=['train_transform', 'predict_transform'])
        self.train_transform = train_transform
        self.predict_transform = predict_transform

    def setup(self, stage: str):
        if stage == "fit":
            train_dataset: Dataset = ImageFolderWithPaths(
                os.path.join(self.hparams.data_dir, self.hparams.train_dir),
                transform=self.train_transform)
            self.train_dataset, self.val_dataset = random_split(
                train_dataset, self.hparams.train_val_split)
        if stage == "predict":
            self.predict_dataset = ImageFolderWithPaths(
                os.path.join(self.hparams.data_dir, self.hparams.test_dir),
                transform=self.predict_transform)

    def train_dataloader(self):
        return DataLoader(self.train_dataset, batch_size=self.hparams.batch_size, shuffle=True)

    def val_dataloader(self):
        return DataLoader(self.val_dataset, batch_size=self.hparams.batch_size, shuffle=False)

    def predict_dataloader(self):
        return DataLoader(self.predict_dataset, batch_size=self.hparams.batch_size, shuffle=False)


@hydra.main(version_base=None, config_path="../../configs", config_name="lvl3")
def main(cfg: DictConfig):
    print(OmegaConf.to_yaml(cfg))

    # TODO: data module
    data_module: L.LightningDataModule = hydra.utils.instantiate(
        cfg.datamodule)
    # test_datamodule(data_module)


def test_datamodule(dm: DogCatDataModule):
    dm.setup(stage="fit")
    print(f"train = {len(dm.train_dataset)}, val = {len(dm.val_dataset)}")
    for x, y, p in dm.train_dataloader():
        print(x.shape, y.shape, p)
        break

    dm.setup(stage="predict")
    for x, y, p in dm.predict_dataloader():
        print(x.shape, y.shape, p)
        break


if __name__ == "__main__":
    L.seed_everything(103)
    rootutils.setup_root(__file__, indicator=".project-root", pythonpath=True)
    main()
