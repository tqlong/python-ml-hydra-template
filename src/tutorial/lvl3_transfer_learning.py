import logging
import os
from typing import Any, Callable
import rootutils
import hydra
from omegaconf import DictConfig, OmegaConf
import torch as T
import torchvision as TV
from torch import nn
from torch.utils.data import DataLoader, Dataset, random_split
import lightning as L
from torchvision.datasets.folder import default_loader
from torchvision.transforms.v2 import (
    Compose, ToDtype, Normalize, RandomResizedCrop, RandomHorizontalFlip, ToTensor,
    Resize, ToImage, RandomCrop, RandAugment, CutMix, MixUp, RandomChoice,
    ColorJitter
)
from torchmetrics import Accuracy, MeanMetric, MaxMetric, MinMetric


class Transform(Compose):
    def __init__(self, size, horizontal_flip_p, mean, std, train=True):
        main_transforms = [
            ToImage(),
            Resize(size=(int(size[0]*1.15),
                   int(size[1]*1.15)), antialias=True),
            # RandAugment(),
            RandomCrop(size=size),
            ColorJitter(brightness=.5, hue=.3),
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


class ImageDataset(Dataset):
    def __init__(self, original_dataset, img_transform):
        self.dataset = original_dataset
        self.transform = img_transform

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, index):
        item = self.dataset[index]
        return (self.transform(item[0]),) + item[1:]


class DogCatDataModule(L.LightningDataModule):
    # , predict_transform):
    def __init__(
            self,
            data_dir, train_dir, test_dir, train_val_split,
            train_transform, predict_transform,
            batch_size, num_workers):
        super().__init__()
        self.save_hyperparameters(
            ignore=['train_transform', 'predict_transform'])
        self.train_transform = train_transform
        self.predict_transform = predict_transform

    def setup(self, stage: str):
        if stage == "fit":
            train_dataset: Dataset = ImageFolderWithPaths(
                os.path.join(self.hparams.data_dir, self.hparams.train_dir))
            train_dataset, val_dataset = random_split(
                train_dataset, self.hparams.train_val_split)
            self.train_dataset = ImageDataset(
                train_dataset, img_transform=self.train_transform)
            self.val_dataset = ImageDataset(
                val_dataset, img_transform=self.predict_transform)
        if stage == "predict":
            self.predict_dataset = ImageFolderWithPaths(
                os.path.join(self.hparams.data_dir, self.hparams.test_dir),
                transform=self.predict_transform)

    def train_dataloader(self):
        return DataLoader(
            self.train_dataset, batch_size=self.hparams.batch_size, shuffle=True,
            num_workers=self.hparams.num_workers, pin_memory=True, drop_last=True)

    def val_dataloader(self):
        return DataLoader(
            self.val_dataset, batch_size=self.hparams.batch_size, shuffle=False,
            num_workers=self.hparams.num_workers, pin_memory=True)

    def predict_dataloader(self):
        return DataLoader(
            self.predict_dataset, batch_size=self.hparams.batch_size, shuffle=False,
            num_workers=self.hparams.num_workers, pin_memory=True)


class Resnet(nn.Module):
    def __init__(self, backbone_name: str, n_classes: int):
        super().__init__()
        backbone = TV.models.get_model(backbone_name, weights='DEFAULT')
        num_features = backbone.fc.in_features
        layers = list(backbone.children())[:-1]
        self.feature_extractor = nn.Sequential(*layers)
        for param in self.feature_extractor.parameters():
            param.requires_grad = False
        self.fc = nn.Sequential(
            nn.Dropout(0.5),
            nn.Linear(in_features=num_features, out_features=1024),
            nn.Dropout(0.2),
            nn.Linear(in_features=1024, out_features=512),
            nn.Dropout(0.1),
            nn.Linear(in_features=512, out_features=n_classes),
        )

    def forward(self, x):
        self.feature_extractor.eval()
        with T.no_grad():
            features = self.feature_extractor(x).flatten(1)
        out = self.fc(features)
        return out


class LitModule(L.LightningModule):
    def __init__(self, net: nn.Module, criterion, optimizer, scheduler, n_classes, csv_output):
        super().__init__()
        self.save_hyperparameters(ignore=['net', 'criterion'])
        self.net = net

        self.example_input_array = T.Tensor(16, 3, 224, 224)

        self.criterion = criterion
        self.train_acc = Accuracy(task="multiclass", num_classes=n_classes)
        self.val_acc = Accuracy(task="multiclass", num_classes=n_classes)
        self.train_loss = MeanMetric()
        self.val_loss = MeanMetric()
        self.val_acc_best = MaxMetric()
        self.val_loss_best = MinMetric()

        cutmix = CutMix(num_classes=n_classes)
        mixup = MixUp(num_classes=n_classes)
        self.cutmix_or_mixup = RandomChoice([cutmix, mixup])

    def forward(self, x):
        return self.net(x)

    def on_train_start(self):
        self.val_loss.reset()
        self.val_acc.reset()
        self.val_acc_best.reset()
        self.val_loss_best.reset()

    def step(self, batch, use_cut_mix_mix_up=True):
        x, y, p = batch
        if use_cut_mix_mix_up:
            x, y = self.cutmix_or_mixup(x, y)
        logits = self.net(x)
        loss = self.criterion(logits, y)
        preds = T.argmax(logits, dim=1)
        return loss, preds, y

    def training_step(self, batch, batch_idx):
        loss, preds, targets = self.step(batch, use_cut_mix_mix_up=False)
        self.train_loss(loss)
        # print(preds.shape, targets.shape)
        # self.train_acc(preds, targets)
        self.log("train/loss", self.train_loss,
                 on_step=False, on_epoch=True, prog_bar=True)
        # self.log("train/acc", self.train_acc, on_step=False,
        #          on_epoch=True, prog_bar=True)
        return loss

    def validation_step(self, batch, batch_idx):
        loss, preds, targets = self.step(batch, use_cut_mix_mix_up=False)
        self.val_loss(loss)
        self.val_acc(preds, targets)
        self.log("val/loss", self.val_loss, on_step=False,
                 on_epoch=True, prog_bar=True)
        self.log("val/acc", self.val_acc, on_step=False,
                 on_epoch=True, prog_bar=True)

    def on_predict_start(self):
        with open(self.hparams.csv_output, 'w') as csv_output:
            print("id,label", file=csv_output)

    def predict_step(self, batch, batch_idx):
        x, _, p = batch
        logits = self.net(x)
        probs = nn.functional.softmax(logits, dim=1)
        with open(self.hparams.csv_output, '+a') as csv_output:
            for path, prediction in zip(p, probs):
                id = (path.split('/')[-1]).split('.')[0]
                print(f"{id},{prediction[1].item():.4f}", file=csv_output)
        return zip(p, probs)

    def on_validation_epoch_end(self):
        acc = self.val_acc.compute()
        self.val_acc_best(acc)
        loss = self.val_loss.compute()
        self.val_loss_best(loss)
        self.log("val/acc_best", self.val_acc_best.compute(),
                 sync_dist=True, prog_bar=True)
        self.log("val/loss_best", self.val_loss_best.compute(),
                 sync_dist=True, prog_bar=True)

    def configure_optimizers(self):
        optimizer = self.hparams.optimizer(params=self.net.parameters())
        if self.hparams.scheduler is not None:
            scheduler = self.hparams.scheduler(optimizer=optimizer)
            return {
                "optimizer": optimizer,
                "lr_scheduler": {
                    "scheduler": scheduler,
                    "monitor": "val/loss",
                    "interval": "epoch",
                    "frequency": 1,
                },
            }
        return {"optimizer": optimizer}


@hydra.main(version_base=None, config_path="../../configs", config_name="lvl3")
def main(cfg: DictConfig):
    print(OmegaConf.to_yaml(cfg))

    data_module: L.LightningDataModule = hydra.utils.instantiate(
        cfg.datamodule)
    # test_datamodule(data_module)
    module: L.LightningModule = hydra.utils.instantiate(cfg.module)
    # test_module(module)
    trainer: L.Trainer = hydra.utils.instantiate(cfg.trainer)

    if cfg.get("train"):
        logging.info("Starting training!")
        trainer.fit(model=module, datamodule=data_module)

    logging.info("Starting prediction!")
    ckpt_path = ""
    if cfg.get("ckpt_path"):
        ckpt_path = cfg.get("ckpt_path")
    elif cfg.get("train"):
        ckpt_path = trainer.checkpoint_callback.best_model_path
    if ckpt_path == "":
        ckpt_path = None  # use current weights
    else:
        logging.info(f"Use best weights at {ckpt_path}")
    trainer.predict(model=module, datamodule=data_module, ckpt_path=ckpt_path)


def test_datamodule(dm: DogCatDataModule):
    dm.setup(stage="fit")
    print(f"train = {len(dm.train_dataset)}, val = {len(dm.val_dataset)}")
    for i in range(20):
        logging.info(f"loop {i}")
        for x, y, p in dm.train_dataloader():
            print(x.shape, y.shape, p)
            break

    dm.setup(stage="predict")
    for x, y, p in dm.predict_dataloader():
        print(x.shape, y.shape, p)
        break


def test_module(m: LitModule):
    x = T.zeros((16, 3, 224, 224))
    out = m(x)
    print(x.shape, out.shape)
    m.configure_optimizers()


if __name__ == "__main__":
    L.seed_everything(103)
    rootutils.setup_root(__file__, indicator=".project-root", pythonpath=True)
    logging.basicConfig(level=logging.INFO)
    main()
