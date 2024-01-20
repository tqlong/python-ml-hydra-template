import torch as T
from torch import nn
from torch.nn import functional as F
from torch.utils.data import DataLoader, Dataset, random_split
import lightning as L

from omegaconf import DictConfig, OmegaConf
import hydra

import rootutils
rootutils.setup_root(__file__, indicator=".project-root", pythonpath=True)


class Encoder(nn.Module):
    def __init__(self, img_width=28, img_height=28, img_channel=1, hiddens=[64, 3]):
        super().__init__()
        self.l1 = nn.Sequential(
            nn.Linear(in_features=img_width*img_height *
                      img_channel, out_features=hiddens[0]),
            nn.ReLU(inplace=True),
            nn.Linear(in_features=hiddens[0], out_features=hiddens[1])
        )

    def forward(self, x):
        return self.l1(x)


class Decoder(nn.Module):
    def __init__(self, img_width=28, img_height=28, img_channel=1, hiddens=[3, 64]):
        super().__init__()
        self.l1 = nn.Sequential(
            nn.Linear(in_features=hiddens[0], out_features=hiddens[1]),
            nn.ReLU(inplace=True),
            nn.Linear(
                in_features=hiddens[1], out_features=img_width*img_height*img_channel),
        )

    def forward(self, x):
        return self.l1(x)


class LitAutoEncoder(L.LightningModule):
    def __init__(self, encoder, decoder, lr=1e-3):
        super().__init__()
        self.save_hyperparameters(ignore=['encoder', 'decoder'])
        self.encoder = encoder
        self.decoder = decoder

    def step(self, batch):
        x, y = batch
        x = x.view(x.size(0), -1)  # B x D
        z = self.encoder(x)
        xhat = self.decoder(z)
        loss = F.mse_loss(xhat, x)
        return x, xhat, loss

    def training_step(self, batch, batch_idx):
        x, xhat, loss = self.step(batch)
        self.log("train/loss", loss)
        return loss

    def test_step(self, batch, batch_idx):
        x, xhat, loss = self.step(batch)
        self.log("test/loss", loss)

    def validation_step(self, batch, batch_idx):
        x, xhat, loss = self.step(batch)
        self.log("val/loss", loss)

    def configure_optimizers(self):
        optimizer = T.optim.Adam(self.parameters(), lr=self.hparams.lr)
        return optimizer


@hydra.main(version_base=None, config_path="../../configs", config_name="lvl2")
def main(cfg: DictConfig):
    print(OmegaConf.to_yaml(cfg))
    dataset: Dataset = hydra.utils.instantiate(cfg.dataset)
    train_dataset, val_dataset = random_split(dataset, cfg.sample_split)
    train_dataloader: DataLoader = hydra.utils.instantiate(
        cfg.train_dataloader)(train_dataset)
    val_dataloader: DataLoader = hydra.utils.instantiate(
        cfg.val_dataloader)(val_dataset)
    test_dataloader: DataLoader = hydra.utils.instantiate(cfg.test_dataloader)
    autoencoder: LitAutoEncoder = hydra.utils.instantiate(cfg.autoencoder)
    trainer: L.Trainer = hydra.utils.instantiate(cfg.trainer)

    trainer.fit(model=autoencoder, train_dataloaders=train_dataloader,
                val_dataloaders=val_dataloader)

    trainer.test(model=autoencoder, dataloaders=test_dataloader)


if __name__ == "__main__":
    main()
