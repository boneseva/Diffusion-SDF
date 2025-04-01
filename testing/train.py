import warnings
warnings.filterwarnings(
    "ignore",
    message="Torchmetrics v0.9 introduced.*full_state_update",
    category=UserWarning
)

from pytorch_lightning.loggers import WandbLogger

import os
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
import torch.nn as nn
import torch.nn.functional as F
import pytorch_lightning as pl
from pytorch_lightning.callbacks import ModelCheckpoint
import nibabel as nib

import wandb


class VoxelSDFDataset(Dataset):
    def __init__(self, sdf_dir):
        self.sdf_files = [os.path.join(sdf_dir, f) for f in os.listdir(sdf_dir)
                          if f.endswith(('.nii', '.nii.gz'))]

    def __len__(self):
        return len(self.sdf_files)

    def __getitem__(self, idx):
        # Load and verify SDF data
        sdf = nib.load(self.sdf_files[idx]).get_fdata()
        assert np.abs(sdf).max() <= 1.0, "SDF values not normalized!"
        return torch.tensor(sdf, dtype=torch.float32).unsqueeze(0)  # (1, 80, 80, 80)


class VoxelVAE(pl.LightningModule):
    def __init__(self, latent_dim=256):
        super().__init__()
        self.save_hyperparameters()
        self.latent_dim = latent_dim

        # Encoder (modified for 80x80x80 input)
        self.encoder = nn.Sequential(
            nn.Conv3d(1, 64, 3, stride=2, padding=1),  # 80 -> 40
            nn.ReLU(),
            nn.Conv3d(64, 128, 3, stride=2, padding=1),  # 40 -> 20
            nn.ReLU(),
            nn.Conv3d(128, 256, 3, stride=2, padding=1),  # 20 -> 10
            nn.ReLU(),
            nn.Flatten(),
            nn.Linear(256 * 10 * 10 * 10, latent_dim * 2)
        )

        # Decoder (paper-style architecture)
        self.decoder = nn.Sequential(
            nn.Linear(latent_dim, 256 * 10 * 10 * 10),
            View((-1, 256, 10, 10, 10)),
            nn.ConvTranspose3d(256, 128, 3, stride=2, padding=1, output_padding=1),
            nn.ReLU(),
            nn.ConvTranspose3d(128, 64, 3, stride=2, padding=1, output_padding=1),
            nn.ReLU(),
            nn.ConvTranspose3d(64, 1, 3, stride=2, padding=1, output_padding=1),
            nn.Tanh()  # Paper uses tanh for SDF [-1,1]
        )

    def forward(self, x):
        mu_logvar = self.encoder(x)
        mu, logvar = mu_logvar.chunk(2, dim=1)
        z = self.reparameterize(mu, logvar)
        return self.decoder(z), mu, logvar

    def reparameterize(self, mu, logvar):
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + eps * std

    def training_step(self, batch, batch_idx):
        recon, mu, logvar = self(batch)

        # Paper Eq.2: L1 reconstruction + KL divergence
        recon_loss = F.l1_loss(recon, batch)
        kl_loss = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp()) / batch.size(0)
        total_loss = recon_loss + 1e-5 * kl_loss  # β=1e-5 from paper

        # Log losses to WandB
        self.log('train_loss', total_loss)
        self.log('recon_loss', recon_loss)
        self.log('kl_loss', kl_loss)

        # Log images every few epochs
        if batch_idx == 0 and self.current_epoch % 5 == 0:
            with torch.no_grad():
                input_slice = batch[0][0][40].cpu().numpy()  # Middle slice of input (z-axis)
                recon_slice = recon[0][0][40].cpu().numpy()  # Middle slice of reconstruction

                if isinstance(self.logger.experiment, wandb.sdk.wandb_run.Run):
                    self.logger.experiment.log({
                        "epoch": self.current_epoch,
                        "input_slice": wandb.Image(input_slice),
                        "reconstruction_slice": wandb.Image(recon_slice),
                        "train_loss": total_loss.item(),  # Log loss here for visualization alongside images
                        "recon_loss": recon_loss.item(),
                        "kl_loss": kl_loss.item()
                    })

        return total_loss

    def configure_optimizers(self):
        return torch.optim.Adam(self.parameters(), lr=1e-4)


class View(nn.Module):
    def __init__(self, shape):
        super().__init__()
        self.shape = shape

    def forward(self, x):
        return x.view(*self.shape)


def train():
    config = {
        'batch_size': 8,
        'latent_dim': 256,
        'max_epochs': 200,
        'data_path': r'C:\Users\evabo\Documents\Repos\Diffusion-SDF\testing\lyso_sdf'
    }

    wandb_logger = WandbLogger(
        project="Diffusion-SDF-VAE",
        name="voxel-vae-training",
        config=config,
    )

    dataset = VoxelSDFDataset(config['data_path'])
    loader = DataLoader(dataset,
                        batch_size=config['batch_size'],
                        shuffle=True,
                        num_workers=4,
                        pin_memory=True)

    model = VoxelVAE(latent_dim=config['latent_dim'])

    checkpoint_cb = ModelCheckpoint(
        dirpath='checkpoints',
        filename='vae-{epoch}-{train_loss:.2f}',
        save_top_k=3,
        monitor='train_loss'
    )

    trainer = pl.Trainer(
        accelerator='gpu' if torch.cuda.is_available() else 'cpu',
        devices=1,
        max_epochs=config['max_epochs'],
        callbacks=[checkpoint_cb],
        logger=wandb_logger,
    )

    wandb_logger.watch(model)  # Watch model gradients and parameters

    trainer.fit(model=model,
                train_dataloaders=loader)

    wandb.finish()


if __name__ == '__main__':
    train()
