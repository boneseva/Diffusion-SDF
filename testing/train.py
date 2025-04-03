import datetime
import warnings

warnings.filterwarnings(
    "ignore",
    message="Torchmetrics v0.9 introduced.*full_state_update",
    category=UserWarning
)

from pytorch_lightning.loggers import WandbLogger, wandb
import os
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
import torch.nn as nn
import torch.nn.functional as F
import pytorch_lightning as pl
from pytorch_lightning.callbacks import ModelCheckpoint
import nibabel as nib
from tqdm import tqdm
import torch.profiler as profiler
import pytz
import wandb

torch.backends.cudnn.benchmark = True

from torchvision import transforms

class RandomFlip3D:
    def __init__(self, p=0.5):
        self.p = p

    def __call__(self, x):
        for axis in [1, 2, 3]:  # Flip along depth (D), height (H), or width (W)
            if torch.rand(1) < self.p:
                x = torch.flip(x, [axis])
        return x

class CropCenter3D:
    def __init__(self, target_size):
        self.target_size = target_size

    def __call__(self, x):
        _, D, H, W = x.shape
        d_start = (D - self.target_size) // 2
        h_start = (H - self.target_size) // 2
        w_start = (W - self.target_size) // 2
        return x[:, d_start:d_start + self.target_size,
                  h_start:h_start + self.target_size,
                  w_start:w_start + self.target_size]


class VoxelSDFDataset(Dataset):
    def __init__(self, sdf_dir, transform=None):
        self.transform = transform
        self.sdf_files = [os.path.join(sdf_dir, f) for f in os.listdir(sdf_dir)
                          if f.endswith(('.nii', '.nii.gz'))]

        self.buffer = []
        for f in tqdm(self.sdf_files, desc="Loading SDF volumes"):
            try:
                vol = nib.load(f).get_fdata()
                assert np.abs(vol).max() <= 1.0, "SDF values not normalized!"
                self.buffer.append(torch.tensor(vol, dtype=torch.float32))
            except Exception as e:
                print(f"Error loading {f}: {str(e)}")

    def __len__(self):
        return len(self.buffer)

    def __getitem__(self, idx):
        vol = self.buffer[idx].unsqueeze(0)
        if self.transform:
            vol = self.transform(vol)
        return vol

class VoxelVAE(pl.LightningModule):
    def __init__(self, latent_dim=256):
        super().__init__()
        self.save_hyperparameters()

        # Encoder with valid group configurations
        self.encoder = nn.Sequential(
            nn.Conv3d(1, 64, 3, stride=2, padding=1),
            nn.GELU(),
            nn.Conv3d(64, 128, 3, stride=2, padding=1),
            nn.GELU(),
            nn.Conv3d(128, 256, 3, stride=2, padding=1),
            nn.GELU(),
            nn.Flatten(),
            nn.Linear(256 * 8 * 8 * 8, latent_dim * 2)
        )

        # Decoder
        self.decoder = nn.Sequential(
            nn.Linear(latent_dim, 256 * 8 * 8 * 8),
            View((-1, 256, 8, 8, 8)),
            nn.ConvTranspose3d(256, 128, 3, stride=2, padding=1, output_padding=1),
            nn.GELU(),
            nn.ConvTranspose3d(128, 64, 3, stride=2, padding=1, output_padding=1),
            nn.GELU(),
            nn.ConvTranspose3d(64, 1, 3, stride=2, padding=1, output_padding=1),
            nn.Tanh()
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

        recon_loss = F.l1_loss(recon, batch)
        kl_loss = -0.5 * torch.mean(1 + logvar - mu.pow(2) - logvar.exp())
        total_loss = recon_loss + 1e-5 * kl_loss

        self.log_dict({
            'train_loss': total_loss,
            'recon_loss': recon_loss,
            'kl_loss': kl_loss,
            'gpu_mem': torch.cuda.max_memory_allocated() / 1e9
        })

        if batch_idx == 0 and self.current_epoch % 5 == 0:  # Log every 5 epochs
            with torch.no_grad():
                input_slice = batch[0][0][32].cpu().numpy()  # Middle slice (z-axis)
                recon_slice = recon[0][0][32].cpu().numpy()

                #change the colors of images to jet color scheme
                input_slice = (input_slice - input_slice.min()) / (input_slice.max() - input_slice.min())
                recon_slice = (recon_slice - recon_slice.min()) / (recon_slice.max() - recon_slice.min())
                input_slice = (input_slice * 255).astype(np.uint8)
                recon_slice = (recon_slice * 255).astype(np.uint8)
                input_slice = np.stack([input_slice] * 3, axis=-1)  # Convert to RGB
                recon_slice = np.stack([recon_slice] * 3, axis=-1)

                self.logger.experiment.log({
                    "epoch": self.current_epoch,
                    "input_slice": wandb.Image(input_slice),
                    "reconstruction_slice": wandb.Image(recon_slice)
                }, commit=False)

        return total_loss

    def configure_optimizers(self):
        return torch.optim.AdamW(self.parameters(), lr=1e-4)


class View(nn.Module):
    def __init__(self, shape):
        super().__init__()
        self.shape = shape

    def forward(self, x):
        return x.view(self.shape)

def get_latest_checkpoint(checkpoint_dir):
    checkpoints = [f for f in os.listdir(checkpoint_dir) if f.endswith('.ckpt')]
    if not checkpoints:
        return None
    latest_checkpoint = max(checkpoints, key=lambda x: os.path.getctime(os.path.join(checkpoint_dir, x)))
    return os.path.join(checkpoint_dir, latest_checkpoint)

def train():
    torch.set_float32_matmul_precision('high')

    organelle = "fv"
    config = {
        'batch_size': 8,
        'latent_dim': 256,
        'max_epochs': 10000,
        'data_path': rf'C:\Users\eva.bones\Documents\Diffusion-SDF\testing\{organelle}_sdf',
        'checkpoint_path': './checkpoints/'
    }

    checkpoint = get_latest_checkpoint(config['checkpoint_path'])

    date = datetime.datetime.now(pytz.timezone("Europe/Berlin"))
    formatted_date = date.strftime("%d/%m")
    formatted_time = date.strftime("%H:%M")
    run_name = f"testing_{organelle}_{formatted_date}@{formatted_time}"

    wandb_logger = WandbLogger(project="Diffusion-SDF-VAE", name=run_name)

    train_transform = transforms.Compose([
        RandomFlip3D(p=0.5),  # 50% chance per axis
        CropCenter3D(target_size=64),  # Center crop to 64x64x64
    ])

    dataset = VoxelSDFDataset(config['data_path'], transform=train_transform)
    loader = DataLoader(
        dataset,
        batch_size=config['batch_size'],
        shuffle=True,
        num_workers=4,
        pin_memory=True,
        persistent_workers=True
    )

    if checkpoint:
        print(f"Loading model from checkpoint: {checkpoint}")
        model = VoxelVAE.load_from_checkpoint(checkpoint, latent_dim=config['latent_dim'])
    else:
        print("No checkpoint found. Training from scratch.")
        model = VoxelVAE(latent_dim=config['latent_dim'])

    trainer = pl.Trainer(
        accelerator='gpu' if torch.cuda.is_available() else 'cpu',
        devices=1,
        precision='16-mixed',
        max_epochs=config['max_epochs'],
        logger=wandb_logger,
        callbacks=[ModelCheckpoint(dirpath='checkpoints')],
        log_every_n_steps=2
    )

    with profiler.profile(
            activities=[
                profiler.ProfilerActivity.CPU,
                profiler.ProfilerActivity.CUDA],
            on_trace_ready=profiler.tensorboard_trace_handler('./log')
    ) as prof:
        trainer.fit(model, loader)

    print(prof.key_averages().table(sort_by="cuda_time_total"))


if __name__ == '__main__':
    train()
