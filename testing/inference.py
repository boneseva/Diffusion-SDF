import os
import torch
import numpy as np
import nibabel as nib
from skimage.measure import marching_cubes
import trimesh
from train import get_latest_checkpoint, VoxelVAE, CropCenter3D

from torchvision import transforms

def preprocess_sdf(input_path, target_size=64):
    """Load and preprocess SDF matching training transforms"""
    # Load SDF volume
    sdf = nib.load(input_path).get_fdata()

    # Apply same transforms as training
    transform = transforms.Compose([
        CropCenter3D(target_size=target_size)
    ])

    # Convert to tensor and apply transforms
    sdf_tensor = torch.tensor(sdf, dtype=torch.float32)
    processed = transform(sdf_tensor)

    # Normalize to [-1,1] if needed
    if processed.abs().max() > 1.0:
        processed = processed / processed.abs().max()

    return processed


def encode_decode(model, input_sdf, device='cuda'):
    """Process SDF through full VAE pipeline"""
    model.eval()
    model.to(device)

    with torch.no_grad():
        # Encode to latent space
        mu_logvar = model.encoder(input_sdf.to(device))
        mu, logvar = mu_logvar.chunk(2, dim=1)
        z = model.reparameterize(mu, logvar)

        # Decode back to SDF
        reconstructed = model.decoder(z)

    return reconstructed.cpu().numpy()


def run_inference(checkpoint_path, input_sdf_path, output_dir):
    # Load model
    model = VoxelVAE.load_from_checkpoint(checkpoint_path)

    # Preprocess input
    input_sdf = preprocess_sdf(input_sdf_path)
    print(f"Input shape: {input_sdf.shape}")

    # Encode and decode
    reconstructed = encode_decode(model, input_sdf)

    # Save results
    os.makedirs(output_dir, exist_ok=True)

    # Save input and output
    input_path = os.path.join(output_dir, "input.nii.gz")
    output_path = os.path.join(output_dir, "reconstructed.nii.gz")

    nib.save(nib.Nifti1Image(input_sdf.squeeze().numpy(), affine=np.eye(4)), input_path)
    nib.save(nib.Nifti1Image(reconstructed.squeeze(), affine=np.eye(4)), output_path)

    # Create meshes
    input_mesh = sdf_to_mesh(input_sdf.squeeze())
    output_mesh = sdf_to_mesh(reconstructed.squeeze())

    input_mesh.export(os.path.join(output_dir, "input_mesh.stl"))
    output_mesh.export(os.path.join(output_dir, "reconstructed_mesh.stl"))

    return input_path, output_path


if __name__ == '__main__':
    checkpoint = get_latest_checkpoint(r"C:\Users\eva.bones\Documents\Diffusion-SDF\testing\checkpoints")
    input_sdf_path = r"C:\Users\eva.bones\Documents\Diffusion-SDF\testing\lyso_sdf"
    for file in os.listdir(input_sdf_path):
        file_path = os.path.join(input_sdf_path, file)
        input_path, output_path = run_inference(
            checkpoint_path=checkpoint,
            input_sdf_path=file_path,
            output_dir=r"C:\Users\eva.bones\Documents\Diffusion-SDF\testing\reconstruction_results"
        )
