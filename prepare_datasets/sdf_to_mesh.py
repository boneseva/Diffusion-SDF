import os
import nibabel as nib
import numpy as np
from skimage.measure import marching_cubes
import trimesh

def sdf_to_mesh(sdf_path, output_dir, level=0.0, step_size=1, visualize=False):
    """
    Convert SDF volume to mesh using Marching Cubes
    Args:
        sdf_path (str): Path to input SDF .nii/.nii.gz file
        output_dir (str): Directory to save output mesh
        level (float): Iso-surface value (typically 0 for SDFs)
        step_size (int): Step size for Marching Cubes
        visualize (bool): Whether to plot a middle slice
    """
    # Load SDF volume
    nii = nib.load(sdf_path)
    sdf_volume = nii.get_fdata()
    affine = nii.affine

    # Verify SDF values are normalized
    assert np.abs(sdf_volume).max() <= 1.0, "SDF values not normalized!"

    try:
        # Run Marching Cubes
        vertices, faces, normals, _ = marching_cubes(
            sdf_volume,
            level=level,
            step_size=step_size,
            allow_degenerate=False
        )

        # Convert vertices to physical coordinates
        vertices_physical = nib.affines.apply_affine(affine, vertices)

        # Create mesh
        mesh = trimesh.Trimesh(
            vertices=vertices_physical,
            faces=faces,
            vertex_normals=normals
        )

        # Save mesh
        filename = os.path.basename(sdf_path).replace('.nii', '')
        output_path = os.path.join(output_dir, f"{filename}_mesh.stl")
        mesh.export(output_path)
        print(f"Saved mesh: {output_path}")

        # Optional visualization
        if visualize:
            import matplotlib.pyplot as plt
            slice_idx = sdf_volume.shape[2] // 2
            plt.imshow(sdf_volume[:, :, slice_idx], cmap='jet')
            plt.colorbar()
            plt.title(f"Middle Slice: {filename}")
            plt.show()

    except Exception as e:
        print(f"Error processing {sdf_path}: {str(e)}")


if "__main__" == __name__:
    sdf_folder = r"C:\Users\eva.bones\Documents\Diffusion-SDF\testing\training_samples"
    result_folder = r"C:\Users\eva.bones\Documents\Diffusion-SDF\testing\resulting_mesh"
    os.makedirs(result_folder, exist_ok=True)

    for sdf in os.listdir(sdf_folder):
        sdf_to_mesh(os.path.join(sdf_folder, sdf), result_folder)
