import os
import nibabel as nib
import numpy as np
import edt


def pad_to_target_shape(data, target_shape):
    """
    Pad a 3D volume symmetrically to match the target shape.
    Args:
        data (np.ndarray): The input 3D volume.
        target_shape (tuple): The desired shape (x, y, z).
    Returns:
        np.ndarray: Padded volume.
    """
    current_shape = data.shape
    pad_width = []
    for curr, target in zip(current_shape, target_shape):
        diff = target - curr
        if diff < 0:
            raise ValueError(f"Current dimension {curr} exceeds target dimension {target}.")
        pad_before = diff // 2
        pad_after = diff - pad_before
        pad_width.append((pad_before, pad_after))

    return np.pad(data, pad_width, mode='constant', constant_values=0)


def process_nifti_file(input_path, target_shape=(80, 80, 80)):
    """
    Load a NIfTI file, pad it to the target shape, compute the SDF, and return the padded SDF volume.
    Args:
        input_path (str): Path to the input NIfTI file.
        target_shape (tuple): Desired output shape (default: 80x80x80).
    Returns:
        np.ndarray: Padded SDF volume.
    """
    # Load the NIfTI file
    nii = nib.load(input_path)
    data = nii.get_fdata().astype(bool).astype(np.uint8)  # Ensure binary input

    # Pad the volume
    padded_data = pad_to_target_shape(data, target_shape)

    # Compute the SDF
    sdf = edt.sdf(
        padded_data,
        anisotropy=nii.header.get_zooms(),
        black_border=False,
        parallel=0
    )

    # Normalize the SDF to [-1, 1]
    max_dist = np.max(np.abs(sdf))
    sdf_normalized = np.clip(sdf / max_dist, -1.0, 1.0)

    return sdf_normalized


def process_all_nifti_files(input_folder, output_folder, target_shape=(80, 80, 80)):
    """
    Process all NIfTI files in a folder by padding them to the target shape and computing their SDFs.
    Args:
        input_folder (str): Path to the folder containing input NIfTI files.
        output_folder (str): Path to save the processed SDF files.
        target_shape (tuple): Desired output shape (default: 80x80x80).
    """
    os.makedirs(output_folder, exist_ok=True)

    for file in os.listdir(input_folder):
        if file.endswith(".nii") or file.endswith(".nii.gz"):
            input_path = os.path.join(input_folder, file)
            output_path = os.path.join(output_folder, f"sdf_{file}")
            try:
                sdf_volume = process_nifti_file(input_path, target_shape)

                # Save the resulting SDF as a new NIfTI file
                nii = nib.load(input_path)  # Reload original NIfTI for affine/header info
                sdf_nii = nib.Nifti1Image(sdf_volume.astype(np.float32), nii.affine)
                nib.save(sdf_nii, output_path)

                print(f"Processed and saved SDF for: {file}")
            except Exception as e:
                print(f"Error processing {file}: {e}")


if __name__ == "__main__":
    # Input and output directories
    input_dir = r"C:\Users\evabo\Documents\Repos\Statistic-Models-For-Cellular-Structures\Lyso_single\In_center\Framed"
    output_dir = r"C:\Users\evabo\Documents\Repos\Diffusion-SDF\testing\lyso_sdf"

    # Target shape for padding
    target_resolution = (80, 80, 80)

    # Process all files in the input directory
    process_all_nifti_files(input_dir, output_dir, target_resolution)
