import os
import nibabel as nib
import numpy as np
from scipy import ndimage


def process_nifti_file(file, input_path, output_path):
    nii = nib.load(input_path)
    data = nii.get_fdata().astype(bool).astype(np.uint8)
    # separate objects inside data into its own volumes
    labeled_data, num_features = ndimage.label(data)
    volumes = []
    for i in range(1, num_features + 1):
        # Extract the i-th object
        object_mask = labeled_data == i
        # get min and max to cut out the volume
        min_coords = np.min(np.argwhere(object_mask), axis=0)
        max_coords = np.max(np.argwhere(object_mask), axis=0)
        # check if min_coordinates are close to 0 and if max coordinates are close to the shape of the data
        if np.any(min_coords <= 0) or np.any(max_coords >= data.shape):
            print(f"Object {i} is out of bounds, skipping.")
            continue
        object_mask = object_mask[min_coords[0]:max_coords[0] + 1,
                                  min_coords[1]:max_coords[1] + 1,
                                  min_coords[2]:max_coords[2] + 1]
        volumes.append(object_mask)
        print(f"Object {i} shape: {object_mask.shape}")
        separated_nii = nib.Nifti1Image(object_mask.astype(np.float32), nii.affine)
        nib.save(separated_nii, os.path.join(output_path, f"{file[:-7]}_{i}.nii.gz"))
    return volumes


def process_all_nifti_files(input_folder, output_folder):
    for file in os.listdir(input_folder):
        if file.endswith(".nii") or file.endswith(".nii.gz"):
            input_path = os.path.join(input_folder, file)
            volumes = process_nifti_file(file, input_path, output_folder)
            print(f"Processed and saved {len(volumes)} volumes for: {file}")

if __name__ == "__main__":
    # Input and output directories
    for organelle in ["mito", "lyso", "golgi", "fv"]:
        if organelle == "mito" or organelle == "fv":
            input_dir = fr"C:\Users\evabo\Documents\Repos\UroCell\UroCell-master\{organelle}\instance"
        elif organelle == "lyso":
            input_dir = fr"C:\Users\evabo\Documents\Repos\UroCell\UroCell-master\{organelle}\binary"
        elif organelle == "golgi":
            input_dir = fr"C:\Users\evabo\Documents\Repos\UroCell\UroCell-master\{organelle}\precise\binary"

        output_dir = fr"C:\Users\evabo\Documents\Repos\Diffusion-SDF\dataset\{organelle}"

        if not os.path.exists(output_dir):
            os.makedirs(output_dir, exist_ok=True)

        # Process all files in the input directory
        process_all_nifti_files(input_dir, output_dir)