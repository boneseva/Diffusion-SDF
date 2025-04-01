import csv
import numpy as np
import matplotlib.pyplot as plt


def read_csv(file_path):
    with open(file_path, 'r') as f:
        reader = csv.reader(f)
        data = list(reader)
    return np.array(data).astype(float)


if __name__ == '__main__':
    file_path = r"C:\Users\evabo\Documents\Repos\Diffusion-SDF\data\acronym\Couch\37cfcafe606611d81246538126da07a8\sdf_data.csv"
#    file_path = r"C:\Users\evabo\Documents\Repos\Diffusion-SDF\data\grid_data\acronym\Couch\37cfcafe606611d81246538126da07a8\grid_gt.csv"
    data = read_csv(file_path)

    # Set your threshold here (absolute value)
    threshold = 0.01  # Adjust this value as needed

    # Filter points based on SDF threshold
    mask = np.abs(data[:, 3]) < threshold
    filtered_data = data[mask]

    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')

    # Plot filtered data
    scatter = ax.scatter(
        filtered_data[:, 0],  # X coordinates
        filtered_data[:, 1],  # Z coordinates (swapped with Y in original data)
        filtered_data[:, 2],  # Y coordinates
        c=filtered_data[:, 3],  # SDF values for coloring
        cmap='viridis'
    )

    ax.set_xlabel('X')
    ax.set_ylabel('Z')
    ax.set_zlabel('Y')

    cbar = plt.colorbar(scatter)
    cbar.set_label('SDF Value')

    plt.title(f'SDF Visualization (|SDF| ≥ {threshold})')
    plt.show()
