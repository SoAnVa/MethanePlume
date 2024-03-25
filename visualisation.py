import os
import numpy as np
import rasterio as rio
import matplotlib.pyplot as plt

def load_image_with_bands(image_path, bands):
    """Load specified bands from a .tif image and return as a numpy array."""
    with rio.open(image_path) as imgfile:
        imgdata = np.stack([imgfile.read(band) for band in bands], axis=-1)
    return imgdata

def visualize_image(image_array, title="Image"):
    """Visualize an image or a single band."""
    plt.figure(figsize=(6, 6))
    if image_array.ndim == 3 and image_array.shape[-1] > 1:  # Multi-band image
        # Normalize or scale pixel values to 0-255 range for RGB visualization
        display_image = (image_array - image_array.min()) / (image_array.max() - image_array.min())
        display_image = (display_image * 255).astype(np.uint8)
        plt.imshow(display_image)
    elif image_array.ndim == 2 or image_array.shape[-1] == 1:  # Single band
        # Normalize single band for visualization
        plt.imshow(image_array, cmap='gray')
    plt.title(title)
    plt.axis('off')
    plt.show()

# Example usage
image_path1 = '/Users/vakili/Documents/Methane-Plume-Segmentation-main/data/DataTrain/input_tiles/S2A_MSIL1C_20171008T064011_N0205_R120_T41SLA_20171008T064617_P933_0.tif'
bands_to_read = [1, 2, 3]  # For RGB visualization
single_band_to_read = [12]  # Example: Visualize only the 3rd band

# Example usage
image_path2 = '/Users/vakili/Documents/Methane-Plume-Segmentation-main/data/DataTrain/augment_input_tiles/augmented_image_2.tif'

# Load the multi-band image for RGB visualization
image_array_rgb = load_image_with_bands(image_path1, bands_to_read)
visualize_image(image_array_rgb, "RGB Image")

# Load a single band for visualization
image_array_single_band = load_image_with_bands(image_path1, single_band_to_read).squeeze()
visualize_image(image_array_single_band, "Single Band Image")

# Load the multi-band image for RGB visualization
image_array_rgb = load_image_with_bands(image_path2, bands_to_read)
visualize_image(image_array_rgb, "RGB Image")

# Load a single band for visualization
image_array_single_band = load_image_with_bands(image_path2, single_band_to_read).squeeze()
visualize_image(image_array_single_band, "Single Band Image")


