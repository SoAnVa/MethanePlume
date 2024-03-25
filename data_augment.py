import os
import random
import numpy as np
from scipy.ndimage import rotate, shift, zoom
import rasterio
import pandas as pd
from data import PlumeSegmentationDataset

#augmented data
augmented_images_dir = './data/DataTrain/augment_input_tiles/'
augmented_masks_dir = './data/DataTrain/augment_output_matrix/'
os.makedirs(augmented_images_dir, exist_ok=True)
os.makedirs(augmented_masks_dir, exist_ok=True)

#original dataset
dataset = PlumeSegmentationDataset(datadir='./data/DataTrain/input_tiles/',
                                   segdir='./data/DataTrain/output_matrix/')

def augment_data(image, mask, original_height, original_width):
    # Random rotation
    angle = random.uniform(0, 360)
    rotated_image = rotate(image, angle, axes=(1, 2), reshape=False, mode='nearest')
    rotated_mask = rotate(mask, angle, axes=(0, 1), reshape=False, mode='nearest')

    # Random translation
    max_trans = 0.2
    trans_x = random.uniform(-max_trans, max_trans) * original_width
    trans_y = random.uniform(-max_trans, max_trans) * original_height
    translated_image = shift(rotated_image, shift=[0, trans_y, trans_x], mode='nearest')
    translated_mask = shift(rotated_mask, shift=[trans_y, trans_x], mode='nearest')

    # Random scaling (zoom in/out), preserving dimensions
    scale_factor = random.uniform(0.8, 1.2)
    new_height = int(scale_factor * original_height)
    new_width = int(scale_factor * original_width)
    zoomed_image = zoom(translated_image, (1, scale_factor, scale_factor), order=1)
    zoomed_mask = zoom(translated_mask, (scale_factor, scale_factor), order=0)

 
    if scale_factor > 1:  
        crop_y = (zoomed_image.shape[1] - original_height) // 2
        crop_x = (zoomed_image.shape[2] - original_width) // 2
        final_image = zoomed_image[:, crop_y:crop_y+original_height, crop_x:crop_x+original_width]
        final_mask = zoomed_mask[crop_y:crop_y+original_height, crop_x:crop_x+original_width]
    else:  
        pad_height = (original_height - new_height) // 2
        pad_width = (original_width - new_width) // 2
        final_image = np.pad(zoomed_image, [(0,0), (pad_height, original_height - new_height - pad_height), (pad_width, original_width - new_width - pad_width)], mode='constant')
        final_mask = np.pad(zoomed_mask, [(pad_height, original_height - new_height - pad_height), (pad_width, original_width - new_width - pad_width)], mode='constant')

    # mask
    final_mask = (final_mask > 0.5).astype(np.uint8)

    return final_image, final_mask

def save_tif_image(image_array, file_path):
    with rasterio.open(
        file_path,
        'w',
        driver='GTiff',
        height=image_array.shape[1],
        width=image_array.shape[2],
        count=image_array.shape[0],
        dtype='float32',  # Ensure dtype matches your data or 'uint16' if needed
        crs='+proj=latlong',
        transform=rasterio.transform.from_origin(-180, 180, 0.1, 0.1)
    ) as dst:
        for i in range(image_array.shape[0]):
            dst.write(image_array[i, :, :], i + 1)

def save_csv_mask(mask_array, file_path):
    pd.DataFrame(mask_array).to_csv(file_path, header=False, index=False)

# Iterate over the dataset and apply augmentation
for idx in range(len(dataset)):
    sample = dataset[idx]
    image = sample['img']
    mask = sample['fpt']
    original_height, original_width = image.shape[1], image.shape[2]

    augmented_image, augmented_mask = augment_data(image, mask, original_height, original_width)

    # File paths
    augmented_image_path = os.path.join(augmented_images_dir, f'augmented_image_{idx}.tif')
    augmented_mask_path = os.path.join(augmented_masks_dir, f'augmented_image_{idx}.csv') #Frocement avoir le meme nom sinon probleme

    # Save the augmented image and mask
    save_tif_image(augmented_image, augmented_image_path)
    save_csv_mask(augmented_mask, augmented_mask_path)

print("Augmented dataset generated successfully")
