import matplotlib.pyplot as plt
import numpy as np
from data import create_dataset  # Ensure this matches your implementation
import albumentations as A

data_dir = "data/DataTrain/input_tiles"
mask_dir = "data/DataTrain/output_matrix"
bands = [4, 3, 2]  # Example for Sentinel-2 RGB bands

# Creating dataset
dataset = create_dataset(datadir=data_dir, segdir=mask_dir, band=bands, apply_transforms=False)
index = 53

# Getting a sample
sample = dataset[index]

image = sample['img'].transpose(1, 2, 0)
image=image[1:90,1:90,:]  # Convert from CxHxW to HxWxC for visualization
image_min = image.min()
image_max = image.max()
image = (image - image_min) / (image_max - image_min)

mask = sample['fpt']
mask=mask[1:90,1:90] 
print('shape of the mask',mask.shape)
# Define a transformation
transform = A.Compose([
    A.OpticalDistortion(distort_limit=(-1.25, 1.25), shift_limit=(0, 0), p=1.0), # Setting p=1 to ensure the transform is always applied
    #A.ElasticTransform(alpha=500, sigma=120 * 0.05, alpha_affine=120 * 0.03, p=1)
    A.GridDistortion(p=1)
])

# Convert mask to a 3-channel dummy image
fptdata_dummy = np.stack((mask,) * 3, axis=-1)

# Apply albumentations transforms
augmented = transform(image=image, mask=fptdata_dummy)
imgdata_aug = augmented['image']
fptdata_aug = augmented['mask'][:, :, 0]  # Get back the 2D mask

# Plotting
fig, ax = plt.subplots(2, 2, figsize=(10, 8))

# Original Image
ax[0, 0].imshow(image)
ax[0, 0].set_title('Original Image')
ax[0, 0].axis('off')

# Original Mask
ax[0, 1].imshow(mask, cmap='gray')
ax[0, 1].set_title('Original Mask')
ax[0, 1].axis('off')

# Augmented Image
ax[1, 0].imshow(imgdata_aug)
ax[1, 0].set_title('Augmented Image')
ax[1, 0].axis('off')

# Augmented Mask
ax[1, 1].imshow(fptdata_aug, cmap='gray')
ax[1, 1].set_title('Augmented Mask')
ax[1, 1].axis('off')

plt.tight_layout()
plt.show()
