import numpy as np
import rasterio as rio
 # Make sure this import matches your dataset class
from data import create_dataset 

def calculate_ndvi_mean_std(dataset):
    ndvi_values = []

    for idx in range(len(dataset)):
        # Read the specific bands needed for NDVI calculation
        with rio.open(dataset.imgfiles[idx]) as imgfile:
            B11 = imgfile.read(11).astype(float)  
            B12 = imgfile.read(12).astype(float)
            B2 = imgfile.read(2).astype(float)  
            B4 = imgfile.read(4).astype(float)  

        # Calculate NDVI
        NDVI = (B12 - B11) / (B12 + B11 +1e-5)
        if idx==1:
            print(NDVI)
            return NDVI

        # Flatten NDVI and add to list
        ndvi_values.append(NDVI.flatten())

    # Concatenate all NDVI values from the dataset to compute overall stats
    all_ndvi_values = np.concatenate(ndvi_values)

    # Compute mean and standard deviation
    mean_ndvi = np.mean(all_ndvi_values)
    std_ndvi = np.std(all_ndvi_values)
    print( np.mean( (all_ndvi_values-mean_ndvi)/std_ndvi) )
          
    return mean_ndvi, std_ndvi

# Assuming your dataset is initialized as follows
data_dir = "data/DataTrain/input_tiles"
mask_dir = "data/DataTrain/output_matrix"
dataset = create_dataset(datadir=data_dir, segdir=mask_dir, band=[1,2,4,5], apply_transforms=False)

# Calculate mean and standard deviation of NDVI
mean_ndvi, std_ndvi = calculate_ndvi_mean_std(dataset)
#print(f"NDVI Mean: {mean_ndvi}, NDVI Std: {std_ndvi}")