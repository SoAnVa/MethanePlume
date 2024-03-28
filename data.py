import torch
import numpy  as np
import os
import rasterio as rio
from torchvision import transforms
import albumentations as A

class PlumeSegmentationDataset():
    """SmokePlumeSegmentation dataset class."""

    def __init__(self, datadir=None, segdir=None, band=[1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17], transform=None):
        """
        :param datadir: data directory
        :param segdir: label directory
        :param band: bands of the Sentinel-2 images to work with. A list of integer between 1 and 13 is expected corresponding to [B1,B2,B3,B4,B5,B6,B7,B8,B8a,B9,B10,B11,B12]
        and where 14=NDVI, 15=NDBI, 16=BSI, 
        :param transform: transformations to apply
        """
        
        self.datadir = datadir
        self.transform = transform
        self.band = band

        # list of image files, labels (positive or negative), segmentation
        self.imgfiles = []
        self.segfiles = []

        
        idx=0
        for root, dirs, files in os.walk(datadir):
            for filename in files:
              if not filename.endswith('.tif'):
                print("Not ending in .tif")
                print(filename)
                continue
              self.imgfiles.append(os.path.join(root, filename))
              segfilename = filename.replace(".tif", ".csv")
              self.segfiles.append(os.path.join(segdir, segfilename))
              idx+=1


        # turn lists into arrays
        self.imgfiles = np.array(self.imgfiles)
        self.segfiles = np.array(self.segfiles)


    def __len__(self):
        """Returns length of data set."""
        return len(self.imgfiles)


    def __getitem__(self, idx):
        """Read in image data, preprocess, build segmentation mask, and apply
        transformations;
        :param idx: idx of the item to get
        :return: sample ready to use
        """

        # read in image data
        imgfile = rio.open(self.imgfiles[idx], nodata = 0)
        imgdata = np.array([imgfile.read(i) for i in self.band if i not in [14,15,16,17]])

        # Calculate NDVI 
        if 14 in self.band:
            B4 = imgfile.read(4).astype(float)
            B8 = imgfile.read(8).astype(float)
            NDVI = (B8 - B4) / ( B8 + B4 +1e-5)
            NDVI = NDVI[None, :, :] # Ensure NDVI has three dimensions
            if len(self.band) == 1:
                imgdata=NDVI
            else: 
                imgdata = np.concatenate((imgdata, NDVI), axis=0)
        
        # Calculate NDBI
        if 15 in self.band:
            B12 = imgfile.read(12).astype(float)
            B8 = imgfile.read(8).astype(float)
            NDBI = (B12 - B8) / (B12 + B8 + 1e-5)
            NDBI = NDBI[None, :, :]  # Ensure NDBI has three dimensions

            if len(self.band) == 1:
                imgdata=NDBI
            else: 
                imgdata = np.concatenate((imgdata, NDBI), axis=0)
        

        # Calculate BSI
        if 16 in self.band:
            B2 = imgfile.read(2).astype(float)
            B4 = imgfile.read(4).astype(float)
            B8 = imgfile.read(8).astype(float)
            B12 = imgfile.read(12).astype(float)
            BSI = ((B12 + B4) - (B8 + B2)) / ((B12 + B4) + (B8 + B2) +1e-5)
            BSI = BSI[None, :, :]  # Ensure BSI has three dimensions

            if len(self.band) == 1:
                imgdata=BSI
            else: 
                imgdata = np.concatenate((imgdata, BSI), axis=0)
        
        # Calculate NDMI
        if 17 in self.band:
            B13 = imgfile.read(13).astype(float)
            B12 = imgfile.read(12).astype(float)
            NDMI = (B13 - B12) / (B13 + B12 +1e-5)
            NDMI = NDMI[None, :, :]  # Ensure NDMI has three dimensions

            if len(self.band) == 1:
                imgdata=NDMI
            else: 
                imgdata = np.concatenate((imgdata, NDMI), axis=0)
        
            
        fptdata = np.loadtxt(self.segfiles[idx], delimiter=",", dtype=float)
        fptdata = np.array(fptdata)

        sample = {'idx': idx,
                  'band' : self.band,
                  'img': imgdata,
                  'fpt': fptdata,
                  'imgfile': self.imgfiles[idx]}

        # apply transformations
        if self.transform:
            sample = self.transform(sample)

        return sample


class Crop(object):
    """Crop 90x90 pixel image in case the dimensions are wrong"""

    def __call__(self, sample):
        """
        :param sample: sample to be cropped
        :return: cropped sample
        """
        imgdata = sample['img']

        x, y = 0, 0

        return {'idx': sample['idx'],
                'band' : sample['band'],
                'img': imgdata.copy()[:, 1:90, 1:90],
                'fpt': sample['fpt'].copy()[1:90, 1:90],
                'imgfile': sample['imgfile']}

class Randomize(object):
    """Randomize image orientation including rotations by integer multiples of
       90 deg, (horizontal) mirroring, and (vertical) flipping."""
    
    def __init__(self):
        self.transform = A.Compose([
            A.RandomRotate90(p=0.5),
            A.Flip(p=0.5),
            A.Transpose(p=0.5),
            A.OpticalDistortion(distort_limit=(-1.25, 1.25), shift_limit=(0, 0), p=0.5), #Ajout de la distorsion pour essayer d'avoir des plumes plus complexes
            A.CLAHE(p=0.8), #augmenter le contraste des plumes avec le reste?
            A.RandomBrightnessContrast(p=0.8) #simuler des conditions meteo differentes 
        ])

    def __call__(self, sample):
        """
        :param sample: sample to be randomized
        :return: randomized sample
        """

        fptdata = sample['fpt']

        imgdata = sample['img'].transpose(1, 2, 0)
        fptdata_dummy = np.stack((fptdata,) * 3, axis=-1)

        # Apply albumentations transforms
        augmented = self.transform(image=imgdata, mask=fptdata_dummy)
        imgdata = augmented['image']
        fptdata = augmented['mask'][:, :, 0]  # Get back the 2D mask

        # # mirror horizontally
        # mirror = np.random.randint(0, 2)
        # if mirror:
        #     imgdata = np.flip(imgdata, 2)
        #     fptdata = np.flip(fptdata, 1)
        # # flip vertically
        # flip = np.random.randint(0, 2)
        # if flip:
        #     imgdata = np.flip(imgdata, 1)
        #     fptdata = np.flip(fptdata, 0)
        # # rotate by [0,1,2,3]*90 deg
        # rot = np.random.randint(0, 4)
        # imgdata = np.rot90(imgdata, rot, axes=(1,2))
        # fptdata = np.rot90(fptdata, rot, axes=(0,1))



        return {'idx': sample['idx'],
                'band' : sample['band'],
                'img': imgdata.transpose(2, 0, 1).copy(),
                'fpt': fptdata.copy(),
                'imgfile': sample['imgfile']}

class Normalize(object):
    """Normalize pixel values to zero mean and range [-1, +1] measured in
    standard deviations."""
    def __init__(self):
        
        self.channel_means = np.array([1909.3802, 1900.5879, 2261.5823, 3164.3564, 3298.6106, 3527.9346, 3791.7458, 3604.5210, 3946.0535, 1223.0176, 27.1881, 4699.9775, 3989.9626, 0.06484, -0.966756, -0.268266, 0.969461])
        self.channel_stds = np.array([498.8658,  507.0728,  573.1718,  965.0130, 1014.2232, 1069.5269, 1133.6522, 1073.3431, 1146.3250,  520.9219,   28.9335, 1360.9994, 1169.5753, 0.04521546, 0.132196162, 0.0554560, 0.133461])
    
    def __call__(self, sample):
        """
        :param sample: sample to be normalized
        :return: normalized sample
        """

        sample['img'] = (sample['img']-self.channel_means[np.array(sample['band'])-1].reshape(
            sample['img'].shape[0], 1, 1))/self.channel_stds[np.array(sample['band'])-1].reshape(
            sample['img'].shape[0], 1, 1)

        return sample
    
class NormalizePerImage(object):
    
     def __call__(self, sample):
        """
        Normalize each image individually by its mean and standard deviation.
        """

        img = sample['img']
        # Calculate mean and std for each channel of the current image
        means = img.mean(axis=(1, 2), keepdims=True)
        stds = img.std(axis=(1, 2), keepdims=True)

        # Avoid division by zero
        stds[stds == 0] = 1

        # Normalize the image
        normalized_img = (img - means) / stds

        # Update the sample
        sample['img'] = normalized_img
        return sample


class ToTensor(object):
    """Convert ndarrays in sample to Tensors."""
    def __call__(self, sample):
        """
        :param sample: sample to be converted to Tensor
        :return: converted Tensor sample
        """

        out = {'idx': sample['idx'],
               'band' : sample['band'],
               'img': torch.from_numpy(sample['img'].copy()),
               'fpt': torch.from_numpy(sample['fpt'].copy()),
               'imgfile': sample['imgfile']}

        return out

def create_dataset(*args, apply_transforms=True, **kwargs):
    """Create a dataset;
    :param apply_transforms: if `True`, apply Randomize transformations
    :return: data set"""
    if apply_transforms:
        data_transforms = transforms.Compose([
            Crop(),
            NormalizePerImage(),
            Randomize(),
            ToTensor()
           ])
    else:
        data_transforms = transforms.Compose([
            Crop(),
            NormalizePerImage(),
            ToTensor()
           ])

    data = PlumeSegmentationDataset(*args, **kwargs,
                                         transform=data_transforms)
    

    return data
