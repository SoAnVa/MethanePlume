import matplotlib.pyplot as plt
import numpy as np
import torch
from data import create_dataset  


data_dir = "data/DataTrain/input_tiles"
mask_dir = "data/DataTrain/output_matrix"
bands = [4,3,2] 
#bands = [1,2,3,4,5,6,7,8,9,10,11,12,13,17]

# Créer le dataset
dataset = create_dataset(datadir=data_dir, segdir=mask_dir, band=bands, apply_transforms=False)
#dataset_augment = create_dataset(datadir=data_dir, segdir=mask_dir, band=bands, apply_transforms=True)

def visualize_dataset_item(dataset, index):
    # Récupérer un élément de l'ensemble de données
    sample = dataset[index]
    image = sample['img']
    mask = sample['fpt']

    # Transposer l'image si nécessaire (C, H, W) -> (H, W, C)
    if image.shape[0] < image.shape[2]:
        image = np.transpose(image, (1, 2, 0))

    # image_NDBI=(image[:,:,11]-image[:,:,10])/(image[:,:,11]+image[:,:,10] + 1e-5)
    # mean=image_NDBI.mean(axis=(0,1))
    # std=image_NDBI.std(axis=(0,1))
    # image_NDBI = (image_NDBI - mean)/std

    # print(image_NDBI)
    # print(image[:,:,-1])
    #print(np.array(image).shape)
    
    #plt.imshow(image_NDBI, cmap="plasma")
    # plt.colorbar(label='NDBI Value')
    # print('shape',image.shape)
    # plt.show()

    #Assurez-vous que l'image a les dimensions correctes (H, W)
    # if image.shape[2] == 1:  # Si C=1
    #     print('test')
    #     image = image.squeeze(0)  # Supprimer la dimension des canaux

        

    fig, ax = plt.subplots(1, 2, figsize=(12, 6))

    im0=ax[0].imshow(image, cmap='plasma')

    ax[0].set_title('Image - NDVI')
    ax[0].axis('off')
    fig.colorbar(im0)
    # Si un masque est présent
    if mask is not None:
        if isinstance(mask, torch.Tensor):
            mask = mask.numpy()
        ax[1].imshow(mask, cmap='plasma')
        ax[1].set_title('Mask')
        ax[1].axis('off')

    
    plt.show()

# Visualiser un élément spécifique de l'ensemble de données
index = 1  # Changer selon l'élément que vous voulez visualiser
visualize_dataset_item(dataset, index)
#visualize_dataset_item(dataset_augment, index)