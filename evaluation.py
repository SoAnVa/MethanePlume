import torch
import os
from torch.utils.data import DataLoader
from model_unet import UNet  
from data import create_dataset  
from torchmetrics import JaccardIndex
import numpy as np
import matplotlib.pyplot as plt

def evaluate_model(model_path, test_data_path, test_labels_path, batch_size=1):
    device = torch.device("cpu")  # CPU
    print(f"Evaluating model on {device}")


    bands=[1,2,3,4,5,6,7,8,9,10,11,12,13]
    bands=[11,12,13,14,15,16,17]
    # Load le model
    model = UNet(n_channels=len(bands), n_classes=1)
    model.load_state_dict(torch.load(model_path, map_location=torch.device('cpu')))
    model.eval()
    # cahrger les datasets
    test_dataset = create_dataset(datadir=test_data_path, segdir=test_labels_path, band=bands, apply_transforms=False)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)
    jaccard_index = JaccardIndex(num_classes=2, task="binary").to(device)
    
    # Evaluate the model
    ious = []

    with torch.no_grad():
        for batch in test_loader:
            images = batch['img'].float().to(device)
            true_masks = batch['fpt'].float().to(device)

            predictions = model(images)
            predictions = torch.sigmoid(predictions)  
            predictions = (predictions > 0.5).float() 
             # IOU
            iou = jaccard_index(predictions, true_masks.unsqueeze(dim=1))
            ious.append(iou.item())
            #print('-------',iou.item())

            # if iou.item()>0.5:
            #     # Visualization
            #     fig, ax = plt.subplots(1, 2, figsize=(10, 5))
            #     ax[0].imshow(true_masks[0].cpu().squeeze(), cmap='gray')
            #     ax[0].set_title('True Mask')
            #     ax[0].axis('off')

            #     ax[1].imshow(predictions[0].cpu().squeeze().numpy(), cmap='gray')
            #     ax[1].set_title('Predicted Mask')
            #     ax[1].axis('off')

            #     plt.show()

    ious = np.array(ious)
    #ious = ious[~np.isnan(ious)]  # enlever les Nan

    avg_iou = np.mean(ious)
    std_iou = np.std(ious)
    max_iou = np.max(ious)
    min_iou = np.min(ious)
    print("for the model ", model_path)
    print(f"Mean IoU on the test set: {avg_iou}")
    print(f"STD IoU on the test set: {std_iou}")
    print(f"Max-min IoU on the test set: {max_iou} - {min_iou}")


if __name__ == "__main__":
    print("Script started")
    model_path = "mod/ep50_lr0.01_bs30_time15 30 10_idd91_Lossv3_smooth1p1.75_adamW_2lr_Disto.model" #Bien faire Attention à utiliser les memes paraletres que pour l'entrainement de ce modele
    test_data_path = "data/DataTest/input_tiles"
    test_labels_path = "data/DataTest/output_matrix"
    evaluate_model(model_path, test_data_path, test_labels_path)

