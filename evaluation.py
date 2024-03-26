import torch
import os
from torch.utils.data import DataLoader
from model_unet import UNet  # Make sure this import matches your model file
from data import create_dataset  # Adjust this import based on your dataset preparation file
from torchmetrics import JaccardIndex
import numpy as np

def evaluate_model(model_path, test_data_path, test_labels_path, batch_size=1):
    device = torch.device("cpu")  # Explicitly setting device to CPU
    print(f"Evaluating model on {device}")

    # Load the model
    model = UNet(n_channels=13, n_classes=1).to(device)
    model.load_state_dict(torch.load(model_path, map_location=torch.device('cpu')))
    model.eval()

    # Prepare the test dataset
    test_dataset = create_dataset(datadir=test_data_path, segdir=test_labels_path, band=[1,2,3,4,5,6,7,8,9,10,11,12,13], apply_transforms=False)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

    # Initialize the Jaccard Index metric
    jaccard_index = JaccardIndex(num_classes=2, task="binary").to(device)
    
    # Evaluate the model
    ious = []
    with torch.no_grad():
        for batch in test_loader:
            images = batch['img'].float().to(device)
            true_masks = batch['fpt'].float().to(device)

            predictions = model(images)
            predictions = torch.sigmoid(predictions)  # Apply sigmoid to get [0, 1] range
            predictions = (predictions > 0.5).float()  # Binarize predictions

            # Compute IoU
            iou = jaccard_index(predictions, true_masks.unsqueeze(dim=1))
            ious.append(iou.item())
            print('-------',iou.item())
    # Compute average IoU
    ious = np.array(ious)
    ious = ious[~np.isnan(ious)]  # Remove NaN values from IoU scores

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
    model_path = "mod/ep50_lr0.01_bs30_time12:23:46_idd31.model"
    test_data_path = "/Users/vakili/Documents/Methane-Plume-Segmentation-main/data/DataTrain/input_tiles"
    test_labels_path = "/Users/vakili/Documents/Methane-Plume-Segmentation-main/data/DataTrain/output_matrix"
    evaluate_model(model_path, test_data_path, test_labels_path)

