import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from torch.optim.lr_scheduler import ReduceLROnPlateau
from tqdm.auto import tqdm, trange
from torchvision import transforms
import os
import numpy as np


from PIL import Image
import matplotlib.pyplot as plt
from tqdm import tqdm
import random
import time
import torchvision.transforms.functional as TF

from config import Config
from models import carla_lane_model
from unet_models import carla_lane_unet

import warnings
warnings.filterwarnings('ignore')

device = 'cuda' if torch.cuda.is_available() else 'cpu'

train_image_dir = './dataset/train'
train_mask_dir = './dataset/train_label'
val_image_dir = './dataset/val'
val_mask_dir = './dataset/val_label'

class SegmentationDataset(Dataset):
    def __init__(self, image_dir, label_dir, transform=None):
        self.image_dir = image_dir
        self.label_dir = label_dir
        self.transform = transform
        self.images = os.listdir(image_dir)
    
    def __len__(self):
        return len(self.images)
    
    def __getitem__(self, idx):
        image_path = os.path.join(self.image_dir, self.images[idx])
        label_name = self.images[idx].replace('.png', '_label.png')
        label_path = os.path.join(self.label_dir, label_name)
        
        
        image = Image.open(image_path).convert('RGB')
        mask = Image.open(label_path).convert('L')  
        
        
        if self.transform:
            image = self.transform(image)
            mask = TF.resize(mask, (256, 256), interpolation=Image.NEAREST)
        
        mask = torch.from_numpy(np.array(mask)).long()  
        
        return image, mask
    
train_transform_resize = (256,256)

train_transform = transforms.Compose([
    transforms.Resize(train_transform_resize),
    transforms.ToTensor(),
])


train_dataset = SegmentationDataset(image_dir=train_image_dir, label_dir=train_mask_dir, transform=train_transform)
train_loader = DataLoader(train_dataset, batch_size=16, shuffle=True)

val_dataset = SegmentationDataset(image_dir=val_image_dir, label_dir=val_mask_dir, transform=train_transform)
val_loader = DataLoader(val_dataset, batch_size=16, shuffle=False)

model = carla_lane_unet(3).to(device)

criterion = nn.CrossEntropyLoss()

optimizer = optim.Adam(model.parameters(), lr=0.001)

def calculate_iou(preds, masks, num_classes):
    ious = []
    preds = torch.argmax(preds, dim=1)
    for cls in range(num_classes):
        pred_cls = (preds == cls).float()
        mask_cls = (masks == cls).float()
        
        intersection = torch.sum(pred_cls * mask_cls)
        union = torch.sum(pred_cls) + torch.sum(mask_cls) - intersection
        
        if union == 0:
            ious.append(1.0)
        else:
            ious.append((intersection / union).item())
    
    return sum(ious) / len(ious)

def calculate_dice(preds, masks, num_classes):
    dices = []
    preds = torch.argmax(preds, dim=1) 
    for cls in range(num_classes):
        pred_cls = (preds == cls).float()
        mask_cls = (masks == cls).float()
        
        intersection = torch.sum(pred_cls * mask_cls)
        dice = (2 * intersection) / (torch.sum(pred_cls) + torch.sum(mask_cls))
        
        if torch.sum(pred_cls) + torch.sum(mask_cls) == 0:
            dices.append(1.0)
        else:
            dices.append(dice.item())
    
    return sum(dices) / len(dices)


num_epochs = 6
train_loss_list = []
val_loss_list = []
iou_list = []
dice_list = []

num_classes = 3  # Number of segmentation classes

# Learning rate scheduler (reduces LR when validation loss plateaus)
scheduler = ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=2)

# Early stopping setup
early_stopping_patience = 5
early_stopping_counter = 0
best_val_loss = float('inf')

for epoch in range(num_epochs):
    model.train()
    running_loss = 0.0
    
    # Training loop
    for images, masks in train_loader:
        images = images.to(device)
        masks = masks.to(device).long()
        
        optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs, masks)
        loss.backward()
        optimizer.step()
        
        running_loss += loss.item()

    # Validation loop
    model.eval()
    val_loss = 0.0
    iou_total = 0.0
    dice_total = 0.0
    with torch.no_grad():
        for images, masks in val_loader:
            images = images.to(device)
            masks = masks.to(device).long()
            
            outputs = model(images)
            loss = criterion(outputs, masks)
            val_loss += loss.item()
            
            iou = calculate_iou(outputs, masks, num_classes=num_classes)
            dice = calculate_dice(outputs, masks, num_classes=num_classes)
            
            iou_total += iou
            dice_total += dice
    
    # Calculate averages
    avg_train_loss = running_loss / len(train_loader)
    avg_val_loss = val_loss / len(val_loader)
    avg_iou = iou_total / len(val_loader)
    avg_dice = dice_total / len(val_loader)
    
    train_loss_list.append(avg_train_loss)
    val_loss_list.append(avg_val_loss)
    iou_list.append(avg_iou)
    dice_list.append(avg_dice)
    
    print(f"Epoch [{epoch+1}/{num_epochs}]")
    print(f"Training Loss: {avg_train_loss:.4f}, Validation Loss: {avg_val_loss:.4f}")
    print(f"MIoU: {avg_iou:.4f}, Dice Coefficient: {avg_dice:.4f}")

    # Update learning rate scheduler
    scheduler.step(avg_val_loss)

    # Early stopping logic
    if avg_val_loss < best_val_loss:
        best_val_loss = avg_val_loss
        early_stopping_counter = 0
        print("Validation loss improved, resetting early stopping counter.")
    else:
        early_stopping_counter += 1
        print(f"Validation loss did not improve for {early_stopping_counter} epochs.")
        if early_stopping_counter >= early_stopping_patience:
            print(f"Early stopping triggered at epoch {epoch+1}.")
            break


def plot_metrics(train_loss_list, val_loss_list, iou_list, dice_list):
    epochs = range(1, len(train_loss_list) + 1)

    # Plot loss values
    plt.figure(figsize=(10, 5))
    plt.plot(epochs, train_loss_list, label='Train Loss')
    plt.plot(epochs, val_loss_list, label='Validation Loss')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.title('Training and Validation Loss')
    plt.legend()
    plt.grid(True)
    plt.show()

    # Plot MIoU and Dice Coefficient
    plt.figure(figsize=(10, 5))
    plt.plot(epochs, iou_list, label='Mean IoU', color='blue', alpha=0.5)
    plt.plot(epochs, dice_list, label='Dice Coefficient', color='green', alpha=0.5)
    plt.xlabel('Epochs')
    plt.ylabel('Metric Value')
    plt.title('MIoU and Dice Coefficient Over Epochs')
    plt.legend()
    plt.grid(True)
    plt.show()

# plot_metrics and visualize_random_predictions remain unchanged


def visualize_random_predictions(model, dataloader, device, num_images=5):
    model.eval()  # Set model to evaluation mode
    
    dataset_size = len(dataloader.dataset)
    random_indices = random.sample(range(dataset_size), num_images)
    
    images_so_far = 0
    fig, ax = plt.subplots(num_images, 3, figsize=(10, num_images * 5))
    
    with torch.no_grad():
        for batch_idx, (images, masks) in enumerate(dataloader):
            batch_size = images.size(0)
            start_idx = batch_idx * batch_size
            end_idx = start_idx + batch_size

            valid_indices = [i for i in random_indices if start_idx <= i < end_idx]
            
            if len(valid_indices) == 0:
                continue
            
            images = images.to(device)
            masks = masks.to(device)
            
            # Get predictions
            outputs = model(images)
            preds = torch.argmax(outputs, dim=1)
            
            for i in valid_indices:
                local_idx = i - start_idx
                
                pred_mask = preds[local_idx].cpu().numpy()
                true_mask = masks[local_idx].cpu().numpy()
                
                ax[images_so_far, 0].imshow(images[local_idx].cpu().permute(1, 2, 0))
                ax[images_so_far, 0].set_title('Input Image')
                
                ax[images_so_far, 1].imshow(true_mask, cmap='gray')
                ax[images_so_far, 1].set_title('True Mask')
                
                ax[images_so_far, 2].imshow(pred_mask, cmap='gray')
                ax[images_so_far, 2].set_title('Predicted Mask')
                
                images_so_far += 1
                
                if images_so_far == num_images:
                    plt.tight_layout()
                    plt.show()
                    return

    plt.tight_layout()
    plt.show()

visualize_random_predictions(model, val_loader, device, num_images=5)