# Model definition of UNet architecture for segmentation of 3x256x256 -> 1x256x256

import torch
import torch.nn as nn
import torch.nn.functional as F

class Down(nn.Module):
    def __init__(self, in_ch, out_ch):
        super(Down, self).__init__()
        self.down = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, kernel_size=3, stride=1, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_ch, out_ch, kernel_size=3, stride=1, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2, 2)
        )
    def forward(self, x):
        return self.down(x)

class ConcatLayer(nn.Module):
    def __init__(self):
        super(ConcatLayer, self).__init__()
    def forward(self, x1, x2):
        return torch.cat((x1, x2), dim=1)

class Up(nn.Module):
    def __init__(self, in_ch, out_ch):
        super(Up, self).__init__()
        self.up_part1 = nn.Sequential(
            nn.Upsample(scale_factor=2, mode='bilinear'),
            nn.Conv2d(in_ch, in_ch, kernel_size=3, stride=1, padding=1),
            nn.ReLU(inplace=True)
        )
        self.concat = ConcatLayer()
        self.up_part2 = nn.Sequential(
            nn.Conv2d(in_ch+out_ch, out_ch, kernel_size=3, stride=1, padding=1),
            nn.ReLU(inplace=True)
        )

    def forward(self, x1, x2):
        x = self.up_part1(x1)
        x = self.concat(x,x2)
        x = self.up_part2(x)
        return x


class UNet(nn.Module):
    def __init__(self, n_channels, n_classes):
        super(UNet, self).__init__()
        self.inc = nn.Conv2d(n_channels, 32, kernel_size=3, stride=1, padding=1)
        self.down1 = Down(32, 64)
        self.down2 = Down(64, 128)
        self.down3 = Down(128, 256)
        self.down4 = Down(256, 512)
        self.down5 = Down(512, 1024)
        
        self.up1 = Up(1024, 512)
        self.up2 = Up(512, 256)
        self.up3 = Up(256, 128)
        self.up4 = Up(128, 64)
        self.up5 = Up(64, 32)
        self.outc = nn.Conv2d(32,n_classes,kernel_size=3, stride=1, padding=1)
        self.out_sig = nn.Sigmoid()

    def forward(self, x):
        x0 = self.inc(x) # Bx3x256x256 -> Bx32x256x256
        x1 = self.down1(x0) # Bx32x256x256 -> Bx64x128x128
        x2 = self.down2(x1) # Bx64x128x128 -> Bx128x64x64
        x3 = self.down3(x2) # Bx128x64x64 -> Bx256x32x32
        x4 = self.down4(x3) # Bx256x16x16 -> Bx512x16x16
        x5 = self.down5(x4) # Bx512x16x16 -> Bx1024x8x8
    
        x = self.up1(x5,x4) # Bx1024x8x8, Bx512x16x16 -> Bx512x16x16
        x = self.up2(x,x3) # Bx512x16x16, Bx256x32x32 -> Bx256x32x32
        x = self.up3(x,x2) # Bx256x32x32, Bx128x64x64 -> Bx128x64x64
        x = self.up4(x,x1) # Bx128x64x64, Bx64x128x128 -> Bx64x128x128
        x = self.up5(x,x0) # Bx64x128x128, Bx32x256x256 -> Bx32x256x256
        x = self.outc(x) # Bx32x256x256 -> Bx1x256x256
        return self.out_sig(x)


device = 'cuda:1'
model = UNet(n_channels=3, n_classes=1)
model.to(device)
# x = torch.randn(1, 3, 256, 256) # Example input (batch size, channels, height, width)
# y = model(x.to(device))
# print(y.shape)  # Should print torch.Size([1, 1, 256, 256])

# Read the data

import glob
def g(x): return glob.glob(x)

tr_img_path = '/disk2/akm_files/akm_exp/RandomExp/seg_exp/duts/DUTS-TR/DUTS-TR-Image'
te_img_path = '/disk2/akm_files/akm_exp/RandomExp/seg_exp/duts/DUTS-TE/DUTS-TE-Image'

all_tr_list = g(tr_img_path+'/*')
all_te_list = g(te_img_path+'/*')

# print(len(all_tr_list))
# print(len(all_te_list))

# Check the data that has been read

import numpy as np
import matplotlib.pyplot as plt
from PIL import Image

# img = np.array(Image.open(all_tr_list[0]))
# mask = np.array(Image.open(all_tr_list[0].replace('/DUTS-TR-Image','/DUTS-TR-Mask').replace('.jpg','.png')).convert('L'))

# print(img.shape)
# print(mask.shape)

# print(np.unique(img))
# print(np.unique(mask))

# plt.imshow(img)
# plt.show()
# plt.imshow(mask)
# plt.show()

from torch.utils.data import Dataset, DataLoader
from torchvision import transforms

class SegmentationDataset(Dataset):
    def __init__(self, image_list, transform=None):
        self.image_list = image_list
        self.transform = transform

    def __len__(self):
        return len(self.image_list)

    def __getitem__(self, idx):
        img_path = self.image_list[idx]
        mask_path = img_path.replace('/DUTS-TR-Image','/DUTS-TR-Mask').replace('.jpg','.png')
        
        image = Image.open(img_path).convert("RGB")
        mask = Image.open(mask_path).convert("L")  # Grayscale for a single channel mask

        if self.transform:
            image = self.transform(image)
            mask = self.transform(mask)
        
        return image, mask

transform = transforms.Compose([
    transforms.Resize((256, 256)),
    transforms.ToTensor()  # Scales data to [0, 1] automatically
])

train_list = all_tr_list[:9000]
val_list = all_tr_list[9000:]

batch_size = 200

tr_dataset = SegmentationDataset(train_list, transform=transform)
train_loader = DataLoader(tr_dataset, batch_size=batch_size, shuffle=True)

val_dataset = SegmentationDataset(val_list, transform=transform)
val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)

# # Checking some data
# for images, masks in train_loader:
#     print(images.shape, masks.shape)  # Expected: torch.Size([batch, 3, 256, 256]) and torch.Size([batch, 1, 256, 256])
#     break

# for images, masks in val_loader:
#     print(images.shape, masks.shape)  # Expected: torch.Size([batch, 3, 256, 256]) and torch.Size([batch, 1, 256, 256])
#     break

def iou_score(output, target):
    smooth = 1e-6  # A small value to avoid division by zero
    if torch.is_tensor(output):
        output = output > 0.5  # Thresholding
        output = output.int()
    if torch.is_tensor(target):
        target = target.int()

    intersection = (output & target).float().sum((1, 2))  # Will sum each channel
    union = (output | target).float().sum((1, 2))  # Will sum each channel

    iou = (intersection + smooth) / (union + smooth)
    return iou.mean()  # Mean over batch

def iou_loss(preds, labels):
    smooth = 1e-6
    # # Sigmoid activation to convert logits to probabilities
    # preds = torch.sigmoid(preds)
    # Flatten label and prediction tensors
    preds = preds.view(-1)
    labels = labels.view(-1)
    
    # Intersection and Union
    intersection = (preds * labels).sum()
    total = (preds + labels).sum()
    union = total - intersection
    
    IoU = (intersection + smooth) / (union + smooth)
    return 1 - IoU  # IoU loss

import torch.optim as optim
from tqdm import tqdm

def train_one_epoch(model, dataloader, optimizer, device):
    model.train()
    total_loss = 0
    total_iou = 0

    for images, masks in tqdm(dataloader):
        images, masks = images.to(device), masks.to(device)
        
        optimizer.zero_grad()
        outputs = model(images)
        
        bce_loss = nn.BCELoss()(outputs, masks)
        iou_loss_val = iou_loss(outputs, masks)
        loss = (bce_loss + 10* iou_loss_val)  # Combining the losses
        
        loss.backward()
        optimizer.step()

        total_loss += loss.item() * images.size(0)
        total_iou += iou_score(outputs, masks).item()

    avg_loss = total_loss / len(dataloader.dataset)
    avg_iou = total_iou / len(dataloader.dataset)
    return avg_loss, avg_iou

def validate(model, dataloader, device):
    model.eval()
    total_loss = 0
    total_iou = 0

    with torch.no_grad():
        for images, masks in tqdm(dataloader):
            images, masks = images.to(device), masks.to(device)
            outputs = model(images)
            
            bce_loss = nn.BCEWithLogitsLoss()(outputs, masks)
            iou_loss_val = iou_loss(outputs, masks)
            loss = (bce_loss + 10*iou_loss_val)  # Combining the losses
            
            total_loss += loss.item() * images.size(0)
            total_iou += iou_score(outputs, masks).item()

    avg_loss = total_loss / len(dataloader.dataset)
    avg_iou = total_iou / len(dataloader.dataset)
    return avg_loss, avg_iou

device = 'cuda:1'
model = UNet(n_channels=3, n_classes=1).to(device)
optimizer = optim.Adam(model.parameters(), lr=1e-3)
criterion = nn.BCELoss()

num_epochs = 100
best_val_loss = float('inf')
ckpt_dir = '/disk2/akm_files/akm_exp/RandomExp/seg_exp/unet_duts_ckpts'

for epoch in range(num_epochs):
    train_loss, train_iou = train_one_epoch(model, train_loader, optimizer, device)
    val_loss, val_iou = validate(model, val_loader, device)
    
    print(f'Epoch {epoch+1}/{num_epochs}, Train Loss: {train_loss:.4f}, Train IoU: {train_iou:.4f}, Val Loss: {val_loss:.4f}, Val IoU: {val_iou:.4f}')
    
    if val_loss < best_val_loss:
        best_val_loss = val_loss
        torch.save(model.state_dict(), f'{ckpt_dir}/best_model.pth')
        print('Saved Best Model')


from torchvision.transforms.functional import to_pil_image

def predict_segmentation(image_np, model_weights_path):
    device = 'cuda:1'
    
    # Load the model
    model = UNet(n_channels=3, n_classes=1).to(device)
    model.load_state_dict(torch.load(model_weights_path))
    model.eval()

    # Prepare the image
    transform = transforms.Compose([
        transforms.Resize((256, 256)),
        transforms.ToTensor()
    ])
    
    image = Image.fromarray(image_np)  # Convert numpy array to PIL Image
    image = transform(image).unsqueeze(0).to(device)  # Apply transform and add batch dimension

    # Predict the segmentation
    with torch.no_grad():
        output = model(image)
        output = torch.sigmoid(output)  # Convert to probability
        output = output.squeeze(0).cpu()  # Remove batch dimension and move to CPU

    # Rescale the output to original image size and convert to 0-255 range
    output_pil = to_pil_image(output)  # Convert tensor to PIL image
    output_pil = output_pil.resize((image_np.shape[1], image_np.shape[0]), Image.NEAREST)  # Resize back to original
    output_np = np.array(output_pil)  # Convert PIL image to numpy array
    output_np = (output_np * 255).astype(np.uint8)  # Scale to 0-255 range

    return output_np

def display_images(img, gt_mask, pred_mask):
    # Assuming img, gt_mask, pred_mask are numpy arrays of shape (H, W, C) where C is 1 or 3
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))  # Create a figure with three subplots

    # Titles for each subplot
    titles = ['img', 'gt_mask', 'pred_mask']

    # Images to display
    images = [img, gt_mask, pred_mask]
    
    for ax, image, title in zip(axes, images, titles):
        # If the image is a single-channel grayscale (H, W, 1), convert it to (H, W) for imshow
        if image.ndim == 3 and image.shape[2] == 1:
            image = image.squeeze(-1)
        
        ax.imshow(image, cmap='gray' if image.ndim == 2 else None)
        ax.set_title(title)
        ax.axis('off')  # Hide axes ticks

    plt.tight_layout()
    plt.show()

# for i in range(len(all_te_list)):
#     img = np.array(Image.open(all_te_list[i]))
#     gt_mask = np.array(Image.open(all_te_list[i].replace('/DUTS-TE-Image','/DUTS-TE-Mask').replace('.jpg','.png')).convert('L'))
#     weights_path = f'{ckpt_dir}/best_model.pth'

#     pred_mask = predict_segmentation(img, weights_path)

#     display_images(img, gt_mask, pred_mask)

#     if i == 10:
#         break

# Create Test Loader so that we can calculate test dataset IoU score.

class TestDataset(Dataset):
    def __init__(self, image_list, transform=None):
        self.image_list = image_list
        self.transform = transform

    def __len__(self):
        return len(self.image_list)

    def __getitem__(self, idx):
        img_path = self.image_list[idx]
        mask_path = img_path.replace('/DUTS-TE-Image','/DUTS-TE-Mask').replace('.jpg','.png')
        
        image = Image.open(img_path).convert("RGB")
        mask = Image.open(mask_path).convert("L")  # Grayscale for a single channel mask

        if self.transform:
            image = self.transform(image)
            mask = self.transform(mask)
        
        return image, mask

transform = transforms.Compose([
    transforms.Resize((256, 256)),
    transforms.ToTensor()  # Scales data to [0, 1] automatically
])

te_dataset = TestDataset(all_te_list, transform=transform)
test_loader = DataLoader(te_dataset, batch_size=1, shuffle=False)

print(len(test_loader))

def evaluate_iou(weights_path, test_loader, device):
    # Load the model
    model = UNet(n_channels=3, n_classes=1).to(device)
    model.load_state_dict(torch.load(weights_path))
    model.eval()
    
    total_iou = 0.0
    count = 0

    with torch.no_grad():
        for images, true_masks in test_loader:
            images = images.to(device)
            true_masks = true_masks.to(device)

            # Predict masks
            predicted_masks = model(images)

            # Calculate IoU score
            iou = iou_score(predicted_masks, true_masks)
            total_iou += iou.item()
            count += 1

    average_iou = total_iou / count
    return average_iou

weights_path = f'{ckpt_dir}/best_model.pth'

test_iou = evaluate_iou(weights_path, test_loader, 'cuda:1')
print('The IoU score on the test set is: ',test_iou)