import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
import numpy as np
from sklearn.model_selection import train_test_split
from torchvision.transforms import Compose, Normalize, ToTensor
import random

# Define a simple 3D nnU-Net-like architecture
class Simple3DUNet(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(Simple3DUNet, self).__init__()
        self.encoder1 = self.conv_block(in_channels, 32)
        self.encoder2 = self.conv_block(32, 64)
        self.encoder3 = self.conv_block(64, 128)
        self.decoder3 = self.conv_block(128, 64)
        self.decoder2 = self.conv_block(64, 32)
        self.decoder1 = nn.Conv3d(32, out_channels, kernel_size=1)

        self.pool = nn.MaxPool3d(2)
        self.up = nn.ConvTranspose3d(128, 128, kernel_size=2, stride=2)

    def conv_block(self, in_channels, out_channels):
        return nn.Sequential(
            nn.Conv3d(in_channels, out_channels, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv3d(out_channels, out_channels, kernel_size=3, padding=1),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        enc1 = self.encoder1(x)
        enc2 = self.encoder2(self.pool(enc1))
        enc3 = self.encoder3(self.pool(enc2))

        dec3 = self.up(enc3)
        dec3 = self.decoder3(dec3 + enc2)
        dec2 = self.decoder2(dec3 + enc1)
        dec1 = self.decoder1(dec2)
        return dec1

# Dummy dataset for demonstration
class DummyMRIDataset(Dataset):
    def __init__(self, num_samples, img_size=(64, 64, 64)):
        self.num_samples = num_samples
        self.img_size = img_size
        self.data = [np.random.rand(*img_size).astype(np.float32) for _ in range(num_samples)]
        self.labels = [np.random.randint(0, 2, img_size).astype(np.float32) for _ in range(num_samples)]

    def __len__(self):
        return self.num_samples

    def __getitem__(self, idx):
        img = self.data[idx]
        label = self.labels[idx]
        transform = Compose([ToTensor(), Normalize(mean=[0.5], std=[0.5])])
        img = transform(img)
        label = torch.tensor(label, dtype=torch.float32)
        return img, label

# Dice loss function
def dice_loss(pred, target):
    smooth = 1.0
    pred = torch.sigmoid(pred)
    intersection = (pred * target).sum()
    dice = (2. * intersection + smooth) / (pred.sum() + target.sum() + smooth)
    return 1 - dice

# Training function
def train_model(model, dataloader, optimizer, criterion, device):
    model.train()
    epoch_loss = 0
    for images, labels in dataloader:
        images, labels = images.to(device), labels.to(device)
        optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
        epoch_loss += loss.item()
    return epoch_loss / len(dataloader)

# Validation function
def validate_model(model, dataloader, criterion, device):
    model.eval()
    epoch_loss = 0
    with torch.no_grad():
        for images, labels in dataloader:
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            loss = criterion(outputs, labels)
            epoch_loss += loss.item()
    return epoch_loss / len(dataloader)

if __name__ == '__main__':
    # Hyperparameters
    num_epochs = 5
    batch_size = 2
    learning_rate = 0.001
    img_size = (64, 64, 64)
    num_samples = 20

    # Prepare dataset
    dataset = DummyMRIDataset(num_samples, img_size)
    train_data, val_data = train_test_split(dataset, test_size=0.2, random_state=42)
    train_loader = DataLoader(train_data, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_data, batch_size=batch_size)

    # Initialize model, optimizer, and loss function
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = Simple3DUNet(in_channels=1, out_channels=1).to(device)
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)
    criterion = dice_loss

    # Training loop
    for epoch in range(num_epochs):
        train_loss = train_model(model, train_loader, optimizer, criterion, device)
        val_loss = validate_model(model, val_loader, criterion, device)
        print(f"Epoch {epoch+1}/{num_epochs}, Train Loss: {train_loss:.4f}, Val Loss: {val_loss:.4f}")