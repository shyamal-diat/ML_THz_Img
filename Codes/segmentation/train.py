import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torch.optim import Adam
from torchvision.utils import save_image
from data import TerahertzDataset, train_val_test_split
from model import UNet
from torchvision import transforms
import matplotlib.pyplot as plt

# Set device
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# Set hyperparameters
num_epochs = 100
batch_size = 1
learning_rate = 0.01

# Load data
data_transform = transforms.ToTensor()
dataset = TerahertzDataset(root_dir='D:\THz\segmentation_new', transform=data_transform)
train_indices, val_indices, test_indices = train_val_test_split(dataset, train_percent=.8, val_percent=.1, test_percent=.1, shuffle=True)

train_loader = DataLoader(dataset, batch_size=batch_size, sampler=train_indices)
val_loader = DataLoader(dataset, batch_size=batch_size, sampler=val_indices)
test_loader = DataLoader(dataset, batch_size=batch_size, sampler=test_indices)

# Initialize model and loss function
model = UNet().to(device)
criterion = nn.MSELoss()

# Initialize optimizer
optimizer = Adam(model.parameters(), lr=learning_rate)

# Initialize lists for plotting
train_loss_list = []
val_loss_list = []

# Training loop
for epoch in range(num_epochs):
    epoch_train_loss = 0.0
    epoch_val_loss = 0.0
    num_train_batches = 0
    num_val_batches = 0
    for i, (hr_image, hr_mask) in enumerate(train_loader):
        hr_image = hr_image.to(device)
        hr_mask = hr_mask.to(device)

        # Forward pass
        sr_image = model(hr_image)

        # Compute loss
        loss = criterion(sr_image, hr_mask)

        # Backward pass and optimization
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        # Append training loss to list
        epoch_train_loss += loss.item()
        num_train_batches += 1

        # Print progress
        if (i + 1) % 100 == 0:
            print(f'Epoch [{epoch + 1}/{num_epochs}], Step [{i + 1}/{len(train_loader)}], Loss: {loss.item():.4f}')

    # Compute average training loss for the epoch
    epoch_train_loss /= num_train_batches
    train_loss_list.append(epoch_train_loss)

    # Validate after each epoch
    with torch.no_grad():
        val_loss = 0
        for hr_image, hr_mask in val_loader:
            hr_image = hr_image.to(device)
            hr_mask = hr_mask.to(device)

            # Forward pass
            sr_image = model(hr_image)

            # Compute loss
            val_loss = criterion(sr_image, hr_mask)

            # Append validation loss to list
            epoch_val_loss += val_loss.item()
            num_val_batches += 1

        # Compute average validation loss for the epoch
        epoch_val_loss /= num_val_batches
        val_loss_list.append(epoch_val_loss)

        # Print validation loss
        print(f'Validation Loss: {val_loss_list[-1]:.4f}')


# Save trained model
torch.save(model.state_dict(), 'model_seg.pth')

# Plot losses
plt.plot(train_loss_list, label='Training Loss')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.legend()
plt.show()

plt.plot(val_loss_list, label='Validation Loss')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.legend()
plt.show()
